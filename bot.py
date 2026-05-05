#!/usr/bin/env python3
"""
TranscriptBot
=============
Telegram bot that transcribes audio files posted to channels or groups,
extracts highlights via Claude Haiku, and replies with an indexed transcript
as a .txt attachment.

Supported input: voice messages, audio files, and audio documents.

Configuration
  TELEGRAM_BOT_TOKEN  — bot token from @BotFather (required)

Usage
  python bot.py
"""

import asyncio
import json
import logging
import os

from dotenv import load_dotenv
load_dotenv()
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# ── PATH setup (same trick as MinimalYoga) ────────────────────────────────

def _ensure_on_path(exe: str, search_roots: list[Path]) -> None:
    try:
        if subprocess.run([exe, "--version" if exe == "ffmpeg" else "-v"],
                          capture_output=True).returncode == 0:
            return
    except FileNotFoundError:
        pass
    for root in search_roots:
        for found in root.rglob(f"{exe}.exe"):
            os.environ["PATH"] = str(found.parent) + os.pathsep + os.environ.get("PATH", "")
            return

_here = Path(__file__).resolve().parent
# Search ancestors too so ffmpeg placed in a parent (e.g. main worktree) is found
_search_roots = [_here] + list(_here.parents)[:4]
_home = Path.home()
_ensure_on_path("ffmpeg", _search_roots)
_ensure_on_path("claude", [_home / ".local" / "bin", _home / "AppData" / "Local"])

# Ensure CUDA 12 cuBLAS is on PATH for GPU inference (ctranslate2 needs it at runtime)
_cuda_bin = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin")
if _cuda_bin.exists() and str(_cuda_bin) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_cuda_bin) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from telegram import Update, InputFile
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# ── Logging ───────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Whisper model (loaded once at startup) ────────────────────────────────

WHISPER_MODEL_SIZE = "small"
# Set WHISPER_DEVICE=cuda in environment to enable GPU (CPU is default for safety)
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu").lower()
_whisper: WhisperModel | None = None

def get_whisper() -> WhisperModel:
    global _whisper
    if _whisper is None:
        import ctranslate2
        if WHISPER_DEVICE == "cuda" and ctranslate2.get_cuda_device_count() > 0:
            try:
                _whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="int8_float16")
                log.info("Whisper loaded on GPU (int8_float16)")
                return _whisper
            except RuntimeError as e:
                log.warning(f"GPU load failed ({e}), falling back to CPU")
        _whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        log.info("Whisper loaded on CPU (int8)")
    return _whisper


# ── Step 1: convert any audio format to 16kHz mono WAV ───────────────────

def to_wav(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-vn", "-ar", "16000", "-ac", "1", "-f", "wav", str(dst)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


# ── Step 2: transcribe, returning segments with timestamps ────────────────

def transcribe(wav: Path, progress_cb=None) -> list[dict]:
    """Return [{start, end, text}, ...] at segment granularity.

    progress_cb(current_s, total_s) is called after each segment if provided.
    """
    model = get_whisper()
    segments, info = model.transcribe(str(wav), word_timestamps=False)
    result = []
    for seg in segments:
        result.append({
            "start": round(seg.start, 2),
            "end":   round(seg.end,   2),
            "text":  seg.text.strip(),
        })
        if progress_cb:
            progress_cb(seg.end, info.duration)
    log.info(f"Transcribed {info.duration:.0f}s -> {len(result)} segments")
    return result


async def transcribe_async(
    wav: Path,
    duration_s: float,
    update_status,
    source_name: str,
) -> list[dict]:
    """Run transcribe() in a thread executor, posting progress every ~5 s."""
    loop = asyncio.get_running_loop()
    last_update = [0.0]   # mutable cell for the closure
    PROGRESS_INTERVAL = 5.0

    def _progress(current_s: float, total_s: float) -> None:
        now = time.time()
        if now - last_update[0] >= PROGRESS_INTERVAL:
            last_update[0] = now
            pct = int(min(current_s, total_s) / total_s * 100) if total_s > 0 else 0
            msg = (
                f"*{source_name}* — transcribing "
                f"{_fmt_mmss(current_s)} / {_fmt_mmss(total_s)} ({pct}%)..."
            )
            log.info(f"Transcription progress: {_fmt_mmss(current_s)} / {_fmt_mmss(total_s)} ({pct}%)")
            fut = asyncio.run_coroutine_threadsafe(update_status(msg), loop)
            fut.add_done_callback(
                lambda f: log.warning("Progress update failed: %s", f.exception())
                if f.exception() else None
            )

    t0 = time.time()
    try:
        result = await loop.run_in_executor(None, lambda: transcribe(wav, _progress))
    except Exception as e:
        log.warning("GPU transcription failed (%s), retrying on CPU", e)
        global _whisper
        _whisper = None  # force CPU reload on next get_whisper()
        os.environ["WHISPER_DEVICE"] = "cpu"
        await update_status(f"*{source_name}* — GPU error, retrying on CPU...")
        result = await loop.run_in_executor(None, lambda: transcribe(wav, _progress))
    log.info(f"Transcription took {time.time() - t0:.1f}s")
    return result


# ── Step 3: highlight extraction via Claude Haiku ─────────────────────────

CLAUDE_MODEL = "claude-haiku-4-5"

HIGHLIGHT_PROMPT = """\
You are analyzing a transcript to extract an index of highlights.

The transcript lists segments as [MM:SS] text.

Work through the transcript and emit a JSON object on its own line for each highlight:
  {{"time_s": <float seconds>, "mm_ss": "<MM:SS>", "type": "topic_change"|"important"|"flagged", "note": "<one concise line>"}}

INCLUDE as highlights
- Significant topic changes or new sections
- Clear turning points in the discussion or narrative
- Anything the speaker explicitly marks as important, key, remember, note, highlight,
  pay attention, this is critical, write this down, etc.

EXCLUDE
- Filler, repeated content, minor digressions

Output ONLY the JSON lines, one per highlight, nothing else.

Transcript:
{transcript}"""


def _claude_bin() -> str:
    for candidate in ["claude", str(_home / ".local" / "bin" / "claude")]:
        try:
            if subprocess.run([candidate, "--version"], capture_output=True).returncode == 0:
                return candidate
        except FileNotFoundError:
            pass
    raise FileNotFoundError("claude CLI not found")


def _fmt_mmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


async def extract_highlights(
    segments: list[dict],
    update_status=None,
    source_name: str = "audio",
) -> list[dict]:
    transcript = "\n".join(
        f"[{_fmt_mmss(s['start'])}] {s['text']}" for s in segments
    )
    prompt = HIGHLIGHT_PROMPT.format(transcript=transcript)

    proc = await asyncio.create_subprocess_exec(
        _claude_bin(), "--model", CLAUDE_MODEL, "-p", prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    highlights = []
    last_status_update = 0.0

    async for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            h = json.loads(line)
            if "time_s" in h and "note" in h:
                highlights.append(h)
                log.info(f"  highlight {h.get('mm_ss','?')}  {h['note']}")
                now = time.time()
                if update_status and (now - last_status_update) >= 1.0:
                    last_status_update = now
                    found = len(highlights)
                    await update_status(
                        f"*{source_name}* — extracting highlights "
                        f"({found} found so far...)"
                    )
        except json.JSONDecodeError:
            pass

    await proc.wait()
    log.info(f"Extracted {len(highlights)} highlights")
    return highlights


# ── Step 4: format the indexed transcript ────────────────────────────────

TYPE_LABELS = {
    "topic_change": "Topic",
    "important":    "★ Important",
    "flagged":      "★ Flagged",
}

def format_transcript(
    segments: list[dict],
    highlights: list[dict],
    source_name: str,
    duration_s: float,
) -> str:
    lines = []

    # Header
    lines += [
        "INDEXED TRANSCRIPT",
        "=" * 60,
        f"Source:    {source_name}",
        f"Duration:  {_fmt_mmss(duration_s)}",
        f"Date:      {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # Index
    lines += ["INDEX", "-" * 40]
    if highlights:
        for h in sorted(highlights, key=lambda x: x["time_s"]):
            label = TYPE_LABELS.get(h.get("type", ""), "Note")
            lines.append(f"  {h.get('mm_ss', _fmt_mmss(h['time_s'])):>6}  [{label}]  {h['note']}")
    else:
        lines.append("  (no highlights detected)")
    lines += ["", ""]

    # Transcript
    lines += ["TRANSCRIPT", "-" * 40]
    # Build a set of highlighted timestamps for inline markers
    hl_by_time = {h["time_s"]: h for h in highlights}
    for seg in segments:
        # Insert highlight marker before the segment if one falls here
        closest = min(hl_by_time, key=lambda t: abs(t - seg["start"]), default=None)
        if closest is not None and abs(closest - seg["start"]) < 2.0:
            h = hl_by_time.pop(closest)
            label = TYPE_LABELS.get(h.get("type", ""), "Note")
            lines.append(f"\n  ── [{label}] {h['note']}")
        lines.append(f"[{_fmt_mmss(seg['start'])}]  {seg['text']}")

    return "\n".join(lines) + "\n"


# ── Telegram bot ──────────────────────────────────────────────────────────

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Works for both channel posts and group/private messages
    msg = update.channel_post or update.message
    if msg is None:
        return

    # Pick the file object — voice, audio, or document
    tg_file = None
    source_name = "audio"
    if msg.voice:
        tg_file = msg.voice
        source_name = "voice_message"
    elif msg.audio:
        tg_file = msg.audio
        source_name = msg.audio.file_name or "audio"
    elif msg.document and msg.document.mime_type and msg.document.mime_type.startswith("audio/"):
        tg_file = msg.document
        source_name = msg.document.file_name or "audio"

    if tg_file is None:
        return

    chat_id = msg.chat_id
    log.info(f"Audio received in chat {chat_id}: {source_name} ({tg_file.file_size} bytes)")

    # Status message
    status = await context.bot.send_message(
        chat_id=chat_id,
        text=f"Received *{source_name}* — transcribing...",
        parse_mode="Markdown",
    )

    async def update_status(text: str) -> None:
        try:
            await status.edit_text(text, parse_mode="Markdown")
        except Exception:
            pass

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw  = tmp / f"input{Path(source_name).suffix or '.ogg'}"
            wav  = tmp / "audio.wav"
            out  = tmp / f"{Path(source_name).stem}_transcript.txt"

            # Download
            await update_status(f"*{source_name}* — downloading...")
            file = await context.bot.get_file(tg_file.file_id)
            await file.download_to_drive(str(raw))

            # Convert
            await update_status(f"*{source_name}* — converting audio...")
            to_wav(raw, wav)
            duration = sf.info(str(wav)).duration

            # Transcribe
            await update_status(
                f"*{source_name}* — transcribing {_fmt_mmss(duration)}..."
            )
            segments = await transcribe_async(wav, duration, update_status, source_name)

            # Highlights
            await update_status(
                f"*{source_name}* — extracting highlights..."
            )
            highlights = await extract_highlights(
                segments, update_status=update_status, source_name=source_name
            )

            # Format
            text = format_transcript(segments, highlights, source_name, duration)
            out.write_text(text, encoding="utf-8")

            # Reply with file
            hl_count  = len(highlights)
            seg_count = len(segments)
            with out.open("rb") as fh:
                await context.bot.send_document(
                    chat_id=chat_id,
                    document=InputFile(fh, filename=out.name),
                    caption=(
                        f"{_fmt_mmss(duration)} audio  •  "
                        f"{seg_count} segments  •  "
                        f"{hl_count} highlight{'s' if hl_count != 1 else ''}"
                    ),
                )
            await status.delete()

    except Exception as e:
        log.exception("Processing failed")
        await update_status(f"Error: `{e}`")


def _has_audio(update) -> bool:
    msg = update.message or update.channel_post
    if not msg:
        return False
    return bool(msg.voice or msg.audio or (msg.document and msg.document.mime_type and msg.document.mime_type.startswith("audio/")))


async def _post_init(app: Application) -> None:
    pending = await app.bot.get_updates(timeout=0, limit=100,
                                        allowed_updates=["message", "channel_post"])
    if pending:
        audio_count = sum(1 for u in pending if _has_audio(u))
        total = len(pending)
        log.info(
            "Catch-up: %d pending update(s) in queue%s",
            total,
            f", {audio_count} with audio" if audio_count else " (no audio)",
        )


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set.")
        print("Get a token from @BotFather and set it before running.")
        sys.exit(1)

    # Pre-load Whisper so the first request isn't slow
    log.info("Loading Whisper model...")
    get_whisper()

    app = Application.builder().token(token).post_init(_post_init).build()

    # Handle both channel posts and group/private messages
    audio_filter = filters.VOICE | filters.AUDIO | filters.Document.AUDIO
    app.add_handler(MessageHandler(audio_filter, handle_audio))
    app.add_handler(MessageHandler(
        filters.UpdateType.CHANNEL_POST & audio_filter, handle_audio
    ))

    log.info("Bot started. Listening for audio files...")
    app.run_polling(allowed_updates=["message", "channel_post"], drop_pending_updates=False)


if __name__ == "__main__":
    main()
