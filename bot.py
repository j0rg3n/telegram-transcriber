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

import json
import logging
import os
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

_here = Path(__file__).parent
_home = Path.home()
_ensure_on_path("ffmpeg", [_here])
_ensure_on_path("claude", [_home / ".local" / "bin", _home / "AppData" / "Local"])

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
_whisper: WhisperModel | None = None

def get_whisper() -> WhisperModel:
    global _whisper
    if _whisper is None:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            try:
                _whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
                log.info("Whisper loaded on GPU (float16)")
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

def transcribe(wav: Path) -> list[dict]:
    """Return [{start, end, text}, ...] at segment granularity."""
    model = get_whisper()
    segments, info = model.transcribe(str(wav), word_timestamps=False)
    result = []
    for seg in segments:
        result.append({
            "start": round(seg.start, 2),
            "end":   round(seg.end,   2),
            "text":  seg.text.strip(),
        })
    log.info(f"Transcribed {info.duration:.0f}s → {len(result)} segments")
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


def extract_highlights(segments: list[dict]) -> list[dict]:
    transcript = "\n".join(
        f"[{_fmt_mmss(s['start'])}] {s['text']}" for s in segments
    )
    prompt = HIGHLIGHT_PROMPT.format(transcript=transcript)

    proc = subprocess.Popen(
        [_claude_bin(), "--model", CLAUDE_MODEL, "-p", prompt],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1,
    )

    highlights = []
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            h = json.loads(line)
            if "time_s" in h and "note" in h:
                highlights.append(h)
                log.info(f"  highlight {h.get('mm_ss','?')}  {h['note']}")
        except json.JSONDecodeError:
            pass

    proc.wait()
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
            t0 = time.time()
            segments = transcribe(wav)
            elapsed = time.time() - t0
            log.info(f"Transcription took {elapsed:.1f}s")

            # Highlights
            await update_status(
                f"*{source_name}* — extracting highlights..."
            )
            highlights = extract_highlights(segments)

            # Format
            text = format_transcript(segments, highlights, source_name, duration)
            out.write_text(text, encoding="utf-8")

            # Reply with file
            hl_count  = len(highlights)
            seg_count = len(segments)
            await context.bot.send_document(
                chat_id=chat_id,
                document=InputFile(str(out), filename=out.name),
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


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set.")
        print("Get a token from @BotFather and set it before running.")
        sys.exit(1)

    # Pre-load Whisper so the first request isn't slow
    log.info("Loading Whisper model...")
    get_whisper()

    app = Application.builder().token(token).build()

    # Handle both channel posts and group/private messages
    audio_filter = filters.VOICE | filters.AUDIO | filters.Document.AUDIO
    app.add_handler(MessageHandler(audio_filter, handle_audio))
    app.add_handler(MessageHandler(
        filters.UpdateType.CHANNEL_POST & audio_filter, handle_audio
    ))

    log.info("Bot started. Listening for audio files...")
    app.run_polling(allowed_updates=["message", "channel_post"])


if __name__ == "__main__":
    main()
