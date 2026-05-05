# TranscriptBot

Telegram bot that transcribes audio files posted to channels or groups,
extracts highlights, and replies with an indexed transcript as a `.txt`
attachment.

## What it does

1. Listens for audio in any channel or group it is a member of
2. Transcribes with faster-whisper locally on GPU (falls back to CPU)
3. Extracts highlights via Claude Haiku — topic changes, turning points,
   and anything the speaker explicitly flags as important
4. Replies with a `.txt` file containing an index at the top and the full
   transcript below, with highlight markers inline

Supported input: voice messages, audio files, and audio documents.

## Prerequisites

### CUDA 12 (optional, for GPU transcription)

ctranslate2/faster-whisper requires CUDA 12 specifically — CUDA 13+ is not
yet supported. Install the CUDA 12 toolkit alongside any newer driver:

- Download: https://developer.nvidia.com/cuda-12-6-0-download-archive
- Install with default options; it coexists with other CUDA versions

Skip this to run on CPU (~1–1.5× realtime for the `small` model).

### Python 3.12

```powershell
winget install Python.Python.3.12
```

### ffmpeg

```powershell
winget install Gyan.FFmpeg
```

Alternatively, place a local ffmpeg build anywhere inside the project
folder — the bot will find it automatically.

### Claude CLI

Used for highlight extraction. Install and authenticate once:

- Install: https://claude.ai/claude-code
- Authenticate: `claude login`

The bot looks for `claude` on PATH and also checks `~/.local/bin/claude`.

### Telegram bot token

Create a bot via [@BotFather](https://t.me/BotFather) and copy the token.

For the bot to see channel posts, add it as an **administrator** of the
channel (read-only permissions are enough).

## Setup

```powershell
cd C:\path\to\TranscriptBot
python -m pip install -r requirements.txt
```

## Running

```powershell
$env:TELEGRAM_BOT_TOKEN = "your-token-here"
python bot.py
```

To persist the token across sessions, set it as a user environment variable:

```powershell
[System.Environment]::SetEnvironmentVariable("TELEGRAM_BOT_TOKEN", "your-token-here", "User")
```

Then restart your terminal.

## Output format

```
INDEXED TRANSCRIPT
============================================================
Source:    recording.mp3
Duration:  45:12
Date:      2026-05-05 23:00

INDEX
----------------------------------------
   2:05  [Topic]        Shifts to implementation details
   8:40  [★ Important]  Speaker flags this as critical
  23:11  [Topic]        Q&A section begins

TRANSCRIPT
----------------------------------------
[0:00]  Hello and welcome...

  ── [Topic] Shifts to implementation details
[2:05]  So let's talk about how this actually works...
```

## Whisper model

The model size is set via `WHISPER_MODEL_SIZE` at the top of `bot.py`:

| Model   | Speed (GPU) | Notes                                  |
|---------|-------------|----------------------------------------|
| `base`  | very fast   | Good for clear studio audio            |
| `small` | fast        | Default, good balance                  |
| `medium`| moderate    | Better for accented or noisy audio     |
