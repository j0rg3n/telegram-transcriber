#!/usr/bin/env python3
"""
CUDA transcription test harness.
Iterates on loading and running Whisper on GPU to diagnose crashes.

Usage:
    python test_cuda.py [path/to/audio.wav]
"""

import os
import sys
import time
import traceback
from pathlib import Path

# ── CUDA PATH setup (mirrors bot.py) ─────────────────────────────────────────
_cuda_bin = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin")
if _cuda_bin.exists():
    os.environ["PATH"] = str(_cuda_bin) + os.pathsep + os.environ.get("PATH", "")
    print(f"[+] Added to PATH: {_cuda_bin}")
else:
    print(f"[!] CUDA bin not found at {_cuda_bin}")

# ── Optional: dump relevant DLLs found on PATH ───────────────────────────────
def find_on_path(dll: str):
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = Path(d) / dll
        if p.exists():
            return p
    return None

for dll in ["cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll", "cudnn64_9.dll"]:
    loc = find_on_path(dll)
    print(f"  {'[+]' if loc else '[-]'} {dll}: {loc or 'NOT FOUND'}")

print()

# ── Import ctranslate2 / faster-whisper ──────────────────────────────────────
print("[*] Importing ctranslate2...")
try:
    import ctranslate2
    print(f"    ctranslate2 version : {ctranslate2.__version__}")
    print(f"    CUDA device count   : {ctranslate2.get_cuda_device_count()}")
except Exception as e:
    print(f"[!] ctranslate2 import failed: {e}")
    sys.exit(1)

from faster_whisper import WhisperModel

AUDIO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\jorge\Downloads\Short test.wav")
MODEL_SIZE = "small"
RUNS = 5

if not AUDIO.exists():
    print(f"[!] Audio file not found: {AUDIO}")
    sys.exit(1)

print(f"[*] Audio file : {AUDIO}")
print(f"[*] Model size : {MODEL_SIZE}")
print(f"[*] Runs       : {RUNS}")
print()

# ── Load model ───────────────────────────────────────────────────────────────
print("[*] Loading Whisper on GPU (int8_float16)...")
try:
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="int8_float16")
    print("[+] Model loaded on GPU\n")
except Exception as e:
    print(f"[!] GPU model load failed: {e}")
    print("    Trying CPU fallback...")
    try:
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        print("[+] Model loaded on CPU (float16 GPU load failed — GPU won't be tested)\n")
    except Exception as e2:
        print(f"[!] CPU load also failed: {e2}")
        sys.exit(1)

# ── Transcription loop ───────────────────────────────────────────────────────
for i in range(1, RUNS + 1):
    print(f"[*] Run {i}/{RUNS} ...", end=" ", flush=True)
    t0 = time.time()
    try:
        segments, info = model.transcribe(str(AUDIO), word_timestamps=False)
        text = " ".join(s.text.strip() for s in segments)
        elapsed = time.time() - t0
        print(f"OK  ({elapsed:.2f}s)  ->  {repr(text[:80])}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED ({elapsed:.2f}s)")
        traceback.print_exc()
        print()

print("\n[*] Done.")
