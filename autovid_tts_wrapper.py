from __future__ import annotations

import os
import subprocess
import sys
import base64
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
CLI = ROOT / "apps" / "voice_clone_cli.py"


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def normalize_device(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"cuda", "gpu"}:
        return "cuda"
    return "cpu"


def env_b64(name: str) -> str:
    value = env(name)
    if not value:
        return ""
    try:
        return base64.b64decode(value).decode("utf-8").strip()
    except Exception:
        return ""


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: autovid_tts_wrapper.py <text-file> <output-wav>", file=sys.stderr)
        return 2

    text_file = sys.argv[1]
    output_file = sys.argv[2]
    mode = env("LOCAL_TTS_MODE", "turbo")
    device = normalize_device(env("LOCAL_TTS_DEVICE", "cpu"))
    speed = env("LOCAL_TTS_SPEED", "1.0")
    emotion = env("LOCAL_TTS_EMOTION", "natural")
    preset_voice = env_b64("LOCAL_TTS_PRESET_VOICE_B64") or env("LOCAL_TTS_PRESET_VOICE") or env("LOCAL_TTS_VOICE")
    ref_audio = env("LOCAL_TTS_REF_AUDIO", "examples\\audio_ref\\example.wav")
    ref_text = env("LOCAL_TTS_REF_TEXT")
    ref_text_file = env("LOCAL_TTS_REF_TEXT_FILE")
    api_base = env("LOCAL_TTS_API_BASE")
    model_name = env("LOCAL_TTS_MODEL_NAME") or env("LOCAL_TTS_MODEL", "pnnbao-ump/VieNeu-TTS")

    args = [
        str(PYTHON),
        str(CLI),
        "--mode",
        mode,
        "--device",
        device,
        "--backbone-device",
        device,
        "--codec-device",
        device,
        "--emotion",
        emotion,
        "--speed",
        speed,
        "--text-file",
        text_file,
        "--output",
        output_file,
        "--no-watermark",
    ]

    if mode == "remote":
        if api_base:
            args.extend(["--api-base", api_base])
        if model_name:
            args.extend(["--model-name", model_name])

    if preset_voice:
        args.extend(["--preset-voice", preset_voice])
    else:
        args.extend(["--ref-audio", ref_audio])
        if mode == "standard":
            if ref_text_file:
                args.extend(["--ref-text-file", ref_text_file])
            elif ref_text:
                args.extend(["--ref-text", ref_text])

    return subprocess.run(args, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
