"""Command-line voice cloning tool for VieNeu-TTS."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

if sys.platform == "win32":
    platform.system = lambda: "Windows"
    platform.machine = lambda: "AMD64"
    platform.release = lambda: "10"
    platform.version = lambda: "10.0.19045"

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vieneu import Vieneu


def _read_text(value: str | None, file_path: str | None, label: str) -> str | None:
    if value and file_path:
        raise ValueError(f"Use either --{label} or --{label}-file, not both.")
    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip()
    return value.strip() if value else None


def _build_tts(args: argparse.Namespace) -> Any:
    kwargs: dict[str, Any] = {}
    if args.hf_token:
        kwargs["hf_token"] = args.hf_token

    if args.mode == "standard":
        kwargs["backbone_device"] = args.backbone_device
        kwargs["codec_device"] = args.codec_device
        kwargs["emotion"] = args.emotion
    elif args.mode in {"turbo", "turbo_gpu"}:
        kwargs["device"] = args.device
    elif args.mode == "remote":
        if not args.api_base:
            raise ValueError("--api-base is required when --mode remote.")
        kwargs["api_base"] = args.api_base
        kwargs["model_name"] = args.model_name

    return Vieneu(mode=args.mode, **kwargs)


def _resolve_voice(tts: Any, args: argparse.Namespace, ref_text: str | None) -> Any:
    if args.preset_voice:
        return tts.get_preset_voice(args.preset_voice)

    if not args.ref_audio:
        return None

    if args.mode == "standard":
        if not ref_text:
            raise ValueError(
                "--ref-text or --ref-text-file is required with --mode standard."
            )
        return {"codes": tts.encode_reference(args.ref_audio), "text": ref_text}

    return tts.encode_reference(args.ref_audio)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clone a Vietnamese voice with VieNeu-TTS and export a WAV file.",
    )
    parser.add_argument("--text", help="Text to synthesize.")
    parser.add_argument("--text-file", help="UTF-8 text file to synthesize.")
    parser.add_argument("--ref-audio", help="Reference voice audio file, ideally 3-10 seconds.")
    parser.add_argument("--ref-text", help="Transcript of the reference audio. Required for standard mode.")
    parser.add_argument("--ref-text-file", help="UTF-8 file containing the reference transcript.")
    parser.add_argument("--preset-voice", help="Use a built-in voice id instead of --ref-audio.")
    parser.add_argument("--list-voices", action="store_true", help="List available preset voices and exit.")
    parser.add_argument("--output", default="outputs/cloned_voice.wav", help="Output WAV path.")
    parser.add_argument(
        "--mode",
        default="turbo",
        choices=["turbo", "turbo_gpu", "standard", "remote"],
        help="Inference backend. Turbo is easiest for zero-shot cloning.",
    )
    parser.add_argument("--device", default="cpu", help="Device for turbo/turbo_gpu, for example cpu or cuda.")
    parser.add_argument("--backbone-device", default="cpu", help="Backbone device for standard mode.")
    parser.add_argument("--codec-device", default="cpu", help="Codec device for standard mode.")
    parser.add_argument("--api-base", help="Remote API base URL for remote mode.")
    parser.add_argument("--model-name", default="pnnbao-ump/VieNeu-TTS", help="Remote model name.")
    parser.add_argument("--hf-token", help="Hugging Face token if needed.")
    parser.add_argument("--emotion", default="natural", choices=["natural", "storytelling"])
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-chars", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=384, help="Maximum output tokens for turbo modes.")
    parser.add_argument("--speed", type=float, choices=[1.0, 1.1, 1.2], default=1.0, help="Post-process speaking speed.")
    parser.add_argument("--no-watermark", action="store_true", help="Disable audio watermarking.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        text = _read_text(args.text, args.text_file, "text")
        ref_text = _read_text(args.ref_text, args.ref_text_file, "ref-text")

        tts = _build_tts(args)

        if args.list_voices:
            voices = tts.list_preset_voices()
            if not voices:
                print("No preset voices found.")
                return 0
            for desc, voice_id in voices:
                print(f"{voice_id}\t{desc}")
            return 0

        if not text:
            raise ValueError("Provide --text or --text-file.")
        if not args.ref_audio and not args.preset_voice:
            raise ValueError("Provide --ref-audio for cloning or --preset-voice for a built-in voice.")

        voice = _resolve_voice(tts, args, ref_text)
        infer_kwargs = {
            "text": text,
            "temperature": args.temperature if args.temperature is not None else (1.0 if args.mode == "standard" else 0.4),
            "top_k": args.top_k,
            "max_chars": args.max_chars,
            "apply_watermark": not args.no_watermark,
        }
        if args.mode != "standard":
            infer_kwargs["max_tokens"] = args.max_tokens

        infer_kwargs["voice"] = voice

        audio = tts.infer(**infer_kwargs)
        if args.speed != 1.0 and len(audio) > 0:
            import librosa

            audio = librosa.effects.time_stretch(
                np.asarray(audio, dtype=np.float32),
                rate=args.speed,
            )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tts.save(audio, output_path)

        close = getattr(tts, "close", None)
        if callable(close):
            close()

        print(f"Saved: {output_path.resolve()}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
