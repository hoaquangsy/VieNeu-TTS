"""Focused Gradio UI for VieNeu-TTS voice cloning."""

from __future__ import annotations

import sys
import time
import os
import platform
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
if sys.platform == "win32":
    platform.system = lambda: "Windows"
    platform.machine = lambda: "AMD64"
    platform.release = lambda: "10"
    platform.version = lambda: "10.0.19045"

import gradio as gr

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vieneu import Vieneu

OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_PATH = ROOT_DIR / "voice_clone_ui.log"

_TTS: Any | None = None
_TTS_KEY: tuple[str, str, str, str] | None = None


def _log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)


def _get_tts(mode: str, device: str, backbone_device: str, codec_device: str) -> Any:
    global _TTS, _TTS_KEY

    key = (mode, device, backbone_device, codec_device)
    if _TTS is not None and _TTS_KEY == key:
        _log(f"Reusing loaded model: {mode}")
        return _TTS

    if _TTS is not None:
        _log("Closing previous model")
        close = getattr(_TTS, "close", None)
        if callable(close):
            close()

    _log(f"Loading model: mode={mode}, device={device}")
    if mode == "standard":
        _TTS = Vieneu(
            mode="standard",
            backbone_device=backbone_device,
            codec_device=codec_device,
        )
    elif mode == "turbo_gpu":
        _TTS = Vieneu(mode="turbo_gpu", device=device)
    else:
        _TTS = Vieneu(mode="turbo", device=device)

    _TTS_KEY = key
    _log("Model ready")
    return _TTS


def clone_voice(
    ref_audio: str | None,
    text: str,
    ref_text: str,
    mode: str,
    device: str,
    backbone_device: str,
    codec_device: str,
    temperature: float,
    top_k: int,
    max_chars: int,
    max_tokens: int,
    speed: str,
    disable_watermark: bool,
) -> tuple[str | None, str]:
    started = time.time()

    if not ref_audio:
        raise gr.Error("Hay tai len file audio mau.")
    if not text or not text.strip():
        raise gr.Error("Hay nhap noi dung can doc.")
    if mode == "standard" and not ref_text.strip():
        raise gr.Error("Standard mode can transcript cua audio mau.")

    text = text.strip()
    if mode != "standard" and len(text) <= 140:
        max_tokens = min(int(max_tokens), 384)

    _log(f"Request started: mode={mode}, text_chars={len(text)}, max_tokens={max_tokens}")
    tts = _get_tts(mode, device, backbone_device, codec_device)

    if mode == "standard":
        _log("Encoding standard reference audio")
        voice = {"codes": tts.encode_reference(ref_audio), "text": ref_text.strip()}
    else:
        _log("Encoding turbo reference audio")
        voice = tts.encode_reference(ref_audio)
    _log("Reference encoded")

    infer_kwargs = {
        "text": text,
        "voice": voice,
        "temperature": temperature,
        "top_k": int(top_k),
        "max_chars": int(max_chars),
        "apply_watermark": not disable_watermark,
    }
    if mode != "standard":
        infer_kwargs["max_tokens"] = int(max_tokens)

    _log("Synthesizing audio")
    audio = tts.infer(**infer_kwargs)
    _log("Audio synthesized")

    speed_factor = float(speed)
    if speed_factor != 1.0 and len(audio) > 0:
        _log(f"Applying speed factor: {speed_factor}")
        import librosa

        audio = librosa.effects.time_stretch(
            np.asarray(audio, dtype=np.float32),
            rate=speed_factor,
        )

    output_path = OUTPUT_DIR / f"ui_clone_{uuid4().hex[:10]}.wav"
    tts.save(audio, output_path)
    elapsed = time.time() - started
    return str(output_path), f"Da tao xong: {output_path.name} ({elapsed:.1f}s)"


def list_voices(mode: str, device: str, backbone_device: str, codec_device: str) -> str:
    tts = _get_tts(mode, device, backbone_device, codec_device)
    voices = tts.list_preset_voices()
    if not voices:
        return "Khong tim thay preset voice."
    return "\n".join(f"{voice_id}: {desc}" for desc, voice_id in voices)


def build_ui() -> gr.Blocks:
    css = """
    .gradio-container { max-width: 1120px !important; }
    textarea { font-size: 15px !important; }
    """

    with gr.Blocks(title="VieNeu-TTS Voice Clone", css=css) as demo:
        gr.Markdown("# VieNeu-TTS Voice Clone")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                ref_audio = gr.Audio(
                    label="Audio mau",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                text = gr.Textbox(
                    label="Noi dung can doc",
                    lines=7,
                    value="Xin chao, day la giong noi duoc clone bang VieNeu TTS.",
                )
                ref_text = gr.Textbox(
                    label="Transcript audio mau (chi can cho Standard mode)",
                    lines=3,
                )
                generate = gr.Button("Tao giong clone", variant="primary")

            with gr.Column(scale=1):
                output_audio = gr.Audio(label="Ket qua", type="filepath")
                status = gr.Textbox(label="Trang thai", interactive=False)
                with gr.Accordion("Cai dat", open=False):
                    mode = gr.Radio(
                        label="Mode",
                        choices=["turbo", "standard", "turbo_gpu"],
                        value="turbo",
                    )
                    with gr.Row():
                        device = gr.Dropdown(
                            label="Turbo device",
                            choices=["cpu", "cuda"],
                            value="cpu",
                        )
                        backbone_device = gr.Dropdown(
                            label="Standard backbone",
                            choices=["cpu", "cuda"],
                            value="cpu",
                        )
                        codec_device = gr.Dropdown(
                            label="Standard codec",
                            choices=["cpu", "cuda"],
                            value="cpu",
                        )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=1.5,
                        value=0.4,
                        step=0.05,
                    )
                    top_k = gr.Slider(
                        label="Top K",
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                    )
                    max_chars = gr.Slider(
                        label="Max chars per chunk",
                        minimum=80,
                        maximum=600,
                        value=256,
                        step=8,
                    )
                    max_tokens = gr.Slider(
                        label="Max output tokens",
                        minimum=128,
                        maximum=2048,
                        value=256,
                        step=64,
                    )
                    speed = gr.Radio(
                        label="Toc do doc",
                        choices=["1.0", "1.1", "1.2"],
                        value="1.0",
                    )
                    disable_watermark = gr.Checkbox(
                        label="Tat watermark audio",
                        value=False,
                    )
                    preset_button = gr.Button("Xem preset voices")
                    preset_output = gr.Textbox(
                        label="Preset voices",
                        lines=6,
                        interactive=False,
                    )

        generate.click(
            clone_voice,
            inputs=[
                ref_audio,
                text,
                ref_text,
                mode,
                device,
                backbone_device,
                codec_device,
                temperature,
                top_k,
                max_chars,
                max_tokens,
                speed,
                disable_watermark,
            ],
            outputs=[output_audio, status],
        )
        preset_button.click(
            list_voices,
            inputs=[mode, device, backbone_device, codec_device],
            outputs=preset_output,
        )

    return demo


def main() -> None:
    print("Starting VieNeu-TTS clone UI on http://127.0.0.1:7861", flush=True)
    demo = build_ui()
    print("UI built. Launching server...", flush=True)
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
