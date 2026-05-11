"""Local HTTP API for Vietnamese voice cloning with VieNeu-TTS."""

from __future__ import annotations

import platform
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    platform.system = lambda: "Windows"
    platform.machine = lambda: "AMD64"
    platform.release = lambda: "10"
    platform.version = lambda: "10.0.19045"

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from vieneu import Vieneu


app = FastAPI(title="VieNeu Voice Clone API", version="1.0.0")
OUTPUT_DIR = ROOT_DIR / "outputs" / "api"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
USER_VOICES_FILE = ROOT_DIR / "user_voices.json"

_model_lock = threading.RLock()
_tts_instance: Any | None = None
_tts_key: tuple[Any, ...] | None = None


class PresetSynthesizeRequest(BaseModel):
    text: str
    preset_voice: str
    mode: str = "turbo"
    device: str = "cpu"
    backbone_device: str = "cpu"
    codec_device: str = "cpu"
    temperature: float | None = None
    top_k: int = 50
    max_chars: int = 256
    max_tokens: int = 384
    apply_watermark: bool = True


def _build_tts_key(
    mode: str,
    device: str,
    backbone_device: str,
    codec_device: str,
    emotion: str,
    hf_token: str | None,
) -> tuple[Any, ...]:
    return (mode, device, backbone_device, codec_device, emotion, hf_token or "")


def _get_tts(
    mode: str,
    device: str = "cpu",
    backbone_device: str = "cpu",
    codec_device: str = "cpu",
    emotion: str = "natural",
    hf_token: str | None = None,
):
    global _tts_instance, _tts_key

    key = _build_tts_key(mode, device, backbone_device, codec_device, emotion, hf_token)
    with _model_lock:
        if _tts_instance is not None and _tts_key == key and _tts_is_ready(_tts_instance, mode):
            return _tts_instance

        if _tts_instance is not None:
            close = getattr(_tts_instance, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        kwargs: dict[str, Any] = {}
        if hf_token:
            kwargs["hf_token"] = hf_token

        if mode == "standard":
            kwargs["backbone_device"] = backbone_device
            kwargs["codec_device"] = codec_device
            kwargs["emotion"] = emotion
        elif mode in {"turbo", "turbo_gpu"}:
            kwargs["device"] = device
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

        _tts_instance = Vieneu(mode=mode, **kwargs)
        load_voices = getattr(_tts_instance, "_load_voices_from_file", None)
        if callable(load_voices) and USER_VOICES_FILE.exists():
            load_voices(USER_VOICES_FILE)
        _tts_key = key
        return _tts_instance

def _tts_is_ready(tts: Any, mode: str) -> bool:
    if tts is None:
        return False
    if mode in {"turbo", "turbo_gpu"} and getattr(tts, "backbone", None) is None:
        return False
    if mode == "standard" and (
        getattr(tts, "backbone", None) is None or getattr(tts, "codec", None) is None
    ):
        return False
    return True


def _save_audio(tts: Any, audio: Any, prefix: str) -> Path:
    output_path = OUTPUT_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}.wav"
    tts.save(audio, output_path)
    return output_path


def _infer_clone(
    *,
    text: str,
    ref_audio_path: str,
    ref_text: str | None,
    mode: str,
    device: str,
    backbone_device: str,
    codec_device: str,
    emotion: str,
    temperature: float | None,
    top_k: int,
    max_chars: int,
    max_tokens: int,
    apply_watermark: bool,
    hf_token: str | None,
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    tts = _get_tts(
        mode=mode,
        device=device,
        backbone_device=backbone_device,
        codec_device=codec_device,
        emotion=emotion,
        hf_token=hf_token,
    )

    if mode == "standard":
        if not ref_text or not ref_text.strip():
            raise HTTPException(status_code=400, detail="ref_text is required for standard mode")
        voice = {"codes": tts.encode_reference(ref_audio_path), "text": ref_text.strip()}
    else:
        voice = tts.encode_reference(ref_audio_path)

    infer_kwargs = {
        "text": text.strip(),
        "voice": voice,
        "temperature": temperature if temperature is not None else (1.0 if mode == "standard" else 0.4),
        "top_k": top_k,
        "max_chars": max_chars,
        "apply_watermark": apply_watermark,
    }
    if mode != "standard":
        infer_kwargs["max_tokens"] = max_tokens

    return tts, tts.infer(**infer_kwargs)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/voices")
def list_voices(
    mode: str = Query(default="turbo"),
    device: str = Query(default="cpu"),
    backbone_device: str = Query(default="cpu"),
    codec_device: str = Query(default="cpu"),
):
    tts = _get_tts(
        mode=mode,
        device=device,
        backbone_device=backbone_device,
        codec_device=codec_device,
    )
    voices = tts.list_preset_voices()
    result: list[dict[str, str]] = []
    for item in voices:
        if isinstance(item, tuple) and len(item) == 2:
            desc, voice_id = item
            result.append({"id": str(voice_id), "name": str(desc)})
        else:
            result.append({"id": str(item), "name": str(item)})
    return {"voices": result}


@app.post("/clone")
async def clone_voice(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: str | None = Form(default=None),
    mode: str = Form(default="turbo"),
    device: str = Form(default="cpu"),
    backbone_device: str = Form(default="cpu"),
    codec_device: str = Form(default="cpu"),
    emotion: str = Form(default="natural"),
    temperature: float | None = Form(default=None),
    top_k: int = Form(default=50),
    max_chars: int = Form(default=256),
    max_tokens: int = Form(default=384),
    apply_watermark: bool = Form(default=True),
    hf_token: str | None = Form(default=None),
):
    suffix = Path(ref_audio.filename or "ref.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await ref_audio.read())
        tmp_path = tmp.name

    try:
        tts, audio = _infer_clone(
            text=text,
            ref_audio_path=tmp_path,
            ref_text=ref_text,
            mode=mode,
            device=device,
            backbone_device=backbone_device,
            codec_device=codec_device,
            emotion=emotion,
            temperature=temperature,
            top_k=top_k,
            max_chars=max_chars,
            max_tokens=max_tokens,
            apply_watermark=apply_watermark,
            hf_token=hf_token,
        )
        output_path = _save_audio(tts, audio, "clone")
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=output_path.name,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/clone-json")
async def clone_voice_json(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: str | None = Form(default=None),
    mode: str = Form(default="turbo"),
    device: str = Form(default="cpu"),
    backbone_device: str = Form(default="cpu"),
    codec_device: str = Form(default="cpu"),
    emotion: str = Form(default="natural"),
    temperature: float | None = Form(default=None),
    top_k: int = Form(default=50),
    max_chars: int = Form(default=256),
    max_tokens: int = Form(default=384),
    apply_watermark: bool = Form(default=True),
    hf_token: str | None = Form(default=None),
):
    suffix = Path(ref_audio.filename or "ref.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await ref_audio.read())
        tmp_path = tmp.name

    try:
        tts, audio = _infer_clone(
            text=text,
            ref_audio_path=tmp_path,
            ref_text=ref_text,
            mode=mode,
            device=device,
            backbone_device=backbone_device,
            codec_device=codec_device,
            emotion=emotion,
            temperature=temperature,
            top_k=top_k,
            max_chars=max_chars,
            max_tokens=max_tokens,
            apply_watermark=apply_watermark,
            hf_token=hf_token,
        )
        output_path = _save_audio(tts, audio, "clone")
        return {
            "status": "ok",
            "output_path": str(output_path),
            "mode": mode,
            "sample_rate": 24000,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/synthesize")
def synthesize_preset(req: PresetSynthesizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    try:
        with _model_lock:
            tts = _get_tts(
                mode=req.mode,
                device=req.device,
                backbone_device=req.backbone_device,
                codec_device=req.codec_device,
            )
            voice = tts.get_preset_voice(req.preset_voice)
            infer_kwargs = {
                "text": req.text.strip(),
                "voice": voice,
                "temperature": req.temperature if req.temperature is not None else (1.0 if req.mode == "standard" else 0.4),
                "top_k": req.top_k,
                "max_chars": req.max_chars,
                "apply_watermark": req.apply_watermark,
            }
            if req.mode != "standard":
                infer_kwargs["max_tokens"] = req.max_tokens
            audio = tts.infer(**infer_kwargs)
            output_path = _save_audio(tts, audio, "preset")
        return {
            "status": "ok",
            "output_path": str(output_path),
            "mode": req.mode,
            "sample_rate": 24000,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def main():
    host = "127.0.0.1"
    port = 8002
    print(f"Voice Clone API: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
