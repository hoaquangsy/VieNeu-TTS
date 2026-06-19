"""Local HTTP API for Vietnamese voice cloning with VieNeu-TTS."""

from __future__ import annotations

import platform
import json
import os
import math
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from vieneu_utils.core_utils import join_audio_chunks, split_text_into_chunks
from vieneu_utils.phonemize_text import phonemize_text_with_emotions


app = FastAPI(title="VieNeu Voice Clone API", version="1.0.0")
OUTPUT_DIR = ROOT_DIR / "outputs" / "api"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
USER_VOICES_FILE = ROOT_DIR / "user_voices.json"
DEFAULT_API_MODEL = "vieneu-v3-turbo-gpu"
DEFAULT_API_MODE = "v3_turbo_gpu"
DEFAULT_API_DEVICE = "cuda"
DEFAULT_REF_AUDIO_REL = "examples/audio_ref/example_ngoc_huyen.wav"
DEFAULT_REF_AUDIO_PATH = ROOT_DIR / DEFAULT_REF_AUDIO_REL
DEFAULT_REF_VOICE_ID = "chill"
NGUYEN_HUYEN_TRANG_VOICE_ID = "Nguyễn Huyền Trang"
NGUYEN_HUYEN_TRANG_VOICE_LABEL = "Nguyễn Huyền Trang"
DEFAULT_V3_MIN_CHUNK_CHARS = 40
DEFAULT_V3_MAX_CHUNK_CHARS = 220
DEFAULT_V3_CHUNK_PACKING_STRATEGY = "legacy"
DEFAULT_V3_BALANCED_MAX_CHARS = 232
DEFAULT_V3_CHUNK_AUDIO_RETRIES = 2
DEFAULT_V3_CHUNK_SILENCE_DB = -35.0
DEFAULT_REF_VOICE_LABEL = "chill - Ngọc Huyền"
DEFAULT_TTS_CHUNK_CONCURRENCY = 1
MAX_TTS_CHUNK_CONCURRENCY = 2
DEFAULT_V3_BATCH_SIZE = 19
MAX_V3_BATCH_SIZE = 32
DEFAULT_V3_BATCH_RETRY_SILENCE_SECONDS = 1.0
DEFAULT_SILENCE_SOFT_WARN_SECONDS = 1.0
DEFAULT_SILENCE_HARD_RETRY_SECONDS = 5.0
DEFAULT_SILENCE_SEVERE_SECONDS = 10.0
DEFAULT_VOLUME_LOCAL_DROP_DB = 4.5
DEFAULT_VOLUME_GLOBAL_DROP_DB = 8.0
DEFAULT_VOLUME_OUTLIER_MIN_SECONDS = 2.0
DEFAULT_RETRY_VOLUME_OUTLIER = 1
DEFAULT_V3_TURBO_TEMPERATURE = 0.8
DEFAULT_V3_TURBO_TOP_K = 25
DEFAULT_V3_TURBO_TOP_P = 0.95
DEFAULT_V3_TURBO_REPETITION_PENALTY = 1.2
DEFAULT_VOLUME_LOCAL_WINDOW = 1
API_MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "vieneu-v2-cpu": {
        "id": "vieneu-v2-cpu",
        "name": "VieNeu v2 CPU - legacy/dev",
        "label": "VieNeu v2 CPU - legacy/dev",
        "mode": "standard",
        "backbone_repo": "pnnbao-ump/VieNeu-TTS-v2",
        "gguf_filename": "VieNeu-TTS-v2-Q4-K-M.gguf",
        "codec_repo": "neuphonic/neucodec-onnx-decoder-int8",
        "device": "cpu",
        "backbone_device": "cpu",
        "codec_device": "cpu",
        "requires_ref_text_for_clone": True,
        "voice_type": "preset_or_ref_audio",
        "default": False,
    },
    "vieneu-v2-gpu": {
        "id": "vieneu-v2-gpu",
        "name": "VieNeu v2 GPU - legacy/dev",
        "label": "VieNeu v2 GPU - legacy/dev",
        "mode": "standard",
        "backbone_repo": "pnnbao-ump/VieNeu-TTS-v2",
        "gguf_filename": "",
        "codec_repo": "neuphonic/distill-neucodec",
        "device": "cuda",
        "backbone_device": "cuda",
        "codec_device": "cuda",
        "requires_ref_text_for_clone": True,
        "voice_type": "preset_or_ref_audio",
        "default": False,
    },
    "vieneu-v3-turbo-gpu": {
        "id": "vieneu-v3-turbo-gpu",
        "name": "VieNeu v3 Turbo GPU - recommended",
        "label": "VieNeu v3 Turbo GPU - recommended",
        "mode": "v3turbo",
        "api_mode": DEFAULT_API_MODE,
        "backbone_repo": "pnnbao-ump/VieNeu-TTS-v3-Turbo",
        "moss_tokenizer": "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        "device": "cuda",
        "backbone_device": "cuda",
        "codec_device": "cuda",
        "dtype": "auto",
        "requires_ref_text_for_clone": False,
        "requires_ref_audio": True,
        "voice_type": "ref_audio",
        "default_ref_audio": DEFAULT_REF_AUDIO_REL,
        "default_voice_id": DEFAULT_REF_VOICE_ID,
        "default_voice_label": DEFAULT_REF_VOICE_LABEL,
        "default": True,
    },
    "vieneu-v2-turbo-cpu": {
        "id": "vieneu-v2-turbo-cpu",
        "name": "VieNeu v2 Turbo CPU - legacy/dev",
        "label": "VieNeu v2 Turbo CPU - legacy/dev",
        "mode": "turbo",
        "backbone_repo": "pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF",
        "decoder_repo": "pnnbao-ump/VieNeu-Codec",
        "device": "cpu",
        "backbone_device": "cpu",
        "codec_device": "cpu",
        "requires_ref_text_for_clone": False,
        "voice_type": "preset_or_ref_audio",
        "default": False,
    },
}

_model_lock = threading.RLock()
_tts_instance: Any | None = None
_tts_key: tuple[Any, ...] | None = None
_whisper_model_lock = threading.RLock()
_whisper_model_cache: dict[tuple[Any, ...], Any] = {}
_v3_ref_codes_lock = threading.RLock()
_v3_ref_codes_cache: dict[tuple[str, int, int], np.ndarray] = {}


class PresetSynthesizeRequest(BaseModel):
    text: str
    preset_voice: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    model: str | None = None
    mode: str | None = None
    device: str = "cpu"
    backbone_device: str = "cpu"
    codec_device: str = "cpu"
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    max_chars: int = 256
    max_tokens: int = 384
    apply_watermark: bool = True
    hf_token: str | None = None

class AlignedSegmentRequest(BaseModel):
    id: str
    text: str

class AlignedSynthesizeRequest(BaseModel):
    request_id: str | None = None
    preset_voice: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    model: str | None = None
    mode: str | None = None
    device: str = "cpu"
    backbone_device: str = "cpu"
    codec_device: str = "cpu"
    speed: float = 1.0
    emotion: str = "storytelling"
    timeline_mode: str = "full_then_align"
    alignment_method: str = "whisper"
    segments: list[AlignedSegmentRequest]
    min_chunk_chars: int | None = None
    max_chunk_chars: int | None = None
    chunk_packing_strategy: str | None = None
    chunk_audio_retries: int | None = None
    v3_batch_size: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    max_chars: int = 256
    max_tokens: int = 384
    apply_watermark: bool = True
    hf_token: str | None = None
    max_full_align_words: int = 1300


def _build_tts_key(
    mode: str,
    device: str,
    backbone_device: str,
    codec_device: str,
    emotion: str,
    hf_token: str | None,
    backbone_repo: str | None,
    codec_repo: str | None,
    decoder_repo: str | None,
    gguf_filename: str | None,
    moss_tokenizer: str | None,
    dtype: str | None,
    cpu_threads_key: str | None,
) -> tuple[Any, ...]:
    return (
        mode,
        device,
        backbone_device,
        codec_device,
        emotion,
        hf_token or "",
        backbone_repo or "",
        codec_repo or "",
        decoder_repo or "",
        gguf_filename or "",
        moss_tokenizer or "",
        dtype or "",
        cpu_threads_key or "",
    )


def _resolve_hf_token(hf_token: str | None) -> str | None:
    value = (hf_token or "").strip()
    if value:
        return value
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return None

def _normalize_api_mode(mode: str | None) -> str | None:
    value = (mode or "").strip().lower()
    if not value:
        return None
    if value in {"v3_turbo_gpu", "v3turbo_gpu", "v3-turbo-gpu", "v3turbo"}:
        return "v3turbo"
    return value

def _public_mode(mode: str | None) -> str:
    if mode == "v3turbo":
        return DEFAULT_API_MODE
    return str(mode or "")

def _resolve_model_config(
    model: str | None,
    mode: str | None,
    device: str,
    backbone_device: str,
    codec_device: str,
) -> dict[str, Any]:
    mode = _normalize_api_mode(mode)
    model_id = (model or "").strip()
    if not model_id:
        if mode == "v3turbo":
            model_id = "vieneu-v3-turbo-gpu"
        elif mode == "turbo" or mode == "turbo_gpu":
            model_id = "vieneu-v2-turbo-cpu"
        else:
            model_id = DEFAULT_API_MODEL

    if model_id not in API_MODEL_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {model_id}. Use /models to list supported models.",
        )

    cfg = dict(API_MODEL_PRESETS[model_id])
    if mode and not model:
        cfg["mode"] = mode
    if device and device.lower() != "auto":
        cfg["device"] = device
    if backbone_device and backbone_device.lower() != "auto":
        cfg["backbone_device"] = backbone_device
    if codec_device and codec_device.lower() != "auto":
        cfg["codec_device"] = codec_device
    return cfg

def _normalize_words(text: str) -> list[str]:
    value = str(text or "").lower()
    return re.findall(r"[\wÀ-ỹ]+", value, flags=re.UNICODE)

def _word_count(text: str) -> int:
    return len(_normalize_words(text))

def _safe_request_id(value: str | None) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    safe = safe.strip("._-")
    if not safe:
        return uuid.uuid4().hex[:8]
    return safe[:120]

def _validate_aligned_segments(segments: list[AlignedSegmentRequest]) -> list[dict[str, Any]]:
    if not segments:
        raise HTTPException(status_code=422, detail="segments is required")

    dangling_words = {
        "của", "về", "và", "hoặc", "từ", "đến", "trong", "với",
        "mở", "áp", "ranh",
    }
    clean_segments: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for index, segment in enumerate(segments):
        segment_id = str(segment.id or "").strip()
        text = re.sub(r"\s+", " ", str(segment.text or "").strip())

        if not segment_id:
            errors.append({"index": str(index), "field": "id", "error": "id is required"})
        elif segment_id in seen_ids:
            errors.append({"id": segment_id, "field": "id", "error": "duplicate id"})
        else:
            seen_ids.add(segment_id)

        if not text:
            errors.append({"id": segment_id, "field": "text", "error": "text is empty"})
            continue

        if re.search(r"[,，]\s*$", text):
            errors.append({"id": segment_id, "field": "text", "error": "segment must not end with a comma"})

        words = _normalize_words(text)
        last_word = words[-1] if words else ""
        if last_word in dangling_words:
            errors.append({"id": segment_id, "field": "text", "error": f"segment ends with dangling word: {last_word}"})

        if not re.search(r"[.!?…]\s*$", text):
            errors.append({"id": segment_id, "field": "text", "error": "segment appears to be cut mid-sentence; end with . ! ? or …"})

        clean_segments.append(
            {
                "id": segment_id,
                "text": text,
                "word_count": len(words),
            }
        )

    if errors:
        raise HTTPException(status_code=422, detail={"message": "Invalid aligned TTS segments", "errors": errors})
    return clean_segments

def _build_full_narration_text(segments: list[dict[str, Any]]) -> str:
    return "\n\n".join(segment["text"] for segment in segments)

_SENTENCE_UNIT_RE = re.compile(r"\S[^.!?\u2026]*(?:[.!?\u2026]+|$)", flags=re.UNICODE)
_ENGLISH_OR_ACRONYM_RE = re.compile(
    r"\b(?:AI|API|SDK|TTS|URL|tools?|chatbot|dashboard|email|YouTube|Meta|Google|"
    r"Fortinet|Amazon|Instagram|Facebook|zero-day|Creator Assistant|Muse Spark|marketing)\b",
    flags=re.IGNORECASE,
)
_SENSITIVE_PHRASE_RE = re.compile(
    r"\b(?:nhạy cảm|quyền quyết định|hồ sơ|hành chính|an ninh mạng|tài khoản|"
    r"chiếm quyền|hacker|khai thác|zero-day)\b",
    flags=re.IGNORECASE,
)


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _now() -> float:
    return time.perf_counter()

def _elapsed_ms(start: float) -> int:
    return int(round((time.perf_counter() - start) * 1000))

def _env_int(name: str, default: int, low: int, high: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        return default
    return _clamp_int(value, low, high)

def _env_float(name: str, default: float, low: float, high: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(str(raw).strip())
    except ValueError:
        return default
    return max(low, min(high, value))

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

def _use_v3_hf_sampler_default(mode: str, cfg_id: str | None) -> bool:
    if mode == "standard" or cfg_id != DEFAULT_API_MODEL:
        return False
    return _env_bool("VIE_TTS_V3_HF_SAMPLER_DEFAULT", True)

def _resolve_effective_sampler(
    *,
    mode: str,
    cfg_id: str | None,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float | None,
) -> dict[str, Any]:
    use_hf_default = _use_v3_hf_sampler_default(mode, cfg_id)
    return {
        "temperature": float(
            temperature
            if temperature is not None
            else (1.0 if mode == "standard" else (DEFAULT_V3_TURBO_TEMPERATURE if use_hf_default else 0.4))
        ),
        "top_k": int(top_k if top_k is not None else (DEFAULT_V3_TURBO_TOP_K if use_hf_default else 50)),
        "top_p": float(top_p if top_p is not None else DEFAULT_V3_TURBO_TOP_P) if mode != "standard" else top_p,
        "repetition_penalty": (
            float(repetition_penalty if repetition_penalty is not None else DEFAULT_V3_TURBO_REPETITION_PENALTY)
            if mode != "standard"
            else repetition_penalty
        ),
        "hfSamplerDefault": bool(use_hf_default),
    }

def _resolve_cpu_threads() -> dict[str, Any]:
    raw = os.getenv("VIE_TTS_CPU_THREADS")
    cpu_count = os.cpu_count() or 1
    if raw is None or not str(raw).strip():
        return {
            "raw": None,
            "resolved": None,
            "key": "default",
            "cpuCount": cpu_count,
        }
    value = str(raw).strip().lower()
    if value == "auto":
        return {
            "raw": raw,
            "resolved": cpu_count,
            "key": f"auto:{cpu_count}",
            "cpuCount": cpu_count,
        }
    try:
        threads = int(value)
    except ValueError:
        return {
            "raw": raw,
            "resolved": None,
            "key": f"invalid:{raw}",
            "cpuCount": cpu_count,
        }
    threads = _clamp_int(threads, 1, max(1, cpu_count))
    return {
        "raw": raw,
        "resolved": threads,
        "key": str(threads),
        "cpuCount": cpu_count,
    }

def _thread_env_snapshot() -> dict[str, str | None]:
    names = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "GGML_N_THREADS",
        "LLAMA_THREADS",
        "TORCH_NUM_THREADS",
        "ONNX_NUM_THREADS",
        "VIE_TTS_THREADS",
        "VIE_TTS_CPU_THREADS",
        "VIE_TTS_ALIGNMENT_CPU_THREADS",
        "VIE_TTS_ALIGNMENT_MODEL",
        "VIE_TTS_ALIGNMENT_DEVICE",
        "VIE_TTS_ALIGNMENT_COMPUTE_TYPE",
    ]
    return {name: os.getenv(name) for name in names}

def _apply_python_thread_settings(cpu_threads: int | None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "torchNumThreads": None,
        "torchInteropThreads": None,
        "torchSetFromVieTtsCpuThreads": False,
    }
    try:
        import torch  # type: ignore

        if cpu_threads:
            try:
                torch.set_num_threads(int(cpu_threads))
                info["torchSetFromVieTtsCpuThreads"] = True
            except Exception as exc:
                info["torchSetError"] = str(exc)
            try:
                current_interop = torch.get_num_interop_threads()
                if current_interop > max(1, min(2, int(cpu_threads))):
                    torch.set_num_interop_threads(max(1, min(2, int(cpu_threads))))
            except Exception as exc:
                info["torchInteropSetError"] = str(exc)
        info["torchNumThreads"] = int(torch.get_num_threads())
        try:
            info["torchInteropThreads"] = int(torch.get_num_interop_threads())
        except Exception:
            pass
    except ImportError:
        info["torchAvailable"] = False
    return info

class _CpuSampler:
    def __init__(self, interval_seconds: float = 0.5):
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.available = False
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = None
        try:
            import psutil  # type: ignore

            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
            self.available = True
        except Exception as exc:
            self._psutil = None
            self.error = str(exc)

    def start(self) -> None:
        if not self.available or self._process is None:
            return
        try:
            self._process.cpu_percent(interval=None)
            self._psutil.cpu_percent(interval=None, percpu=True)
        except Exception:
            pass
        self._thread = threading.Thread(target=self._run, name="vieneu-cpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=max(1.0, self.interval_seconds * 3))
        return self.summary()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                memory = self._process.memory_info() if self._process is not None else None
                self.samples.append(
                    {
                        "processCpuPercent": float(self._process.cpu_percent(interval=None)) if self._process is not None else 0.0,
                        "perCoreCpuPercent": [float(value) for value in self._psutil.cpu_percent(interval=None, percpu=True)],
                        "memoryRssMb": float(memory.rss) / (1024 * 1024) if memory is not None else 0.0,
                    }
                )
            except Exception as exc:
                self.error = str(exc)
                break

    def summary(self) -> dict[str, Any]:
        process_values = [float(item.get("processCpuPercent", 0.0)) for item in self.samples]
        memory_values = [float(item.get("memoryRssMb", 0.0)) for item in self.samples]
        per_core_peak = 0.0
        per_core_avg = 0.0
        flattened: list[float] = []
        for item in self.samples:
            flattened.extend(float(value) for value in item.get("perCoreCpuPercent", []))
        if flattened:
            per_core_peak = max(flattened)
            per_core_avg = sum(flattened) / len(flattened)
        return {
            "samplerAvailable": self.available,
            "samplerError": self.error,
            "sampleCount": len(self.samples),
            "cpuAvgPercent": round(sum(process_values) / len(process_values), 2) if process_values else None,
            "cpuPeakPercent": round(max(process_values), 2) if process_values else None,
            "perCoreCpuAvgPercent": round(per_core_avg, 2) if flattened else None,
            "perCoreCpuPeakPercent": round(per_core_peak, 2) if flattened else None,
            "memoryPeakMb": round(max(memory_values), 2) if memory_values else None,
        }

def _gpu_memory_snapshot() -> dict[str, Any] | None:
    snapshot: dict[str, Any] = {}
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            device_index = int(torch.cuda.current_device())
            snapshot.update(
                {
                    "torchDevice": str(torch.cuda.get_device_name(device_index)),
                    "torchAllocatedMb": round(float(torch.cuda.memory_allocated(device_index)) / (1024 * 1024), 2),
                    "torchReservedMb": round(float(torch.cuda.memory_reserved(device_index)) / (1024 * 1024), 2),
                    "torchMaxAllocatedMb": round(float(torch.cuda.max_memory_allocated(device_index)) / (1024 * 1024), 2),
                }
            )
    except Exception as exc:
        snapshot["torchError"] = str(exc)

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        first_line = (completed.stdout or "").strip().splitlines()[0].strip()
        parts = [part.strip() for part in first_line.split(",")]
        if len(parts) >= 3:
            snapshot.update(
                {
                    "nvidiaUtilPercent": float(parts[0]),
                    "nvidiaMemoryUsedMb": float(parts[1]),
                    "nvidiaMemoryTotalMb": float(parts[2]),
                }
            )
    except Exception as exc:
        snapshot["nvidiaSmiError"] = str(exc)

    return snapshot or None

class _GpuSampler:
    def __init__(self, interval_seconds: float = 0.5):
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="vieneu-gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=max(1.0, self.interval_seconds * 3))
        return self.summary()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                snapshot = _gpu_memory_snapshot()
                if snapshot:
                    self.samples.append(snapshot)
            except Exception as exc:
                self.error = str(exc)
                break

    def summary(self) -> dict[str, Any]:
        util_values = [float(item["nvidiaUtilPercent"]) for item in self.samples if item.get("nvidiaUtilPercent") is not None]
        mem_values = [float(item["nvidiaMemoryUsedMb"]) for item in self.samples if item.get("nvidiaMemoryUsedMb") is not None]
        torch_alloc_values = [float(item["torchAllocatedMb"]) for item in self.samples if item.get("torchAllocatedMb") is not None]
        return {
            "samplerAvailable": bool(util_values or mem_values or torch_alloc_values),
            "samplerError": self.error,
            "sampleCount": len(self.samples),
            "gpuAvgUtilPercent": round(sum(util_values) / len(util_values), 2) if util_values else None,
            "gpuPeakUtilPercent": round(max(util_values), 2) if util_values else None,
            "gpuMemoryAvgMb": round(sum(mem_values) / len(mem_values), 2) if mem_values else None,
            "gpuMemoryPeakMb": round(max(mem_values), 2) if mem_values else None,
            "torchAllocatedPeakMb": round(max(torch_alloc_values), 2) if torch_alloc_values else None,
        }

def _log_aligned(message: str) -> None:
    print(message, flush=True)

def _split_sentence_units(text: str) -> list[str]:
    units = [match.group(0).strip() for match in _SENTENCE_UNIT_RE.finditer(text or "")]
    return [unit for unit in units if unit]


def _sentence_risk(sentence: str, segment_id: str) -> tuple[bool, list[str], str]:
    reasons: list[str] = []
    char_count = len(sentence.strip())
    comma_count = sentence.count(",")
    clause_punct_count = sum(sentence.count(mark) for mark in [",", ";", ":"])

    if char_count > 90:
        reasons.append("sentenceCharCount>90")
    if comma_count >= 2 or (comma_count >= 1 and char_count > 70):
        reasons.append("long_comma_sentence")
    if _ENGLISH_OR_ACRONYM_RE.search(sentence):
        reasons.append("english_or_acronym")
    if _NUMBER_RE.search(sentence):
        reasons.append("number")
    if _SENSITIVE_PHRASE_RE.search(sentence):
        reasons.append("sensitive_phrase")
    if clause_punct_count >= 2 or (len(sentence.split()) >= 24 and clause_punct_count >= 1):
        reasons.append("multiple_clauses")
    segment_id_lower = segment_id.lower()
    if "title" in segment_id_lower or segment_id_lower.endswith("-title") or re.search(r"\bChương\b", sentence):
        reasons.append("title_or_chapter_boundary")

    risk_level = "high" if reasons else "low"
    return bool(reasons), reasons, risk_level


def _resolve_chunk_packing_strategy(request_value: str | None = None) -> tuple[str, str]:
    raw = request_value if request_value is not None else os.getenv("VIE_TTS_CHUNK_PACKING_STRATEGY")
    source = "request" if request_value is not None else ("env" if raw else "default")
    value = str(raw or DEFAULT_V3_CHUNK_PACKING_STRATEGY).strip().lower().replace("-", "_")
    aliases = {
        "old": "legacy",
        "safe": "legacy",
        "sentence_aware": "legacy",
        "new": "balanced",
        "pack": "balanced",
        "packed": "balanced",
    }
    value = aliases.get(value, value)
    if value not in {"legacy", "balanced"}:
        return DEFAULT_V3_CHUNK_PACKING_STRATEGY, "default"
    return value, source

def _make_chunk_from_items(items: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    risk_reasons = sorted({risk_reason for item in items for risk_reason in (item.get("riskReasons") or [])})
    return {
        "sentences": [str(item["text"]) for item in items],
        "segmentIds": [str(item["segmentId"]) for item in items],
        "charCount": sum(int(item["charCount"]) for item in items) + max(0, len(items) - 1),
        "sentenceCount": len(items),
        "riskLevel": "high" if any(bool(item.get("risky")) for item in items) else "low",
        "riskReasons": risk_reasons,
        "reason": reason,
    }

def _is_balanced_hard_risk(item: dict[str, Any]) -> bool:
    return bool(item.get("risky"))

def _build_balanced_sentence_chunks(
    sentence_items: list[dict[str, Any]],
    *,
    max_chars_used: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def pending_chars(extra: dict[str, Any] | None = None) -> int:
        items = pending + ([extra] if extra is not None else [])
        return sum(int(item["charCount"]) for item in items) + max(0, len(items) - 1)

    def flush(reason: str = "balanced_sentence_pack") -> None:
        nonlocal pending
        if not pending:
            return
        if len(pending) == 1 and bool(pending[0].get("risky")):
            chunks.append(_make_chunk_from_items(pending, "risky_sentence_single_chunk"))
        elif len(pending) == 1:
            chunks.append(_make_chunk_from_items(pending, "balanced_single_sentence"))
        else:
            chunks.append(_make_chunk_from_items(pending, reason))
        pending = []

    for item in sentence_items:
        if _is_balanced_hard_risk(item):
            flush()
            chunks.append(_make_chunk_from_items([item], "risky_sentence_single_chunk"))
            continue

        if int(item["charCount"]) >= max_chars_used:
            flush()
            chunks.append(_make_chunk_from_items([item], "balanced_single_long_sentence"))
            continue

        if pending and pending_chars(item) > max_chars_used:
            flush()
        pending.append(item)

    flush()
    return chunks

def _build_sentence_aware_chunk_plan(
    segments: list[dict[str, Any]],
    requested_max_chars: int,
    min_chunk_chars: int,
    max_chunk_chars: int,
    chunk_packing_strategy: str = "legacy",
    safety_margin: int = 30,
) -> dict[str, Any]:
    sentence_items: list[dict[str, Any]] = []
    for segment in segments:
        segment_id = str(segment.get("id") or "")
        for sentence in _split_sentence_units(str(segment.get("text") or "")):
            risky, reasons, risk_level = _sentence_risk(sentence, segment_id)
            sentence_items.append(
                {
                    "text": sentence,
                    "segmentId": segment_id,
                    "charCount": len(sentence),
                    "risky": risky,
                    "riskReasons": reasons,
                    "riskLevel": risk_level,
                }
            )

    if not sentence_items:
        full_text = _build_full_narration_text(segments).strip()
        if full_text:
            risky, reasons, risk_level = _sentence_risk(full_text, "*")
            sentence_items.append(
                {
                    "text": full_text,
                    "segmentId": "*",
                    "charCount": len(full_text),
                    "risky": risky,
                    "riskReasons": reasons,
                    "riskLevel": risk_level,
                }
            )

    longest_sentence_chars = max((item["charCount"] for item in sentence_items), default=0)
    requested_max_clamped = _clamp_int(int(requested_max_chars or max_chunk_chars), min_chunk_chars, 320)
    if chunk_packing_strategy == "balanced":
        balanced_max_chars = _env_int(
            "VIE_TTS_BALANCED_MAX_CHARS",
            DEFAULT_V3_BALANCED_MAX_CHARS,
            max_chunk_chars,
            320,
        )
        max_chars_used = min(requested_max_clamped, max(max_chunk_chars, balanced_max_chars))
    else:
        max_chars_used = _clamp_int(longest_sentence_chars + safety_margin, min_chunk_chars, max_chunk_chars)
        if requested_max_chars:
            max_chars_used = min(max_chars_used, _clamp_int(int(requested_max_chars), min_chunk_chars, max_chunk_chars))

    if chunk_packing_strategy == "balanced":
        chunks = _build_balanced_sentence_chunks(sentence_items, max_chars_used=max_chars_used)
        chunks = _merge_micro_sentence_chunks(chunks, min_chunk_chars=min_chunk_chars, max_chunk_chars=max_chars_used)
    else:
        chunks: list[dict[str, Any]] = []
        pending: dict[str, Any] | None = None

        def flush_pending() -> None:
            nonlocal pending
            if pending is not None:
                chunks.append(pending)
                pending = None

        for item in sentence_items:
            if item["risky"]:
                flush_pending()
                chunks.append(
                    {
                        "sentences": [item["text"]],
                        "segmentIds": [item["segmentId"]],
                        "charCount": item["charCount"],
                        "sentenceCount": 1,
                        "riskLevel": "high",
                        "riskReasons": item["riskReasons"],
                        "reason": "risky_sentence_single_chunk",
                    }
                )
                continue

            if (
                pending is not None
                and pending["sentenceCount"] < 2
                and pending["segmentIds"][-1] == item["segmentId"]
                and pending["charCount"] + 1 + item["charCount"] <= max_chars_used
            ):
                pending["sentences"].append(item["text"])
                pending["segmentIds"].append(item["segmentId"])
                pending["charCount"] += 1 + item["charCount"]
                pending["sentenceCount"] += 1
                pending["reason"] = "safe_short_sentence_pack"
            else:
                flush_pending()
                pending = {
                    "sentences": [item["text"]],
                    "segmentIds": [item["segmentId"]],
                    "charCount": item["charCount"],
                    "sentenceCount": 1,
                    "riskLevel": "low",
                    "riskReasons": [],
                    "reason": "safe_single_sentence",
                }

        flush_pending()
        chunks = _merge_micro_sentence_chunks(chunks, min_chunk_chars=min_chunk_chars, max_chunk_chars=max_chunk_chars)

    report_chunks: list[dict[str, Any]] = []
    tts_chunks: list[str] = []
    risky_single_count = 0
    for index, chunk in enumerate(chunks):
        text = " ".join(chunk["sentences"]).strip()
        tts_chunks.append(text)
        if chunk["reason"] == "risky_sentence_single_chunk":
            risky_single_count += 1
        report_chunks.append(
            {
                "chunkIndex": index,
                "charCount": len(text),
                "sentenceCount": chunk["sentenceCount"],
                "riskLevel": chunk["riskLevel"],
                "reason": chunk["reason"],
                "riskReasons": chunk["riskReasons"],
                "segmentIds": sorted(set(chunk["segmentIds"])),
                "sourceSegmentIds": sorted(set(chunk["segmentIds"])),
                "wasMergedMicroSentence": str(chunk["reason"] or "").startswith("micro_sentence_merge"),
                "textPreview": text[:180],
            }
        )

    return {
        "maxCharsStrategy": "request-max-balanced" if chunk_packing_strategy == "balanced" else "sentence-length-adaptive",
        "chunkingStrategy": "sentence-aware",
        "chunkPackingStrategy": chunk_packing_strategy,
        "maxCharsUsed": max_chars_used,
        "minChunkChars": min_chunk_chars,
        "maxChunkChars": max_chunk_chars,
        "longestSentenceChars": longest_sentence_chars,
        "chunkCount": len(tts_chunks),
        "maxSentencesPerChunk": max((chunk["sentenceCount"] for chunk in chunks), default=0),
        "riskySentenceSingleChunkCount": risky_single_count,
        "interChunkSilenceSeconds": 0.12,
        "chunks": report_chunks,
        "ttsChunks": tts_chunks,
        "textChangedForTts": False,
    }


def _merge_micro_sentence_chunks(
    chunks: list[dict[str, Any]],
    *,
    min_chunk_chars: int,
    max_chunk_chars: int,
) -> list[dict[str, Any]]:
    if not chunks:
        return chunks

    def merge_pair(left: dict[str, Any], right: dict[str, Any], reason: str) -> dict[str, Any]:
        merged_sentences = list(left.get("sentences") or []) + list(right.get("sentences") or [])
        merged_segment_ids = list(left.get("segmentIds") or []) + list(right.get("segmentIds") or [])
        merged_risk_reasons = sorted(set((left.get("riskReasons") or []) + (right.get("riskReasons") or [])))
        merged_risk_level = "high" if left.get("riskLevel") == "high" or right.get("riskLevel") == "high" else "low"
        return {
            "sentences": merged_sentences,
            "segmentIds": merged_segment_ids,
            "charCount": int(left.get("charCount", 0)) + 1 + int(right.get("charCount", 0)),
            "sentenceCount": int(left.get("sentenceCount", 0)) + int(right.get("sentenceCount", 0)),
            "riskLevel": merged_risk_level,
            "riskReasons": merged_risk_reasons,
            "reason": reason,
        }

    merged: list[dict[str, Any]] = []
    for chunk in chunks:
        current = dict(chunk)
        while merged:
            previous = merged[-1]
            previous_chars = int(previous.get("charCount", 0))
            current_chars = int(current.get("charCount", 0))
            if previous_chars <= min_chunk_chars:
                if previous_chars + 1 + current_chars <= max_chunk_chars:
                    merged[-1] = merge_pair(previous, current, "micro_sentence_merge")
                    current = {}
                    break
            break
        if current:
            merged.append(current)

    cleanup: list[dict[str, Any]] = []
    for chunk in merged:
        current = dict(chunk)
        if cleanup and int(current.get("charCount", 0)) <= min_chunk_chars:
            previous = cleanup[-1]
            if int(previous.get("charCount", 0)) + 1 + int(current.get("charCount", 0)) <= max_chunk_chars:
                cleanup[-1] = merge_pair(previous, current, "micro_sentence_merge_prev")
                continue
        cleanup.append(current)
    merged = cleanup

    if len(merged) > 1 and int(merged[-1].get("charCount", 0)) <= min_chunk_chars:
        tail = merged.pop()
        previous = merged[-1]
        if int(previous.get("charCount", 0)) + 1 + int(tail.get("charCount", 0)) <= max_chunk_chars:
            merged[-1] = merge_pair(previous, tail, "micro_sentence_merge_tail")
        else:
            merged.append(tail)

    return merged

def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool).tolist():
        if value:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest

def _chunk_audio_qa(
    audio: np.ndarray,
    sample_rate: int,
    text: str,
    silence_db: float = DEFAULT_V3_CHUNK_SILENCE_DB,
) -> dict[str, Any]:
    wav = np.asarray(audio, dtype=np.float32).squeeze()
    total_samples = int(wav.size)
    safe_sample_rate = int(sample_rate or 1)
    duration_seconds = round(float(total_samples) / float(safe_sample_rate), 3)
    peak = float(np.max(np.abs(wav))) if total_samples > 0 else 0.0
    rms = float(np.sqrt(np.mean(np.square(wav)))) if total_samples > 0 else 0.0
    silence_threshold = float(10 ** (float(silence_db) / 20.0))
    silent_mask = np.abs(wav) <= silence_threshold if total_samples > 0 else np.array([], dtype=bool)
    longest_silence_samples = _longest_true_run(silent_mask)
    silence_ratio = float(np.mean(silent_mask)) if total_samples > 0 else 1.0
    char_count = len(str(text or "").strip())
    peak_db = 20.0 * math.log10(max(peak, 1e-8))
    rms_db = 20.0 * math.log10(max(rms, 1e-8))
    longest_silence_seconds = float(longest_silence_samples) / float(safe_sample_rate)
    near_silent = peak <= 1e-4 or rms <= silence_threshold / 4.0
    duration_outlier = duration_seconds >= max(6.0, char_count * 0.45 + 1.5)
    silent_span_outlier = duration_seconds >= 3.0 and longest_silence_seconds >= max(3.0, duration_seconds * 0.7)
    valid = True
    reason: str | None = None
    if total_samples == 0:
        valid = False
        reason = "empty_audio"
    elif (near_silent and duration_seconds >= 1.5) or (duration_outlier and silence_ratio >= 0.9) or silent_span_outlier:
        valid = False
        reason = "suspected_silent_chunk"
    return {
        "valid": valid,
        "reason": reason,
        "sampleRate": safe_sample_rate,
        "charCount": char_count,
        "durationSeconds": duration_seconds,
        "peak": round(peak, 6),
        "peakDb": round(peak_db, 2),
        "rms": round(rms, 6),
        "rmsDb": round(rms_db, 2),
        "silenceThreshold": round(silence_threshold, 6),
        "silenceRatio": round(silence_ratio, 6),
        "longestSilenceSeconds": round(longest_silence_seconds, 3),
    }

_SENTENCE_RE = re.compile(r"[^.!?\u2026]+[.!?\u2026]*", flags=re.UNICODE)
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
_NUMBER_RE = re.compile(r"\d")
_LINKER_WORDS = {"nhung", "va", "khi", "neu", "voi", "trong", "de", "vi", "nen"}

def _resolve_preset_voice_id(tts: Any, preset_voice: str) -> str:
    requested = str(preset_voice or "").strip()
    if not requested:
        raise HTTPException(status_code=400, detail="preset_voice is required when ref_audio is not provided")

    try:
        voices = tts.list_preset_voices()
    except Exception:
        voices = []

    requested_key = " ".join(_normalize_words(requested))
    for item in voices:
        if isinstance(item, tuple) and len(item) == 2:
            label, value = str(item[0]), str(item[1])
            if requested in {label, value}:
                return value
            if requested_key and requested_key in {" ".join(_normalize_words(label)), " ".join(_normalize_words(value))}:
                return value
        elif requested == str(item):
            return str(item)
        elif requested_key and requested_key == " ".join(_normalize_words(str(item))):
            return str(item)

    return requested

def _resolve_repo_path(value: str | None) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path

def _encode_v3_ref_audio_cached(tts: Any, ref_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    resolved = Path(ref_path).resolve()
    try:
        stat = resolved.stat()
        cache_key = (str(resolved), int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        cache_key = (str(resolved), 0, 0)

    with _v3_ref_codes_lock:
        cached = _v3_ref_codes_cache.get(cache_key)
        if cached is not None:
            return _coerce_v3_ref_codes(cached), {
                "cacheHit": True,
                "refAudioPath": str(resolved),
                "cacheKey": {"mtimeNs": cache_key[1], "sizeBytes": cache_key[2]},
                "encodeMs": 0,
            }

    encode_start = _now()
    codes = _coerce_v3_ref_codes(tts.encode_reference(str(resolved)))
    encode_ms = _elapsed_ms(encode_start)
    with _v3_ref_codes_lock:
        _v3_ref_codes_cache[cache_key] = np.asarray(codes).copy()
    return codes, {
        "cacheHit": False,
        "refAudioPath": str(resolved),
        "cacheKey": {"mtimeNs": cache_key[1], "sizeBytes": cache_key[2]},
        "encodeMs": encode_ms,
    }

def _is_v3_ref_audio_voice(voice: Any) -> bool:
    return isinstance(voice, dict) and str(voice.get("voice_type") or voice.get("_voice_type") or "").lower() == "ref_audio"

def _retry_max_new_frames_for_text(text: str, requested_max_tokens: int | None = None) -> int:
    char_count = len(str(text or "").strip())
    # One generated frame is about 80 ms. Vietnamese narration here is usually
    # far below char_count frames; this cap keeps broken retry attempts from
    # running all the way to the 300-frame default while leaving spoken content room.
    estimated = int(math.ceil(max(48.0, char_count * 1.25 + 24.0)))
    hard_limit = _env_int("VIE_TTS_REF_AUDIO_RETRY_MAX_NEW_FRAMES", 190, 64, 300)
    requested = int(requested_max_tokens or 300)
    return _clamp_int(min(estimated, hard_limit, requested), 48, 300)

def _ref_audio_batch_max_new_frames(texts: list[str]) -> int:
    if not texts:
        return 300
    return max(_retry_max_new_frames_for_text(text, 300) for text in texts)

def _ref_audio_retry_variant(
    *,
    base_temperature: float,
    base_top_k: int,
    attempt_number: int,
    chunk_text: str,
    max_tokens: int | None,
) -> dict[str, Any]:
    retry_index = max(1, int(attempt_number) - 1)
    temp_lift = [0.25, 0.4, 0.5, 0.6]
    top_k_values = [35, 25, 20, 20]
    variant_pos = min(retry_index - 1, len(temp_lift) - 1)
    temperature = max(float(base_temperature), min(0.9, float(base_temperature) + temp_lift[variant_pos]))
    top_k = min(int(base_top_k or 50), top_k_values[variant_pos])
    return {
        "temperature": round(float(temperature), 3),
        "top_k": int(max(1, top_k)),
        "max_new_frames": _retry_max_new_frames_for_text(chunk_text, max_tokens),
        "strategy": "ref_audio_retry_sampler_shift_and_frame_cap",
        "retryIndex": retry_index,
    }

def _apply_ref_audio_retry_variant(
    infer_kwargs: dict[str, Any],
    *,
    attempt_number: int,
    retry_variant_offset: int,
    chunk_text: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    voice = infer_kwargs.get("voice")
    if not _is_v3_ref_audio_voice(voice):
        return infer_kwargs, None
    effective_attempt = int(attempt_number) + int(retry_variant_offset)
    if effective_attempt <= 1:
        return infer_kwargs, None
    variant = _ref_audio_retry_variant(
        base_temperature=float(infer_kwargs.get("temperature") or 0.4),
        base_top_k=int(infer_kwargs.get("top_k") or 50),
        attempt_number=effective_attempt,
        chunk_text=chunk_text,
        max_tokens=int(infer_kwargs.get("max_tokens") or 300),
    )
    adjusted = dict(infer_kwargs)
    adjusted.update({
        "temperature": variant["temperature"],
        "top_k": variant["top_k"],
        "max_new_frames": variant["max_new_frames"],
    })
    return adjusted, variant

def _v3_preset_ref_audio_path(tts: Any, voice_id: str) -> Path | None:
    voices = getattr(tts, "_preset_voices", {}) or {}
    voice_data = voices.get(str(voice_id))
    if not isinstance(voice_data, dict):
        return None
    ref_path = _resolve_repo_path(voice_data.get("ref_audio") or voice_data.get("refAudioPath"))
    if ref_path and ref_path.exists():
        return ref_path
    return None

def _v3_codes_are_usable(codes: Any) -> tuple[bool, str | None]:
    if codes is None:
        return False, "Preset has no v3 reference codes."
    try:
        arr = np.asarray(codes)
    except Exception as exc:
        return False, f"Preset codes cannot be read: {exc}"
    if arr.ndim != 2:
        return False, f"Preset codes are {arr.ndim}D; VieNeu v3 Turbo GPU built-in presets require 2D MOSS codes."
    if arr.shape[1] <= 0:
        return False, "Preset codes have an invalid channel dimension."
    return True, None

def _resolve_v3_preset_voice(tts: Any, voice_id: str) -> dict[str, Any]:
    voices = getattr(tts, "_preset_voices", {}) or {}
    voice_data = voices.get(str(voice_id))
    if not isinstance(voice_data, dict):
        raise HTTPException(status_code=422, detail=f"Preset voice '{voice_id}' metadata is missing.")

    ref_path = _v3_preset_ref_audio_path(tts, voice_id)
    codes_usable, _ = _v3_codes_are_usable(voice_data.get("codes"))
    if ref_path is not None and not codes_usable:
        return {"codes": tts.encode_reference(str(ref_path)), "text": voice_data.get("text", "")}
    return tts.get_preset_voice(voice_id)

def _resolve_aligned_voice(tts: Any, req: AlignedSynthesizeRequest, mode: str) -> Any:
    if req.ref_audio:
        ref_path = Path(req.ref_audio)
        if not ref_path.is_absolute():
            ref_path = ROOT_DIR / ref_path
        if not ref_path.exists():
            raise HTTPException(status_code=400, detail=f"ref_audio does not exist: {req.ref_audio}")
        if mode == "standard":
            if not req.ref_text or not req.ref_text.strip():
                raise HTTPException(status_code=400, detail="ref_text is required for standard mode when ref_audio is used")
            return {"codes": tts.encode_reference(str(ref_path)), "text": req.ref_text.strip()}
        if mode == "v3turbo":
            ref_codes, ref_report = _encode_v3_ref_audio_cached(tts, ref_path)
            return {
                "codes": ref_codes,
                "voice_type": "ref_audio",
                "_voice_type": "ref_audio",
                "ref_audio": str(ref_path),
                "refAudioPath": str(ref_path),
                "refAudioCache": ref_report,
            }
        return tts.encode_reference(str(ref_path))

    requested_voice = str(req.preset_voice or "").strip()
    requested_voice_key = " ".join(_normalize_words(requested_voice))
    nguyen_huyen_key = " ".join(_normalize_words(NGUYEN_HUYEN_TRANG_VOICE_ID))
    if mode == "v3turbo" and requested_voice_key and requested_voice_key == nguyen_huyen_key:
        if not DEFAULT_REF_AUDIO_PATH.exists():
            raise HTTPException(status_code=400, detail=f"ref_audio does not exist: {DEFAULT_REF_AUDIO_REL}")
        ref_codes, ref_report = _encode_v3_ref_audio_cached(tts, DEFAULT_REF_AUDIO_PATH)
        return {
            "codes": ref_codes,
            "voice_type": "ref_audio",
            "_voice_type": "ref_audio",
            "ref_audio": str(DEFAULT_REF_AUDIO_PATH),
            "refAudioPath": str(DEFAULT_REF_AUDIO_PATH),
            "refAudioCache": ref_report,
            "presetVoiceId": NGUYEN_HUYEN_TRANG_VOICE_ID,
        }

    voice_id = _resolve_preset_voice_id(tts, req.preset_voice or "")
    if mode == "v3turbo":
        usable, warning = _v3_preset_voice_usability(tts, voice_id)
        if not usable:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Preset voice '{voice_id}' is not compatible with VieNeu v3 Turbo GPU built-in synthesis. "
                    f"{warning or 'Use a v3 built-in preset with 2D MOSS codes or provide ref_audio.'}"
                ),
            )
        return _resolve_v3_preset_voice(tts, voice_id)
    return tts.get_preset_voice(voice_id)

def _v3_preset_voice_usability(tts: Any, voice_id: str) -> tuple[bool, str | None]:
    voices = getattr(tts, "_preset_voices", {}) or {}
    voice_data = voices.get(str(voice_id))
    if not isinstance(voice_data, dict):
        return False, "Preset metadata is missing."
    codes_usable, warning = _v3_codes_are_usable(voice_data.get("codes"))
    if codes_usable:
        return True, None
    ref_path = _v3_preset_ref_audio_path(tts, voice_id)
    if ref_path is not None:
        return True, None
    return False, warning

def _save_wav(audio: Any, output_path: Path, sample_rate: int = 24000) -> None:
    import soundfile as sf

    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.squeeze()
    sf.write(str(output_path), wav, sample_rate)

def _apply_speed(audio: Any, speed: float) -> np.ndarray:
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.squeeze()
    safe_speed = float(speed or 1.0)
    if abs(safe_speed - 1.0) < 1e-6:
        return wav
    if safe_speed < 0.5 or safe_speed > 2.0:
        raise HTTPException(status_code=422, detail="speed must be between 0.5 and 2.0")

    import librosa

    return librosa.effects.time_stretch(wav, rate=safe_speed).astype(np.float32)

def _build_chunk_infer_kwargs(
    *,
    chunk_text: str,
    voice: Any,
    mode: str,
    temperature: float,
    top_k: int,
    max_chars: int,
    max_tokens: int,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
) -> dict[str, Any]:
    infer_kwargs = {
        "text": chunk_text,
        "voice": voice,
        "temperature": temperature,
        "top_k": top_k,
        "max_chars": max_chars,
        "apply_watermark": False,
    }
    if top_p is not None:
        infer_kwargs["top_p"] = float(top_p)
    if repetition_penalty is not None:
        infer_kwargs["repetition_penalty"] = float(repetition_penalty)
    if mode != "standard":
        infer_kwargs["max_tokens"] = max_tokens
    return infer_kwargs

def _coerce_v3_ref_codes(codes: Any) -> np.ndarray:
    arr = np.asarray(codes)
    if arr.ndim == 1 and arr.size % 16 == 0:
        arr = arr.reshape((-1, 16))
    return arr

def _get_v3_batch_size() -> int:
    return _clamp_int(
        _env_int("VIE_TTS_V3_BATCH_SIZE", DEFAULT_V3_BATCH_SIZE, 1, MAX_V3_BATCH_SIZE),
        1,
        MAX_V3_BATCH_SIZE,
    )

def _resolve_v3_batch_size(request_value: int | None = None) -> tuple[int, str]:
    if request_value is not None:
        return _clamp_int(int(request_value), 1, MAX_V3_BATCH_SIZE), "request"
    raw = os.getenv("VIE_TTS_V3_BATCH_SIZE")
    if raw is not None and str(raw).strip():
        return _get_v3_batch_size(), "env"
    return DEFAULT_V3_BATCH_SIZE, "default"

def _get_v3_batch_retry_silence_seconds() -> float:
    raw = os.getenv("VIE_TTS_V3_BATCH_RETRY_SILENCE_SECONDS")
    if raw is None or not str(raw).strip():
        return DEFAULT_V3_BATCH_RETRY_SILENCE_SECONDS
    try:
        value = float(str(raw).strip())
    except ValueError:
        return DEFAULT_V3_BATCH_RETRY_SILENCE_SECONDS
    return max(0.5, min(5.0, value))

def _silence_policy_config() -> dict[str, float]:
    soft_warn = _env_float(
        "VIE_TTS_SILENCE_SOFT_WARN_SECONDS",
        _get_v3_batch_retry_silence_seconds(),
        0.5,
        10.0,
    )
    hard_retry = _env_float(
        "VIE_TTS_SILENCE_HARD_RETRY_SECONDS",
        DEFAULT_SILENCE_HARD_RETRY_SECONDS,
        max(soft_warn, 0.5),
        30.0,
    )
    severe = _env_float(
        "VIE_TTS_SILENCE_SEVERE_SECONDS",
        DEFAULT_SILENCE_SEVERE_SECONDS,
        max(hard_retry, 1.0),
        60.0,
    )
    return {
        "softWarnSeconds": round(float(soft_warn), 3),
        "hardRetrySeconds": round(float(hard_retry), 3),
        "severeSeconds": round(float(severe), 3),
    }

def _annotate_silence_policy(qa: dict[str, Any], policy: dict[str, float]) -> None:
    try:
        longest = float(qa.get("longestSilenceSeconds") or 0.0)
        silence_ratio = float(qa.get("silenceRatio") or 0.0)
        rms_db = float(qa.get("rmsDb") or -120.0)
        duration_seconds = float(qa.get("durationSeconds") or 0.0)
    except (TypeError, ValueError):
        return

    valid = bool(qa.get("valid", True))
    reason = qa.get("reason")
    severity = "ok"
    retry_required = False
    retry_reason = None

    if longest >= float(policy["severeSeconds"]):
        severity = "severe_dropout"
        retry_required = True
        retry_reason = "silence_severe"
    elif longest >= float(policy["hardRetrySeconds"]):
        severity = "hard_dropout"
        retry_required = True
        retry_reason = "silence_hard"
    elif longest >= float(policy["softWarnSeconds"]):
        severity = "soft_pause"
        try:
            char_count = float(qa.get("charCount") or 1.0)
        except (TypeError, ValueError):
            char_count = 1.0
        soft_bad_signs = (
            silence_ratio >= 0.85
            or rms_db <= -30.0
            or duration_seconds >= max(6.0, 0.45 * max(1.0, char_count) + 1.5)
        )
        if soft_bad_signs and not valid:
            retry_required = True
            retry_reason = "silence_soft_with_bad_audio"
        elif reason in {"empty_audio", "suspected_silent_chunk"}:
            retry_required = True
            retry_reason = "silence_soft_with_invalid_audio"

    if not valid and retry_reason is None:
        retry_required = True
        retry_reason = "invalid_audio"
        if severity == "ok":
            severity = "hard_dropout"

    qa["silenceSeverity"] = severity
    qa["silenceRetryRequired"] = bool(retry_required)
    qa["silenceRetryReason"] = retry_reason
    qa["silenceWarning"] = "soft_pause" if severity == "soft_pause" and not retry_required else None
    qa["silenceSoftWarnSeconds"] = float(policy["softWarnSeconds"])
    qa["silenceHardRetrySeconds"] = float(policy["hardRetrySeconds"])
    qa["silenceSevereSeconds"] = float(policy["severeSeconds"])

def _is_v3_cuda_batch_ready(tts: Any, mode: str) -> bool:
    if mode != "v3turbo":
        return False
    engine = getattr(tts, "engine", None)
    device = getattr(engine, "device", None)
    return device is not None and getattr(device, "type", None) == "cuda"

def _resolve_v3_batch_voice(tts: Any, voice: Any) -> tuple[np.ndarray, int | None]:
    resolver = getattr(tts, "_resolve_v3_ref", None)
    if callable(resolver):
        ref_codes, voice_token_id = resolver(voice, None, None)
    else:
        ref_codes, voice_token_id = (None, None)
        if isinstance(voice, dict):
            ref_codes = voice.get("codes")
            voice_token_id = voice.get("reserved_id")
        elif isinstance(voice, str):
            voice_data = getattr(tts, "_preset_voices", {}).get(voice, {})
            ref_codes = voice_data.get("codes")
            voice_token_id = voice_data.get("reserved_id")
        if ref_codes is None:
            raise HTTPException(status_code=422, detail="Unable to resolve v3 voice for batch synthesis")
    if ref_codes is None:
        raise HTTPException(status_code=422, detail="Unable to resolve v3 reference codes")
    return _coerce_v3_ref_codes(ref_codes), (int(voice_token_id) if voice_token_id is not None else None)

def _build_v3_batch_jobs(
    tts_chunks: list[str],
    *,
    voice: Any,
    mode: str,
    temperature: float,
    top_k: int,
    max_chars: int,
    max_tokens: int,
    batch_size: int,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    parent_chunks: list[dict[str, Any]] = []

    for parent_index, chunk_text in enumerate(tts_chunks):
        subchunks = split_text_into_chunks(chunk_text, max_chars=max_chars) or [chunk_text]
        parent_chunks.append(
            {
                "parentIndex": parent_index,
                "text": chunk_text,
                "subchunkCount": len(subchunks),
                "charCount": len(chunk_text),
            }
        )
        for sub_index, subchunk_text in enumerate(subchunks):
            jobs.append(
                {
                    "jobIndex": len(jobs),
                    "parentIndex": parent_index,
                    "subIndex": sub_index,
                    "text": subchunk_text,
                    "inferKwargs": _build_chunk_infer_kwargs(
                        chunk_text=subchunk_text,
                        voice=voice,
                        mode=mode,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        max_chars=max_chars,
                        max_tokens=max_tokens,
                    ),
                }
            )

    if batch_size > 1 and len(jobs) > 1:
        for job in jobs:
            job["batchIndex"] = job["jobIndex"] // batch_size
    else:
        for job in jobs:
            job["batchIndex"] = 0
    return jobs, parent_chunks

def _infer_one_chunk(
    *,
    tts: Any,
    index: int,
    chunk_text: str,
    infer_kwargs: dict[str, Any],
    sample_rate: int,
    chunk_audio_retries: int,
    use_lock: bool,
    max_silence_seconds: float | None = None,
    silence_policy: dict[str, float] | None = None,
    retry_variant_offset: int = 0,
) -> tuple[int, np.ndarray | None, int, dict[str, Any]]:
    total_start = _now()
    attempts: list[dict[str, Any]] = []
    max_attempts = max(1, int(chunk_audio_retries) + 1)
    last_audio: np.ndarray | None = None
    last_qa: dict[str, Any] | None = None

    for attempt in range(1, max_attempts + 1):
        active_infer_kwargs, retry_variant = _apply_ref_audio_retry_variant(
            infer_kwargs,
            attempt_number=attempt,
            retry_variant_offset=retry_variant_offset,
            chunk_text=chunk_text,
        )
        start = _now()
        if use_lock:
            with _model_lock:
                chunk_audio = tts.infer(**active_infer_kwargs)
        else:
            chunk_audio = tts.infer(**active_infer_kwargs)
        infer_duration_ms = _elapsed_ms(start)
        if chunk_audio is None or len(chunk_audio) == 0:
            qa = {
                "valid": False,
                "reason": "empty_audio",
                "attempt": attempt,
                "inferDurationMs": infer_duration_ms,
                "temperature": active_infer_kwargs.get("temperature"),
                "topK": active_infer_kwargs.get("top_k"),
                "maxNewFrames": active_infer_kwargs.get("max_new_frames"),
                "retryVariant": retry_variant,
            }
            active_silence_policy = silence_policy
            if active_silence_policy is None and max_silence_seconds is not None:
                active_silence_policy = {
                    "softWarnSeconds": round(float(max_silence_seconds), 3),
                    "hardRetrySeconds": round(float(max_silence_seconds), 3),
                    "severeSeconds": max(round(float(max_silence_seconds), 3), DEFAULT_SILENCE_SEVERE_SECONDS),
                }
            if active_silence_policy is not None:
                _annotate_silence_policy(qa, active_silence_policy)
            attempts.append(qa)
            last_qa = qa
            continue

        audio = np.asarray(chunk_audio, dtype=np.float32).squeeze()
        qa = _chunk_audio_qa(audio, sample_rate, chunk_text)
        active_silence_policy = silence_policy
        if active_silence_policy is None and max_silence_seconds is not None:
            active_silence_policy = {
                "softWarnSeconds": round(float(max_silence_seconds), 3),
                "hardRetrySeconds": round(float(max_silence_seconds), 3),
                "severeSeconds": max(round(float(max_silence_seconds), 3), DEFAULT_SILENCE_SEVERE_SECONDS),
            }
        if active_silence_policy is not None:
            _annotate_silence_policy(qa, active_silence_policy)
            if qa.get("silenceRetryRequired"):
                qa["valid"] = False
                qa["reason"] = qa.get("silenceRetryReason") or qa.get("reason") or "suspected_long_silence"
                qa["maxAllowedSilenceSeconds"] = round(float(active_silence_policy["hardRetrySeconds"]), 3)
        qa["attempt"] = attempt
        qa["inferDurationMs"] = infer_duration_ms
        qa["temperature"] = active_infer_kwargs.get("temperature")
        qa["topK"] = active_infer_kwargs.get("top_k")
        qa["maxNewFrames"] = active_infer_kwargs.get("max_new_frames")
        qa["retryVariant"] = retry_variant
        attempts.append(qa)
        last_audio = audio
        last_qa = qa
        if qa.get("valid"):
            return index, audio, _elapsed_ms(total_start), {
                **qa,
                "attemptCount": attempt,
                "retried": attempt > 1,
                "attempts": attempts,
            }

    preview = str(chunk_text or "").replace("\n", " ")[:140]
    raise HTTPException(
        status_code=500,
        detail=(
            f"VieNeu v3 chunk audio QA failed after {max_attempts} attempt(s) for chunk {index}. "
            f"reason={last_qa.get('reason') if last_qa else 'unknown'} text={preview!r}"
        ),
    )

def _init_chunk_perf(tts_chunks: list[str], chunk_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "charCount": len(chunk_text),
            "sentenceCount": int(chunk_reports[index].get("sentenceCount", 0)) if index < len(chunk_reports) else 0,
            "durationMs": 0,
            "textPreview": chunk_text[:180],
            "riskLevel": chunk_reports[index].get("riskLevel") if index < len(chunk_reports) else None,
            "reason": chunk_reports[index].get("reason") if index < len(chunk_reports) else None,
            "sourceSegmentIds": chunk_reports[index].get("sourceSegmentIds") if index < len(chunk_reports) else [],
            "wasMergedMicroSentence": bool(chunk_reports[index].get("wasMergedMicroSentence")) if index < len(chunk_reports) else False,
            "attemptCount": None,
            "retryCount": 0,
            "chunkAudioQa": None,
        }
        for index, chunk_text in enumerate(tts_chunks)
    ]

def _set_chunk_perf_audio_qa(
    chunk_perf: list[dict[str, Any]],
    index: int,
    duration_ms: int,
    audio_qa: dict[str, Any],
) -> None:
    chunk_perf[index]["durationMs"] = duration_ms
    chunk_perf[index]["chunkAudioQa"] = audio_qa
    chunk_perf[index]["attemptCount"] = int(audio_qa.get("attemptCount") or 1)
    chunk_perf[index]["retryCount"] = max(0, int(audio_qa.get("attemptCount") or 1) - 1)
    chunk_perf[index]["retryAttemptCount"] = max(0, int(audio_qa.get("attemptCount") or 1) - 1)
    audio_qa["retryAttemptCount"] = max(0, int(audio_qa.get("attemptCount") or 1) - 1)

def _median(values: list[float]) -> float | None:
    clean_values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean_values:
        return None
    mid = len(clean_values) // 2
    if len(clean_values) % 2:
        return clean_values[mid]
    return (clean_values[mid - 1] + clean_values[mid]) / 2.0

def _volume_consistency_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "localDropDbThreshold": round(
            _env_float("VIE_TTS_VOLUME_LOCAL_DROP_DB", DEFAULT_VOLUME_LOCAL_DROP_DB, 1.0, 18.0),
            2,
        ),
        "globalDropDbThreshold": round(
            _env_float("VIE_TTS_VOLUME_GLOBAL_DROP_DB", DEFAULT_VOLUME_GLOBAL_DROP_DB, 2.0, 30.0),
            2,
        ),
        "minDurationSeconds": round(
            _env_float("VIE_TTS_VOLUME_OUTLIER_MIN_SECONDS", DEFAULT_VOLUME_OUTLIER_MIN_SECONDS, 0.5, 10.0),
            2,
        ),
        "retryVolumeOutlier": _env_int("VIE_TTS_RETRY_VOLUME_OUTLIER", DEFAULT_RETRY_VOLUME_OUTLIER, 0, 3),
        "localWindow": _env_int("VIE_TTS_VOLUME_LOCAL_WINDOW", DEFAULT_VOLUME_LOCAL_WINDOW, 1, 2),
    }

def _qa_rms_db_for_volume(qa: dict[str, Any], min_duration_seconds: float) -> float | None:
    try:
        rms_db = float(qa.get("rmsDb"))
        duration_seconds = float(qa.get("durationSeconds") if qa.get("durationSeconds") is not None else 0.0)
        silence_ratio = float(qa.get("silenceRatio") if qa.get("silenceRatio") is not None else 1.0)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rms_db):
        return None
    if not bool(qa.get("valid", True)):
        return None
    if duration_seconds < float(min_duration_seconds):
        return None
    if silence_ratio >= 0.85:
        return None
    return rms_db

def _annotate_one_volume_qa(
    qa: dict[str, Any],
    *,
    global_median: float | None,
    local_median: float | None,
    config: dict[str, Any],
) -> None:
    try:
        rms_db = float(qa.get("rmsDb"))
        duration_seconds = float(qa.get("durationSeconds") or 0.0)
    except (TypeError, ValueError):
        return
    if not math.isfinite(rms_db):
        return

    min_duration_seconds = float(config["minDurationSeconds"])
    eligible = duration_seconds >= min_duration_seconds and bool(qa.get("valid", True))
    qa["volumeOutlierEligible"] = bool(eligible)
    if global_median is not None:
        delta_global = rms_db - float(global_median)
        qa["globalMedianRmsDb"] = round(float(global_median), 2)
        qa["deltaVsGlobalDb"] = round(delta_global, 2)
        qa["volumeOutlierGlobal"] = bool(eligible and -delta_global >= float(config["globalDropDbThreshold"]))
    else:
        qa["globalMedianRmsDb"] = None
        qa["deltaVsGlobalDb"] = None
        qa["volumeOutlierGlobal"] = False

    if local_median is not None:
        delta_local = rms_db - float(local_median)
        qa["localNeighborMedianRmsDb"] = round(float(local_median), 2)
        qa["deltaVsLocalDb"] = round(delta_local, 2)
        qa["volumeOutlierLocal"] = bool(eligible and -delta_local >= float(config["localDropDbThreshold"]))
    else:
        qa["localNeighborMedianRmsDb"] = None
        qa["deltaVsLocalDb"] = None
        qa["volumeOutlierLocal"] = False

def _annotate_volume_consistency(chunk_perf: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    min_duration_seconds = float(config["minDurationSeconds"])
    local_window = int(config["localWindow"])
    rms_by_index: dict[int, float] = {}
    for item in chunk_perf:
        qa = item.get("chunkAudioQa") or {}
        rms_db = _qa_rms_db_for_volume(qa, min_duration_seconds)
        if rms_db is not None:
            rms_by_index[int(item["index"])] = rms_db

    global_median = _median(list(rms_by_index.values()))
    worst_local_drop = 0.0
    local_outlier_indexes: list[int] = []
    global_outlier_indexes: list[int] = []

    for item in chunk_perf:
        index = int(item["index"])
        qa = item.get("chunkAudioQa") or {}
        neighbor_values: list[float] = []
        for neighbor_index in range(index - local_window, index + local_window + 1):
            if neighbor_index == index:
                continue
            value = rms_by_index.get(neighbor_index)
            if value is not None:
                neighbor_values.append(value)
        local_median = _median(neighbor_values)
        _annotate_one_volume_qa(
            qa,
            global_median=global_median,
            local_median=local_median,
            config=config,
        )
        if qa.get("volumeOutlierLocal"):
            local_outlier_indexes.append(index)
        if qa.get("volumeOutlierGlobal"):
            global_outlier_indexes.append(index)
        delta_local = qa.get("deltaVsLocalDb")
        if delta_local is not None:
            try:
                worst_local_drop = max(worst_local_drop, max(0.0, -float(delta_local)))
            except (TypeError, ValueError):
                pass

    outlier_indexes = sorted(set(local_outlier_indexes + global_outlier_indexes))
    return {
        "enabled": True,
        "localDropDbThreshold": float(config["localDropDbThreshold"]),
        "globalDropDbThreshold": float(config["globalDropDbThreshold"]),
        "minDurationSeconds": float(config["minDurationSeconds"]),
        "localWindow": local_window,
        "globalMedianRmsDb": round(global_median, 2) if global_median is not None else None,
        "outlierCount": len(outlier_indexes),
        "localOutlierCount": len(local_outlier_indexes),
        "globalOutlierCount": len(global_outlier_indexes),
        "outlierIndexes": outlier_indexes,
        "localOutlierIndexes": local_outlier_indexes,
        "globalOutlierIndexes": global_outlier_indexes,
        "worstLocalDropDb": round(worst_local_drop, 2),
        "retriedVolumeOutlierCount": 0,
        "warningVolumeOutlierCount": 0,
    }

def _volume_retry_score(qa: dict[str, Any], target_rms_db: float | None, retry_silence_seconds: float) -> float:
    if target_rms_db is None:
        target_rms_db = qa.get("globalMedianRmsDb")
    try:
        rms_db = float(qa.get("rmsDb"))
    except (TypeError, ValueError):
        rms_db = -120.0
    target = float(target_rms_db) if target_rms_db is not None else rms_db
    score = abs(rms_db - target)
    if not bool(qa.get("valid", True)):
        score += 100.0
    try:
        if float(qa.get("longestSilenceSeconds") or 0.0) >= retry_silence_seconds:
            score += 20.0
    except (TypeError, ValueError):
        pass
    try:
        peak = float(qa.get("peak") or 0.0)
        if peak >= 1.5:
            score += (peak - 1.5) * 10.0
    except (TypeError, ValueError):
        pass
    return score

def _apply_volume_outlier_retry(
    *,
    tts: Any,
    voice: Any,
    mode: str,
    sample_rate: int,
    tts_chunks: list[str],
    wav_chunks: list[np.ndarray | None],
    chunk_perf: list[dict[str, Any]],
    jobs_by_parent: dict[int, list[dict[str, Any]]],
    batch_reports_by_index: dict[int, dict[str, Any]],
    temperature: float,
    top_k: int,
    top_p: float | None,
    repetition_penalty: float | None,
    max_chars: int,
    max_tokens: int,
    retry_silence_seconds: float,
    silence_policy: dict[str, float] | None = None,
) -> dict[str, Any]:
    config = _volume_consistency_config()
    initial_summary = _annotate_volume_consistency(chunk_perf, config)
    retry_limit = int(config["retryVolumeOutlier"])
    retry_reports: list[dict[str, Any]] = []
    warning_indexes: set[int] = set()

    candidate_indexes = [
        int(item["index"])
        for item in chunk_perf
        if (item.get("chunkAudioQa") or {}).get("volumeOutlierLocal")
        or (item.get("chunkAudioQa") or {}).get("volumeOutlierGlobal")
    ]

    if retry_limit <= 0:
        final_summary = _annotate_volume_consistency(chunk_perf, config)
        final_summary["detectedOutlierCount"] = initial_summary["outlierCount"]
        final_summary["detectedOutlierIndexes"] = initial_summary["outlierIndexes"]
        final_summary["retryVolumeOutlier"] = retry_limit
        final_summary["retryReports"] = retry_reports
        final_summary["warningVolumeOutlierCount"] = final_summary["outlierCount"]
        return final_summary

    for index in candidate_indexes:
        if index < 0 or index >= len(chunk_perf):
            continue
        item = chunk_perf[index]
        original_qa = dict(item.get("chunkAudioQa") or {})
        target_rms_db = original_qa.get("localNeighborMedianRmsDb")
        if target_rms_db is None:
            target_rms_db = original_qa.get("globalMedianRmsDb")
        original_score = _volume_retry_score(original_qa, target_rms_db, retry_silence_seconds)
        best_audio = wav_chunks[index]
        best_qa = original_qa
        best_duration_ms = int(item.get("durationMs") or 0)
        retry_ms_total = 0
        retry_error: str | None = None

        for _ in range(retry_limit):
            retry_start = _now()
            try:
                _, retry_audio, retry_ms, retry_qa = _infer_one_chunk(
                    tts=tts,
                    index=index,
                    chunk_text=tts_chunks[index],
                    infer_kwargs=_build_chunk_infer_kwargs(
                        chunk_text=tts_chunks[index],
                        voice=voice,
                        mode=mode,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        max_chars=max_chars,
                        max_tokens=max_tokens,
                    ),
                    sample_rate=sample_rate,
                    chunk_audio_retries=0,
                    use_lock=True,
                    max_silence_seconds=retry_silence_seconds,
                    silence_policy=silence_policy,
                    retry_variant_offset=1,
                )
            except HTTPException as exc:
                retry_ms = _elapsed_ms(retry_start)
                retry_audio = None
                retry_qa = {}
                retry_error = str(exc.detail)
            retry_ms_total += int(retry_ms)
            if retry_audio is None or len(retry_audio) == 0 or not retry_qa:
                continue
            _annotate_one_volume_qa(
                retry_qa,
                global_median=original_qa.get("globalMedianRmsDb"),
                local_median=original_qa.get("localNeighborMedianRmsDb"),
                config=config,
            )
            retry_score = _volume_retry_score(retry_qa, target_rms_db, retry_silence_seconds)
            if retry_score < original_score:
                original_score = retry_score
                best_audio = retry_audio
                best_qa = retry_qa
                best_duration_ms = int(item.get("durationMs") or 0) + retry_ms_total

        retry_report = {
            "chunkIndex": index,
            "retryReason": "volume_outlier_local" if original_qa.get("volumeOutlierLocal") else "volume_outlier_global",
            "retryCount": retry_limit,
            "originalRmsDb": original_qa.get("rmsDb"),
            "retryRmsDb": best_qa.get("rmsDb") if best_qa is not original_qa else None,
            "localNeighborMedianRmsDb": original_qa.get("localNeighborMedianRmsDb"),
            "globalMedianRmsDb": original_qa.get("globalMedianRmsDb"),
            "originalDeltaVsLocalDb": original_qa.get("deltaVsLocalDb"),
            "retryDeltaVsLocalDb": best_qa.get("deltaVsLocalDb") if best_qa is not original_qa else None,
            "retryMs": retry_ms_total,
            "status": "retry_not_better",
        }
        if retry_error:
            retry_report["retryError"] = retry_error

        if best_qa is not original_qa and best_audio is not None:
            original_attempts = list(original_qa.get("attempts") or [original_qa])
            retry_attempts = list(best_qa.get("attempts") or [best_qa])
            final_qa = {
                **best_qa,
                "attemptCount": int(original_qa.get("attemptCount") or 1) + int(best_qa.get("attemptCount") or 1),
                "retried": True,
                "retryReason": retry_report["retryReason"],
                "attempts": original_attempts + retry_attempts,
            }
            retry_report["status"] = (
                "retried_ok"
                if not (best_qa.get("volumeOutlierLocal") or best_qa.get("volumeOutlierGlobal"))
                else "retried_better_warning_volume_outlier"
            )
            final_qa["volumeRetry"] = retry_report
            _set_chunk_perf_audio_qa(chunk_perf, index, best_duration_ms, final_qa)
            wav_chunks[index] = np.asarray(best_audio, dtype=np.float32).squeeze()
        else:
            warning_indexes.add(index)
            original_qa["volumeWarning"] = "warning_volume_outlier"
            original_qa["volumeRetry"] = retry_report
            _set_chunk_perf_audio_qa(chunk_perf, index, int(item.get("durationMs") or 0) + retry_ms_total, original_qa)

        parent_jobs = jobs_by_parent.get(index, [])
        for batch_index_for_parent in sorted({int(job.get("batchIndex", 0)) for job in parent_jobs}):
            batch_report = batch_reports_by_index.get(batch_index_for_parent)
            if batch_report is not None:
                batch_report["volumeRetryMs"] = int(batch_report.get("volumeRetryMs", 0)) + retry_ms_total
                retry_chunk_indexes = list(batch_report.get("volumeRetryChunkIndexes") or [])
                if index not in retry_chunk_indexes:
                    retry_chunk_indexes.append(index)
                batch_report["volumeRetryChunkIndexes"] = sorted(retry_chunk_indexes)

        retry_reports.append(retry_report)

    final_summary = _annotate_volume_consistency(chunk_perf, config)
    final_outlier_indexes = set(final_summary["outlierIndexes"])
    warning_indexes.update(final_outlier_indexes)
    final_summary["detectedOutlierCount"] = initial_summary["outlierCount"]
    final_summary["detectedOutlierIndexes"] = initial_summary["outlierIndexes"]
    final_summary["retryVolumeOutlier"] = retry_limit
    final_summary["retriedVolumeOutlierCount"] = sum(1 for report in retry_reports if str(report.get("status", "")).startswith("retried"))
    final_summary["warningVolumeOutlierCount"] = len(warning_indexes)
    final_summary["warningVolumeOutlierIndexes"] = sorted(warning_indexes)
    final_summary["retryReports"] = retry_reports
    return final_summary

def _summarize_silence_policy(chunk_perf: list[dict[str, Any]], policy: dict[str, float]) -> dict[str, Any]:
    summary = {
        "softPauseCount": 0,
        "hardDropoutCount": 0,
        "severeDropoutCount": 0,
        "finalSoftPauseCount": 0,
        "finalHardDropoutCount": 0,
        "finalSevereDropoutCount": 0,
        "retryDueToSilenceCount": 0,
        "detectedSoftPauseCount": 0,
        "detectedHardDropoutCount": 0,
        "detectedSevereDropoutCount": 0,
    }

    for item in chunk_perf:
        qa = item.get("chunkAudioQa") or {}
        severity = str(qa.get("silenceSeverity") or "ok")
        attempts = list(qa.get("attempts") or [])
        if severity == "soft_pause":
            summary["finalSoftPauseCount"] += 1
        elif severity == "hard_dropout":
            summary["finalHardDropoutCount"] += 1
        elif severity == "severe_dropout":
            summary["finalSevereDropoutCount"] += 1

        if any(bool(attempt.get("silenceRetryRequired")) for attempt in attempts):
            summary["retryDueToSilenceCount"] += 1

        chunk_has_soft = False
        chunk_has_hard = False
        chunk_has_severe = False
        for attempt in attempts:
            att_severity = str(attempt.get("silenceSeverity") or "ok")
            if att_severity == "soft_pause":
                summary["detectedSoftPauseCount"] += 1
                chunk_has_soft = True
            elif att_severity == "hard_dropout":
                summary["detectedHardDropoutCount"] += 1
                chunk_has_hard = True
            elif att_severity == "severe_dropout":
                summary["detectedSevereDropoutCount"] += 1
                chunk_has_severe = True
        if chunk_has_soft:
            summary["softPauseCount"] += 1
        if chunk_has_hard:
            summary["hardDropoutCount"] += 1
        if chunk_has_severe:
            summary["severeDropoutCount"] += 1

    return {
        "enabled": True,
        "policy": {
            "softWarnSeconds": float(policy["softWarnSeconds"]),
            "hardRetrySeconds": float(policy["hardRetrySeconds"]),
            "severeSeconds": float(policy["severeSeconds"]),
        },
        **summary,
    }

def _synthesize_v3_batch_cuda_chunks(
    *,
    tts: Any,
    voice: Any,
    mode: str,
    sample_rate: int,
    tts_chunks: list[str],
    chunk_reports: list[dict[str, Any]],
    temperature: float,
    top_k: int,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    max_chars: int,
    max_tokens: int,
    chunk_audio_retries: int,
    batch_size: int,
    silence_policy: dict[str, float] | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    from vieneu.v3_turbo_serve import V3TurboBatchEngine

    batch_start = _now()
    chunk_perf = _init_chunk_perf(tts_chunks, chunk_reports)
    wav_chunks: list[np.ndarray | None] = [None] * len(tts_chunks)
    ref_codes, voice_token_id = _resolve_v3_batch_voice(tts, voice)
    ref_audio_optimization = {
        "enabled": _is_v3_ref_audio_voice(voice),
        "refAudioPath": voice.get("ref_audio") if isinstance(voice, dict) else None,
        "refAudioCache": voice.get("refAudioCache") if isinstance(voice, dict) else None,
        "retryStrategy": "sampler_shift_and_frame_cap" if _is_v3_ref_audio_voice(voice) else None,
        "retryMaxNewFramesEnv": os.getenv("VIE_TTS_REF_AUDIO_RETRY_MAX_NEW_FRAMES") or "190",
        "initialBatchUnchanged": True,
    }
    silence_policy = silence_policy or _silence_policy_config()
    retry_silence_seconds = float(silence_policy["hardRetrySeconds"])
    jobs, parent_chunks = _build_v3_batch_jobs(
        tts_chunks,
        voice=voice,
        mode=mode,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_chars=max_chars,
        max_tokens=max_tokens,
        batch_size=batch_size,
    )

    if getattr(tts, "_v3_batch_engine", None) is None:
        tts._v3_batch_engine = V3TurboBatchEngine(tts.engine)
    batch_engine = tts._v3_batch_engine

    job_wavs: list[np.ndarray | None] = [None] * len(jobs)
    batch_reports: list[dict[str, Any]] = []
    batch_reports_by_index: dict[int, dict[str, Any]] = {}
    for batch_index, start_index in enumerate(range(0, len(jobs), batch_size)):
        group_jobs = jobs[start_index:start_index + batch_size]
        chunk_indexes = sorted({int(job["parentIndex"]) for job in group_jobs})
        chunk_char_counts = [len(tts_chunks[index]) for index in chunk_indexes]
        job_char_counts = [len(str(job["text"])) for job in group_jobs]
        group_requests = [
            {
                "phonemes": phonemize_text_with_emotions(job["text"]),
                "ref_codes": ref_codes,
                "voice_token_id": voice_token_id,
            }
            for job in group_jobs
        ]
        group_max_new_frames = _ref_audio_batch_max_new_frames([str(job["text"]) for job in group_jobs]) if _is_v3_ref_audio_voice(voice) else 300
        gpu_memory_before = _gpu_memory_snapshot()
        group_start = _now()
        group_wavs = batch_engine.generate_batch(
            group_requests,
            temperature=temperature,
            top_k=top_k,
            top_p=float(top_p if top_p is not None else DEFAULT_V3_TURBO_TOP_P),
            repetition_penalty=float(repetition_penalty if repetition_penalty is not None else DEFAULT_V3_TURBO_REPETITION_PENALTY),
            max_new_frames=group_max_new_frames,
        )
        group_ms = _elapsed_ms(group_start)
        gpu_memory_after = _gpu_memory_snapshot()
        batch_report = {
            "batchIndex": batch_index,
            "jobCount": len(group_jobs),
            "durationMs": group_ms,
            "batchSynthMs": group_ms,
            "firstAttemptMs": group_ms,
            "retryMs": 0,
            "chunkIndexes": chunk_indexes,
            "parentIndexes": chunk_indexes,
            "jobIndexes": [int(job["jobIndex"]) for job in group_jobs],
            "chunkCharCounts": chunk_char_counts,
            "jobCharCounts": job_char_counts,
            "batchCharCount": sum(job_char_counts),
            "parentChunkCharCount": sum(chunk_char_counts),
            "maxNewFrames": group_max_new_frames,
            "refAudioFrameCapApplied": bool(_is_v3_ref_audio_voice(voice)),
            "topP": float(top_p if top_p is not None else DEFAULT_V3_TURBO_TOP_P),
            "repetitionPenalty": float(repetition_penalty if repetition_penalty is not None else DEFAULT_V3_TURBO_REPETITION_PENALTY),
            "retryChunkIndexes": [],
            "maxChunkDurationSeconds": 0.0,
            "gpuMemoryBefore": gpu_memory_before,
            "gpuMemoryAfter": gpu_memory_after,
        }
        batch_reports.append(batch_report)
        batch_reports_by_index[batch_index] = batch_report
        estimated_job_ms = int(round(group_ms / max(1, len(group_jobs))))
        for job, wav in zip(group_jobs, group_wavs):
            job["batchIndex"] = batch_index
            job["batchDurationMs"] = group_ms
            job["estimatedInferDurationMs"] = estimated_job_ms
            job_wavs[int(job["jobIndex"])] = np.asarray(wav, dtype=np.float32).squeeze() if wav is not None else None

    jobs_by_parent: dict[int, list[dict[str, Any]]] = {}
    for job in jobs:
        jobs_by_parent.setdefault(int(job["parentIndex"]), []).append(job)

    retry_count = 0
    for parent in parent_chunks:
        parent_index = int(parent["parentIndex"])
        parent_jobs = sorted(jobs_by_parent.get(parent_index, []), key=lambda item: int(item["subIndex"]))
        sub_wavs = [
            job_wavs[int(job["jobIndex"])]
            for job in parent_jobs
            if job_wavs[int(job["jobIndex"])] is not None and len(job_wavs[int(job["jobIndex"])]) > 0
        ]
        if sub_wavs:
            parent_audio = join_audio_chunks(sub_wavs, sample_rate, silence_p=0.15)
        else:
            parent_audio = np.zeros(0, dtype=np.float32)

        qa = _chunk_audio_qa(parent_audio, sample_rate, str(parent["text"]))
        _annotate_silence_policy(qa, silence_policy)
        qa["attempt"] = 1
        qa["inferDurationMs"] = sum(int(job.get("estimatedInferDurationMs", 0)) for job in parent_jobs)
        qa["attemptCount"] = 1
        qa["retried"] = False
        qa["attempts"] = [dict(qa)]
        if parent_jobs:
            qa["batchIndexes"] = sorted({int(job.get("batchIndex", 0)) for job in parent_jobs})
            qa["subchunkCount"] = len(parent_jobs)
        if qa.get("silenceRetryRequired"):
            qa["valid"] = False
            qa["reason"] = qa.get("silenceRetryReason") or qa.get("reason") or "suspected_long_silence"
            qa["maxAllowedSilenceSeconds"] = round(retry_silence_seconds, 3)
            qa["attempts"] = [dict(qa)]

        duration_ms = sum(int(job.get("estimatedInferDurationMs", 0)) for job in parent_jobs)
        if not qa.get("valid"):
            retry_count += 1
            retry_start = _now()
            _, retried_audio, retry_ms, retry_qa = _infer_one_chunk(
                tts=tts,
                index=parent_index,
                chunk_text=str(parent["text"]),
                infer_kwargs=_build_chunk_infer_kwargs(
                    chunk_text=str(parent["text"]),
                    voice=voice,
                    mode=mode,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_chars=max_chars,
                    max_tokens=max_tokens,
                ),
                sample_rate=sample_rate,
                chunk_audio_retries=chunk_audio_retries,
                use_lock=True,
                max_silence_seconds=retry_silence_seconds,
                silence_policy=silence_policy,
                retry_variant_offset=1,
            )
            retry_wall_ms = _elapsed_ms(retry_start)
            parent_audio = retried_audio if retried_audio is not None else parent_audio
            retry_attempts = [dict(qa)] + list(retry_qa.get("attempts") or [])
            retry_qa = {
                **retry_qa,
                "attemptCount": int(retry_qa.get("attemptCount") or 1) + 1,
                "retried": True,
                "attempts": retry_attempts,
                "batchRetry": True,
                "retryWallMs": retry_wall_ms,
            }
            qa = retry_qa
            duration_ms += retry_ms
            for batch_index_for_parent in sorted({int(job.get("batchIndex", 0)) for job in parent_jobs}):
                batch_report = batch_reports_by_index.get(batch_index_for_parent)
                if batch_report is not None:
                    batch_report["retryMs"] = int(batch_report.get("retryMs", 0)) + retry_ms
                    retry_chunk_indexes = list(batch_report.get("retryChunkIndexes") or [])
                    if parent_index not in retry_chunk_indexes:
                        retry_chunk_indexes.append(parent_index)
                    batch_report["retryChunkIndexes"] = sorted(retry_chunk_indexes)

        _set_chunk_perf_audio_qa(chunk_perf, parent_index, duration_ms, qa)
        for batch_index_for_parent in sorted({int(job.get("batchIndex", 0)) for job in parent_jobs}):
            batch_report = batch_reports_by_index.get(batch_index_for_parent)
            if batch_report is not None:
                batch_report["maxChunkDurationSeconds"] = round(
                    max(
                        float(batch_report.get("maxChunkDurationSeconds") or 0.0),
                        float(qa.get("durationSeconds") or 0.0),
                    ),
                    3,
                )
        wav_chunks[parent_index] = parent_audio

    volume_consistency = _apply_volume_outlier_retry(
        tts=tts,
        voice=voice,
        mode=mode,
        sample_rate=sample_rate,
        tts_chunks=tts_chunks,
        wav_chunks=wav_chunks,
        chunk_perf=chunk_perf,
        jobs_by_parent=jobs_by_parent,
        batch_reports_by_index=batch_reports_by_index,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_chars=max_chars,
        max_tokens=max_tokens,
        retry_silence_seconds=retry_silence_seconds,
        silence_policy=silence_policy,
    )

    metadata = {
        "synthStrategy": "v3_batch_cuda",
        "sampler": {
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p if top_p is not None else DEFAULT_V3_TURBO_TOP_P),
            "repetition_penalty": float(repetition_penalty if repetition_penalty is not None else DEFAULT_V3_TURBO_REPETITION_PENALTY),
            "max_chars": int(max_chars),
        },
        "frameCap": {
            "mode": "ref_audio_dynamic" if _is_v3_ref_audio_voice(voice) else "default",
            "maxNewFrames": None,
            "dynamicRefAudioCapEnabled": bool(_is_v3_ref_audio_voice(voice)),
        },
        "batchSynthMs": _elapsed_ms(batch_start),
        "v3BatchSize": batch_size,
        "v3BatchJobCount": len(jobs),
        "v3BatchCount": len(batch_reports),
        "v3BatchRetries": retry_count,
        "v3BatchRetrySilenceSeconds": retry_silence_seconds,
        "silencePolicy": silence_policy,
        "v3BatchReports": batch_reports,
        "volumeConsistency": volume_consistency,
        "refAudioOptimization": ref_audio_optimization,
        "warnings": [
            "warning_volume_outlier"
            for _ in range(1)
            if int(volume_consistency.get("warningVolumeOutlierCount") or 0) > 0
        ],
    }
    return [chunk for chunk in wav_chunks if chunk is not None and len(chunk) > 0], chunk_perf, metadata

def _synthesize_tts_chunks(
    *,
    tts: Any,
    voice: Any,
    mode: str,
    sample_rate: int,
    tts_chunks: list[str],
    chunk_reports: list[dict[str, Any]],
    temperature: float,
    top_k: int,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    max_chars: int,
    max_tokens: int,
    concurrency: int,
    chunk_audio_retries: int,
    silence_policy: dict[str, float] | None = None,
    v3_batch_size: int | None = None,
    batch_size_source: str | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    if _is_v3_cuda_batch_ready(tts, mode):
        batch_size, resolved_batch_size_source = _resolve_v3_batch_size(v3_batch_size)
        if batch_size > 1 and len(tts_chunks) > 1:
            wav_chunks, chunk_perf, metadata = _synthesize_v3_batch_cuda_chunks(
                tts=tts,
                voice=voice,
                mode=mode,
                sample_rate=sample_rate,
                tts_chunks=tts_chunks,
                chunk_reports=chunk_reports,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_chars=max_chars,
                max_tokens=max_tokens,
                chunk_audio_retries=chunk_audio_retries,
                batch_size=batch_size,
                silence_policy=silence_policy,
            )
            metadata["batchSizeSource"] = batch_size_source or resolved_batch_size_source
            return wav_chunks, chunk_perf, metadata

    wav_chunks: list[np.ndarray | None] = [None] * len(tts_chunks)
    chunk_perf = _init_chunk_perf(tts_chunks, chunk_reports)
    jobs = [
        (
            index,
            chunk_text,
            _build_chunk_infer_kwargs(
                chunk_text=chunk_text,
                voice=voice,
                mode=mode,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_chars=max_chars,
                max_tokens=max_tokens,
            ),
        )
        for index, chunk_text in enumerate(tts_chunks)
    ]
    metadata = {
        "synthStrategy": "sequential",
        "batchSynthMs": 0,
        "v3BatchSize": _resolve_v3_batch_size(v3_batch_size)[0] if _is_v3_cuda_batch_ready(tts, mode) else 1,
        "batchSizeSource": batch_size_source or (_resolve_v3_batch_size(v3_batch_size)[1] if _is_v3_cuda_batch_ready(tts, mode) else "default"),
        "v3BatchJobCount": len(jobs),
        "v3BatchCount": 0,
        "v3BatchRetries": 0,
        "sampler": {
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p if top_p is not None else DEFAULT_V3_TURBO_TOP_P) if mode != "standard" else None,
            "repetition_penalty": float(repetition_penalty if repetition_penalty is not None else DEFAULT_V3_TURBO_REPETITION_PENALTY) if mode != "standard" else None,
            "max_chars": int(max_chars),
        },
        "frameCap": {
            "mode": "default",
            "maxNewFrames": None,
        },
    }

    if concurrency <= 1 or len(jobs) <= 1:
        for index, chunk_text, infer_kwargs in jobs:
            _, audio, duration_ms, audio_qa = _infer_one_chunk(
                tts=tts,
                index=index,
                chunk_text=chunk_text,
                infer_kwargs=infer_kwargs,
                sample_rate=sample_rate,
                chunk_audio_retries=chunk_audio_retries,
                use_lock=True,
                silence_policy=silence_policy,
            )
            _set_chunk_perf_audio_qa(chunk_perf, index, duration_ms, audio_qa)
            wav_chunks[index] = audio
        return [chunk for chunk in wav_chunks if chunk is not None and len(chunk) > 0], chunk_perf, metadata

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _infer_one_chunk,
                tts=tts,
                index=index,
                chunk_text=chunk_text,
                infer_kwargs=infer_kwargs,
                sample_rate=sample_rate,
                chunk_audio_retries=chunk_audio_retries,
                use_lock=False,
                silence_policy=silence_policy,
            )
            for index, chunk_text, infer_kwargs in jobs
        ]
        for future in as_completed(futures):
            index, audio, duration_ms, audio_qa = future.result()
            _set_chunk_perf_audio_qa(chunk_perf, index, duration_ms, audio_qa)
            wav_chunks[index] = audio

    metadata["synthStrategy"] = "threaded"
    return [chunk for chunk in wav_chunks if chunk is not None and len(chunk) > 0], chunk_perf, metadata

def _estimate_words_for_interval(start: float, end: float, words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [word for word in words if word.get("start", 0.0) >= start and word.get("end", 0.0) <= end]

def _resolve_alignment_runtime(warnings: list[str]) -> dict[str, Any]:
    whisper_model_name = os.getenv("VIE_TTS_ALIGNMENT_MODEL") or os.getenv("VIENEU_WHISPER_MODEL", "base")
    alignment_device = (os.getenv("VIE_TTS_ALIGNMENT_DEVICE") or "cpu").strip().lower()
    if alignment_device not in {"cpu", "cuda"}:
        warnings.append(f"Unsupported VIE_TTS_ALIGNMENT_DEVICE={alignment_device}; using cpu")
        alignment_device = "cpu"
    compute_type = os.getenv("VIE_TTS_ALIGNMENT_COMPUTE_TYPE") or ("int8" if alignment_device == "cpu" else "float16")
    alignment_threads = _env_int(
        "VIE_TTS_ALIGNMENT_CPU_THREADS",
        int(_resolve_cpu_threads().get("resolved") or 0),
        0,
        os.cpu_count() or 1,
    )
    return {
        "model_name": whisper_model_name,
        "device": alignment_device,
        "compute_type": compute_type,
        "cpu_threads": alignment_threads,
    }

def _get_faster_whisper_model(config: dict[str, Any]) -> Any:
    key = (
        config["model_name"],
        config["device"],
        config["compute_type"],
        int(config.get("cpu_threads") or 0),
    )
    with _whisper_model_lock:
        model = _whisper_model_cache.get(key)
        if model is not None:
            return model

        from faster_whisper import WhisperModel  # type: ignore

        whisper_kwargs: dict[str, Any] = {
            "device": config["device"],
            "compute_type": config["compute_type"],
        }
        if int(config.get("cpu_threads") or 0) > 0:
            whisper_kwargs["cpu_threads"] = int(config["cpu_threads"])
            whisper_kwargs["num_workers"] = 1
        model = WhisperModel(config["model_name"], **whisper_kwargs)
        _whisper_model_cache[key] = model
        return model

def _preload_faster_whisper_for_v3(requested_method: str, warnings: list[str]) -> int:
    method = (requested_method or "whisper").strip().lower()
    if method not in {"whisper", "faster_whisper"}:
        return 0
    start = _now()
    try:
        _get_faster_whisper_model(_resolve_alignment_runtime(warnings))
        return _elapsed_ms(start)
    except ImportError:
        return 0
    except Exception as exc:
        warnings.append(f"faster_whisper preload failed: {exc}")
        return _elapsed_ms(start)

def _run_whisper_alignment(audio_path: Path, requested_method: str, warnings: list[str]) -> tuple[list[dict[str, Any]], str, str | None, dict[str, Any]]:
    method = (requested_method or "whisper").strip().lower()
    alignment_config = _resolve_alignment_runtime(warnings)
    alignment_report: dict[str, Any] = {
        "requestedMethod": requested_method,
        "method": method,
        "model": alignment_config["model_name"],
        "device": alignment_config["device"],
        "computeType": alignment_config["compute_type"],
        "cpuThreads": alignment_config["cpu_threads"] or None,
        "effectiveDevice": alignment_config["device"],
        "effectiveComputeType": alignment_config["compute_type"],
        "backend": None,
    }
    if method not in {"whisper", "faster_whisper", "openai_whisper"}:
        alignment_report["backend"] = "unsupported"
        alignment_report["fallbackReason"] = f"unsupported alignment_method: {requested_method}"
        return [], "word_ratio_fallback", f"unsupported alignment_method: {requested_method}", alignment_report

    try:
        model = _get_faster_whisper_model(alignment_config)
        alignment_report["backend"] = "faster_whisper"
        segments, _ = model.transcribe(
            str(audio_path),
            language="vi",
            vad_filter=False,
            word_timestamps=True,
        )
        words: list[dict[str, Any]] = []
        for segment in segments:
            segment_words = getattr(segment, "words", None) or []
            for word in segment_words:
                words.append(
                    {
                        "word": str(getattr(word, "word", "")).strip(),
                        "start": float(getattr(word, "start", 0.0) or 0.0),
                        "end": float(getattr(word, "end", 0.0) or 0.0),
                        "probability": float(getattr(word, "probability", 0.75) or 0.75),
                    }
                )
        if words:
            return words, "whisper", None, alignment_report
        alignment_report["fallbackReason"] = "faster_whisper returned no word timestamps"
        return [], "word_ratio_fallback", "faster_whisper returned no word timestamps", alignment_report
    except ImportError:
        warnings.append("faster_whisper is not installed")
        alignment_report["fasterWhisperError"] = "not installed"
    except Exception as exc:
        warnings.append(f"faster_whisper failed: {exc}")
        alignment_report["fasterWhisperError"] = str(exc)

    try:
        import whisper  # type: ignore

        alignment_report["backend"] = "openai_whisper"
        alignment_report["effectiveComputeType"] = "float16" if alignment_config["device"] == "cuda" else "float32"
        model = whisper.load_model(alignment_config["model_name"], device=alignment_config["device"])
        result = model.transcribe(
            str(audio_path),
            language="vi",
            word_timestamps=True,
            fp16=alignment_config["device"] == "cuda",
        )
        words = []
        for segment in result.get("segments", []):
            segment_words = segment.get("words") or []
            if segment_words:
                for word in segment_words:
                    words.append(
                        {
                            "word": str(word.get("word", "")).strip(),
                            "start": float(word.get("start", 0.0) or 0.0),
                            "end": float(word.get("end", 0.0) or 0.0),
                            "probability": float(word.get("probability", 0.75) or 0.75),
                        }
                    )
                continue

            estimated_words = _normalize_words(segment.get("text", ""))
            if not estimated_words:
                continue
            start = float(segment.get("start", 0.0) or 0.0)
            end = float(segment.get("end", start) or start)
            step = max(0.01, (end - start) / len(estimated_words))
            for idx, word_text in enumerate(estimated_words):
                words.append(
                    {
                        "word": word_text,
                        "start": start + idx * step,
                        "end": start + (idx + 1) * step,
                        "probability": 0.6,
                    }
                )
        if words:
            return words, "whisper", None, alignment_report
        alignment_report["fallbackReason"] = "openai whisper returned no timestamps"
        return [], "word_ratio_fallback", "openai whisper returned no timestamps", alignment_report
    except ImportError:
        warnings.append("openai-whisper is not installed")
        alignment_report["openaiWhisperError"] = "not installed"
    except Exception as exc:
        warnings.append(f"openai-whisper failed: {exc}")
        alignment_report["openaiWhisperError"] = str(exc)

    alignment_report["backend"] = alignment_report.get("backend") or "none"
    alignment_report["fallbackReason"] = "no local Whisper aligner is installed"
    return [], "word_ratio_fallback", "no local Whisper aligner is installed", alignment_report

def _timeline_from_whisper_words(
    segments: list[dict[str, Any]],
    words: list[dict[str, Any]],
    total_duration: float,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    source_word_total = sum(max(1, segment["word_count"]) for segment in segments)
    if not words:
        return [], True, "no aligned words"
    if len(words) < max(2, int(source_word_total * 0.55)):
        return [], True, "low confidence or no matching text"

    timeline: list[dict[str, Any]] = []
    cursor = 0
    last_end = 0.0
    for idx, segment in enumerate(segments):
        needed = max(1, segment["word_count"])
        remaining_segments = len(segments) - idx - 1
        max_allowed_end = max(cursor + 1, len(words) - remaining_segments)
        end_idx = min(cursor + needed, max_allowed_end)
        if idx == len(segments) - 1:
            end_idx = len(words)
        selected = words[cursor:end_idx]
        if not selected:
            return [], True, "empty word slice while mapping segments"

        start = max(last_end, float(selected[0]["start"]))
        end = max(start + 0.01, float(selected[-1]["end"]))
        end = min(max(end, start + 0.01), total_duration)
        probs = [float(word.get("probability", 0.75)) for word in selected if word.get("probability") is not None]
        avg_prob = sum(probs) / len(probs) if probs else 0.75
        coverage = min(1.0, len(selected) / needed)
        confidence = round(max(0.45, min(0.95, 0.55 * coverage + 0.4 * avg_prob)), 3)
        timeline.append(
            {
                "id": segment["id"],
                "text": segment["text"],
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "word_count": segment["word_count"],
                "alignment_confidence": confidence,
            }
        )
        cursor = end_idx
        last_end = end

    return _ensure_valid_timeline(timeline, total_duration), False, None

def _timeline_from_word_ratio(
    segments: list[dict[str, Any]],
    total_duration: float,
) -> list[dict[str, Any]]:
    total_words = sum(max(1, segment["word_count"]) for segment in segments)
    cursor = 0.0
    timeline: list[dict[str, Any]] = []
    for idx, segment in enumerate(segments):
        if idx == len(segments) - 1:
            end = total_duration
        else:
            share = max(1, segment["word_count"]) / total_words
            end = cursor + total_duration * share
        end = max(cursor + 0.01, min(end, total_duration))
        timeline.append(
            {
                "id": segment["id"],
                "text": segment["text"],
                "start": round(cursor, 3),
                "end": round(end, 3),
                "duration": round(end - cursor, 3),
                "word_count": segment["word_count"],
                "alignment_confidence": 0.35,
            }
        )
        cursor = end
    return _ensure_valid_timeline(timeline, total_duration)

def _ensure_valid_timeline(timeline: list[dict[str, Any]], total_duration: float) -> list[dict[str, Any]]:
    previous_end = 0.0
    for idx, item in enumerate(timeline):
        start = max(previous_end, float(item["start"]))
        end = max(start + 0.01, float(item["end"]))
        if idx == len(timeline) - 1:
            end = max(start + 0.01, total_duration)
        end = min(end, total_duration)
        if end <= start:
            end = min(total_duration, start + 0.01)
        item["start"] = round(start, 3)
        item["end"] = round(end, 3)
        item["duration"] = round(max(0.01, end - start), 3)
        previous_end = end
    return timeline



def _get_tts(
    mode: str,
    device: str = "cpu",
    backbone_device: str = "cpu",
    codec_device: str = "cpu",
    emotion: str = "natural",
    hf_token: str | None = None,
    backbone_repo: str | None = None,
    codec_repo: str | None = None,
    decoder_repo: str | None = None,
    gguf_filename: str | None = None,
    moss_tokenizer: str | None = None,
    dtype: str | None = None,
):
    global _tts_instance, _tts_key

    cpu_thread_config = _resolve_cpu_threads()
    resolved_hf_token = _resolve_hf_token(hf_token)
    key = _build_tts_key(
        mode,
        device,
        backbone_device,
        codec_device,
        emotion,
        resolved_hf_token,
        backbone_repo,
        codec_repo,
        decoder_repo,
        gguf_filename,
        moss_tokenizer,
        dtype,
        cpu_thread_config["key"],
    )
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
        if resolved_hf_token:
            kwargs["hf_token"] = resolved_hf_token
        _apply_python_thread_settings(cpu_thread_config.get("resolved"))

        if mode == "standard":
            kwargs["backbone_device"] = backbone_device
            kwargs["codec_device"] = codec_device
            kwargs["emotion"] = emotion
            if cpu_thread_config.get("resolved"):
                kwargs["cpu_threads"] = int(cpu_thread_config["resolved"])
            if backbone_repo:
                kwargs["backbone_repo"] = backbone_repo
            if codec_repo:
                kwargs["codec_repo"] = codec_repo
            if gguf_filename is not None:
                kwargs["gguf_filename"] = gguf_filename
        elif mode == "v3turbo":
            kwargs["device"] = device
            if dtype:
                kwargs["dtype"] = dtype
            if backbone_repo:
                kwargs["backbone_repo"] = backbone_repo
            if moss_tokenizer:
                kwargs["moss_tokenizer"] = moss_tokenizer
        elif mode in {"turbo", "turbo_gpu"}:
            kwargs["device"] = device
            if backbone_repo:
                kwargs["backbone_repo"] = backbone_repo
            if decoder_repo:
                kwargs["decoder_repo"] = decoder_repo
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
    if mode == "v3turbo" and getattr(tts, "engine", None) is None:
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
    model: str | None,
    mode: str | None,
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

    cfg = _resolve_model_config(model, mode, device, backbone_device, codec_device)
    mode = cfg["mode"]
    tts = _get_tts(
        mode=mode,
        device=cfg["device"],
        backbone_device=cfg["backbone_device"],
        codec_device=cfg["codec_device"],
        emotion=emotion,
        hf_token=hf_token,
        backbone_repo=cfg.get("backbone_repo"),
        codec_repo=cfg.get("codec_repo"),
        decoder_repo=cfg.get("decoder_repo"),
        gguf_filename=cfg.get("gguf_filename"),
        moss_tokenizer=cfg.get("moss_tokenizer"),
        dtype=cfg.get("dtype"),
    )

    if mode == "standard":
        if not ref_text or not ref_text.strip():
            raise HTTPException(status_code=400, detail="ref_text is required for standard mode")
        voice = {"codes": tts.encode_reference(ref_audio_path), "text": ref_text.strip()}
    elif mode == "v3turbo":
        voice = {"codes": tts.encode_reference(ref_audio_path)}
    else:
        voice = tts.encode_reference(ref_audio_path)

    effective_sampler = _resolve_effective_sampler(
        mode=mode,
        cfg_id=cfg["id"],
        temperature=temperature,
        top_k=top_k,
        top_p=None,
        repetition_penalty=None,
    )
    infer_kwargs = {
        "text": text.strip(),
        "voice": voice,
        "temperature": float(effective_sampler["temperature"]),
        "top_k": int(effective_sampler["top_k"]),
        "max_chars": max_chars,
        "apply_watermark": apply_watermark,
    }
    if mode != "standard":
        infer_kwargs["max_tokens"] = max_tokens
        infer_kwargs["top_p"] = effective_sampler.get("top_p")
        infer_kwargs["repetition_penalty"] = effective_sampler.get("repetition_penalty")

    return cfg, tts, tts.infer(**infer_kwargs)


@app.get("/health")
def health():
    silence_policy = _silence_policy_config()
    return {
        "status": "ok",
        "defaultModel": DEFAULT_API_MODEL,
        "defaultMode": DEFAULT_API_MODE,
        "defaultDevice": DEFAULT_API_DEVICE,
        "defaultRefAudio": DEFAULT_REF_AUDIO_REL,
        "defaultRefAudioPath": str(DEFAULT_REF_AUDIO_PATH),
        "defaultVoice": DEFAULT_REF_VOICE_ID,
        "defaultVoiceLabel": DEFAULT_REF_VOICE_LABEL,
        "voiceType": "ref_audio",
        "requiresRefAudio": True,
        "noSilentFallbackToV2": True,
        "supportsV3BatchSizeOverride": True,
        "defaultV3BatchSize": _get_v3_batch_size(),
        "v3BatchSizeRange": {"min": 1, "max": MAX_V3_BATCH_SIZE},
        "v3BatchRetrySilenceSeconds": float(silence_policy["hardRetrySeconds"]),
        "v3BatchSoftWarnSilenceSeconds": float(silence_policy["softWarnSeconds"]),
        "v3BatchSevereSilenceSeconds": float(silence_policy["severeSeconds"]),
        "silencePolicy": silence_policy,
    }


@app.get("/models")
def list_models():
    models = sorted(
        API_MODEL_PRESETS.values(),
        key=lambda item: (not bool(item.get("default")), str(item.get("id", ""))),
    )
    return {
        "default_model": DEFAULT_API_MODEL,
        "defaultModel": DEFAULT_API_MODEL,
        "defaultMode": DEFAULT_API_MODE,
        "defaultDevice": DEFAULT_API_DEVICE,
        "defaultRefAudio": DEFAULT_REF_AUDIO_REL,
        "models": [
            {
                **model,
                "value": model["id"],
                "label": model.get("label") or model.get("name") or model["id"],
                "mode": _public_mode(model.get("mode")),
                "internal_mode": model.get("mode"),
                "voiceType": model.get("voice_type"),
                "default": bool(model.get("default")),
            }
            for model in models
        ],
    }


@app.get("/voices")
def list_voices(
    model: str | None = Query(default=None),
    mode: str | None = Query(default=None),
    device: str = Query(default="cpu"),
    backbone_device: str = Query(default="cpu"),
    codec_device: str = Query(default="cpu"),
):
    cfg = _resolve_model_config(model, mode, device, backbone_device, codec_device)
    result: list[dict[str, Any]] = []
    tts = _get_tts(
        mode=cfg["mode"],
        device=cfg["device"],
        backbone_device=cfg["backbone_device"],
        codec_device=cfg["codec_device"],
        backbone_repo=cfg.get("backbone_repo"),
        codec_repo=cfg.get("codec_repo"),
        decoder_repo=cfg.get("decoder_repo"),
        gguf_filename=cfg.get("gguf_filename"),
        moss_tokenizer=cfg.get("moss_tokenizer"),
        dtype=cfg.get("dtype"),
    )
    voices = tts.list_preset_voices()
    if cfg["id"] == DEFAULT_API_MODEL:
        result.append({
            "id": NGUYEN_HUYEN_TRANG_VOICE_ID,
            "value": NGUYEN_HUYEN_TRANG_VOICE_ID,
            "name": NGUYEN_HUYEN_TRANG_VOICE_LABEL,
            "label": NGUYEN_HUYEN_TRANG_VOICE_LABEL,
            "type": "ref_audio",
            "voiceType": "ref_audio",
            "voiceSource": "production-ref-audio",
            "refAudioPath": DEFAULT_REF_AUDIO_REL,
            "supportedModels": [DEFAULT_API_MODEL],
            "usable": DEFAULT_REF_AUDIO_PATH.exists(),
            "disabled": not DEFAULT_REF_AUDIO_PATH.exists(),
            "warning": None if DEFAULT_REF_AUDIO_PATH.exists() else f"Missing ref audio: {DEFAULT_REF_AUDIO_REL}",
            "default": False,
            "aliases": [
                "Nguyễn Huyền Trang",
                "Nguyen Huyen Trang",
            ],
        })
    for item in voices:
        if isinstance(item, tuple) and len(item) == 2:
            desc, voice_id = item
            usable, warning = (True, None)
            ref_path = None
            if cfg["id"] == DEFAULT_API_MODEL:
                usable, warning = _v3_preset_voice_usability(tts, str(voice_id))
                ref_path = _v3_preset_ref_audio_path(tts, str(voice_id))
            result.append({
                "id": str(voice_id),
                "value": str(voice_id),
                "name": str(desc),
                "label": str(desc),
                "type": "ref_audio" if ref_path is not None else "preset",
                "voiceType": "ref_audio" if ref_path is not None else "preset",
                "refAudioPath": ref_path.relative_to(ROOT_DIR).as_posix() if ref_path is not None else None,
                "usable": usable,
                "disabled": not usable,
                "warning": warning,
                "default": False,
            })
        else:
            usable, warning = (True, None)
            ref_path = None
            if cfg["id"] == DEFAULT_API_MODEL:
                usable, warning = _v3_preset_voice_usability(tts, str(item))
                ref_path = _v3_preset_ref_audio_path(tts, str(item))
            result.append({
                "id": str(item),
                "value": str(item),
                "name": str(item),
                "label": str(item),
                "type": "ref_audio" if ref_path is not None else "preset",
                "voiceType": "ref_audio" if ref_path is not None else "preset",
                "refAudioPath": ref_path.relative_to(ROOT_DIR).as_posix() if ref_path is not None else None,
                "usable": usable,
                "disabled": not usable,
                "warning": warning,
                "default": False,
            })
    return {
        "model": cfg["id"],
        "mode": _public_mode(cfg["mode"]),
        "internal_mode": cfg["mode"],
        "defaultVoice": DEFAULT_REF_VOICE_ID if cfg["id"] == DEFAULT_API_MODEL else None,
        "defaultRefAudio": DEFAULT_REF_AUDIO_REL if cfg["id"] == DEFAULT_API_MODEL else None,
        "voices": result,
    }


@app.post("/clone")
async def clone_voice(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: str | None = Form(default=None),
    model: str | None = Form(default=None),
    mode: str | None = Form(default=None),
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
        cfg, tts, audio = _infer_clone(
            text=text,
            ref_audio_path=tmp_path,
            ref_text=ref_text,
            model=model,
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
    model: str | None = Form(default=None),
    mode: str | None = Form(default=None),
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
        cfg, tts, audio = _infer_clone(
            text=text,
            ref_audio_path=tmp_path,
            ref_text=ref_text,
            model=model,
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
            "model": cfg["id"],
            "mode": _public_mode(cfg["mode"]),
            "internal_mode": cfg["mode"],
            "sample_rate": int(getattr(tts, "sample_rate", 24000) or 24000),
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
        cfg = _resolve_model_config(
            req.model,
            req.mode,
            req.device,
            req.backbone_device,
            req.codec_device,
        )
        with _model_lock:
            tts = _get_tts(
                mode=cfg["mode"],
                device=cfg["device"],
                backbone_device=cfg["backbone_device"],
                codec_device=cfg["codec_device"],
                hf_token=req.hf_token,
                backbone_repo=cfg.get("backbone_repo"),
                codec_repo=cfg.get("codec_repo"),
                decoder_repo=cfg.get("decoder_repo"),
                gguf_filename=cfg.get("gguf_filename"),
                moss_tokenizer=cfg.get("moss_tokenizer"),
                dtype=cfg.get("dtype"),
            )
            if req.ref_audio:
                ref_path = Path(req.ref_audio)
                if not ref_path.is_absolute():
                    ref_path = ROOT_DIR / ref_path
                if not ref_path.exists():
                    raise HTTPException(status_code=400, detail=f"ref_audio does not exist: {req.ref_audio}")
                if cfg["mode"] == "standard":
                    if not req.ref_text or not req.ref_text.strip():
                        raise HTTPException(status_code=400, detail="ref_text is required for standard mode when ref_audio is used")
                    voice = {"codes": tts.encode_reference(str(ref_path)), "text": req.ref_text.strip()}
                elif cfg["mode"] == "v3turbo":
                    voice = {"codes": tts.encode_reference(str(ref_path))}
                else:
                    voice = tts.encode_reference(str(ref_path))
            else:
                if cfg["id"] == DEFAULT_API_MODEL and not req.preset_voice:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "VieNeu v3 Turbo GPU requires ref_audio; no silent fallback to v2 preset voice. "
                            f"Use ref_audio='{DEFAULT_REF_AUDIO_REL}'."
                        ),
                    )
                if not req.preset_voice:
                    raise HTTPException(status_code=400, detail="preset_voice is required when ref_audio is not provided")
                requested_voice = str(req.preset_voice or "").strip()
                requested_voice_key = " ".join(_normalize_words(requested_voice))
                nguyen_huyen_key = " ".join(_normalize_words(NGUYEN_HUYEN_TRANG_VOICE_ID))
                if cfg["mode"] == "v3turbo" and requested_voice_key == nguyen_huyen_key:
                    if not DEFAULT_REF_AUDIO_PATH.exists():
                        raise HTTPException(status_code=400, detail=f"ref_audio does not exist: {DEFAULT_REF_AUDIO_REL}")
                    ref_codes, _ = _encode_v3_ref_audio_cached(tts, DEFAULT_REF_AUDIO_PATH)
                    voice = {
                        "codes": ref_codes,
                        "voice_type": "ref_audio",
                        "_voice_type": "ref_audio",
                        "ref_audio": str(DEFAULT_REF_AUDIO_PATH),
                        "refAudioPath": str(DEFAULT_REF_AUDIO_PATH),
                        "presetVoiceId": NGUYEN_HUYEN_TRANG_VOICE_ID,
                    }
                else:
                    voice_id = _resolve_preset_voice_id(tts, req.preset_voice)
                    if cfg["mode"] == "v3turbo":
                        usable, warning = _v3_preset_voice_usability(tts, voice_id)
                        if not usable:
                            raise HTTPException(
                                status_code=422,
                                detail=(
                                    f"Preset voice '{voice_id}' is not compatible with VieNeu v3 Turbo GPU built-in synthesis. "
                                    f"{warning or 'Use a v3 built-in preset with 2D MOSS codes or provide ref_audio.'}"
                                ),
                            )
                        voice = _resolve_v3_preset_voice(tts, voice_id)
                    else:
                        voice = tts.get_preset_voice(voice_id)
            effective_sampler = _resolve_effective_sampler(
                mode=cfg["mode"],
                cfg_id=cfg["id"],
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
            )
            infer_kwargs = {
                "text": req.text.strip(),
                "voice": voice,
                "temperature": float(effective_sampler["temperature"]),
                "top_k": int(effective_sampler["top_k"]),
                "max_chars": req.max_chars,
                "apply_watermark": req.apply_watermark,
            }
            if cfg["mode"] != "standard":
                infer_kwargs["max_tokens"] = req.max_tokens
                infer_kwargs["top_p"] = effective_sampler.get("top_p")
                infer_kwargs["repetition_penalty"] = effective_sampler.get("repetition_penalty")
            audio = tts.infer(**infer_kwargs)
            output_path = _save_audio(tts, audio, "preset")
        return {
            "status": "ok",
            "output_path": str(output_path),
            "model": cfg["id"],
            "mode": _public_mode(cfg["mode"]),
            "internal_mode": cfg["mode"],
            "device": cfg["device"],
            "ref_audio": req.ref_audio,
            "sample_rate": int(getattr(tts, "sample_rate", 24000) or 24000),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/synthesize-aligned")
def synthesize_aligned(req: AlignedSynthesizeRequest):
    request_start = _now()
    validation_start = _now()
    if req.timeline_mode not in {"full_then_align", "full_then_align_verified"}:
        raise HTTPException(status_code=422, detail="Only timeline_mode='full_then_align' or 'full_then_align_verified' is supported")

    clean_segments = _validate_aligned_segments(req.segments)
    total_words = sum(segment["word_count"] for segment in clean_segments)
    if total_words <= 0:
        raise HTTPException(status_code=422, detail="No readable words found in segments")
    if total_words > int(req.max_full_align_words or 1300):
        raise HTTPException(
            status_code=422,
            detail=f"Full alignment script is too long: {total_words} words > {req.max_full_align_words}",
        )

    cfg = _resolve_model_config(
        req.model,
        req.mode,
        req.device,
        req.backbone_device,
        req.codec_device,
    )
    mode = cfg["mode"]
    warnings: list[str] = []
    requested_timeline_mode = req.timeline_mode
    effective_timeline_mode = "full_then_align"
    if requested_timeline_mode == "full_then_align_verified":
        warnings.append("full_then_align_verified maps to full_then_align; Whisper is used only for timing alignment")
    output_suffix = _safe_request_id(req.request_id) if req.request_id else uuid.uuid4().hex[:8]
    output_base = OUTPUT_DIR / f"full_{output_suffix}"
    audio_path = output_base.with_suffix(".wav")
    text_path = output_base.with_suffix(".txt")
    timings_path = output_base.with_suffix(".timings.json")
    parse_validation_ms = _elapsed_ms(validation_start)
    _log_aligned(
        f"[synthesize-aligned] requestId={req.request_id or output_suffix} "
        f"segments={len(clean_segments)} mode={mode} model={cfg['id']}"
    )
    cpu_thread_config = _resolve_cpu_threads()
    thread_env = _thread_env_snapshot()
    torch_thread_info = _apply_python_thread_settings(cpu_thread_config.get("resolved"))
    cpu_sampler = _CpuSampler()
    gpu_sampler = _GpuSampler()
    cpu_sampler.start()
    gpu_sampler.start()
    _log_aligned(
        "[perf] cpuConfig="
        f"cpuCount={cpu_thread_config['cpuCount']} "
        f"VIE_TTS_CPU_THREADS={thread_env.get('VIE_TTS_CPU_THREADS') or 'default'} "
        f"resolved={cpu_thread_config.get('resolved') or 'backend-default'} "
        f"torchThreads={torch_thread_info.get('torchNumThreads')}"
    )

    try:
        model_start = _now()
        with _model_lock:
            tts = _get_tts(
                mode=mode,
                device=cfg["device"],
                backbone_device=cfg["backbone_device"],
                codec_device=cfg["codec_device"],
                emotion=req.emotion,
                hf_token=req.hf_token,
                backbone_repo=cfg.get("backbone_repo"),
                codec_repo=cfg.get("codec_repo"),
                decoder_repo=cfg.get("decoder_repo"),
                gguf_filename=cfg.get("gguf_filename"),
                moss_tokenizer=cfg.get("moss_tokenizer"),
                dtype=cfg.get("dtype"),
            )
            voice = _resolve_aligned_voice(tts, req, mode)
        model_load_voice_ms = _elapsed_ms(model_start)
        _log_aligned(f"[perf] modelLoadVoice={model_load_voice_ms}ms")

        sample_rate = int(getattr(tts, "sample_rate", 24000) or 24000)
        preprocess_start = _now()
        full_text = _build_full_narration_text(clean_segments)
        preprocess_ms = _elapsed_ms(preprocess_start)

        min_chunk_chars = _clamp_int(
            int(req.min_chunk_chars) if req.min_chunk_chars is not None else _env_int("VIE_TTS_MIN_CHUNK_CHARS", DEFAULT_V3_MIN_CHUNK_CHARS, 20, 120),
            20,
            120,
        )
        max_chunk_chars = _clamp_int(
            int(req.max_chunk_chars) if req.max_chunk_chars is not None else _env_int("VIE_TTS_MAX_CHUNK_CHARS", DEFAULT_V3_MAX_CHUNK_CHARS, min_chunk_chars, 320),
            min_chunk_chars,
            320,
        )
        chunk_audio_retries = _clamp_int(
            int(req.chunk_audio_retries) if req.chunk_audio_retries is not None else _env_int("VIE_TTS_CHUNK_AUDIO_RETRIES", DEFAULT_V3_CHUNK_AUDIO_RETRIES, 0, 5),
            0,
            5,
        )
        v3_batch_size, batch_size_source = _resolve_v3_batch_size(req.v3_batch_size)
        chunk_packing_strategy, chunk_packing_source = _resolve_chunk_packing_strategy(req.chunk_packing_strategy)

        chunking_start = _now()
        chunking_plan = _build_sentence_aware_chunk_plan(
            clean_segments,
            req.max_chars,
            min_chunk_chars,
            max_chunk_chars,
            chunk_packing_strategy=chunk_packing_strategy,
        )
        tts_chunks = list(chunking_plan.pop("ttsChunks"))
        if not tts_chunks:
            raise HTTPException(status_code=500, detail="Sentence-aware chunking produced no TTS chunks")
        chunking_ms = _elapsed_ms(chunking_start)
        chunk_reports = list(chunking_plan.get("chunks") or [])
        chunk_concurrency = _env_int(
            "VIE_TTS_CHUNK_CONCURRENCY",
            DEFAULT_TTS_CHUNK_CONCURRENCY,
            1,
            MAX_TTS_CHUNK_CONCURRENCY,
        )
        requested_chunk_concurrency = chunk_concurrency
        if mode == "standard" and chunk_concurrency > 1:
            chunk_concurrency = 1
            warnings.append("VIE_TTS_CHUNK_CONCURRENCY>1 disabled for standard GGUF backend; chunk synthesis is not thread-safe")
        elif chunk_concurrency > 1:
            warnings.append("Experimental VIE_TTS_CHUNK_CONCURRENCY>1 enabled; verify audio continuity before using in production")
        _log_aligned(f"[perf] preprocess={preprocess_ms}ms parseValidation={parse_validation_ms}ms")
        _log_aligned(
            f"[perf] chunking={chunking_ms}ms chunkCount={len(tts_chunks)} "
            f"minChunkChars={min_chunk_chars} maxChunkChars={max_chunk_chars} "
            f"chunkPacking={chunk_packing_strategy} chunkPackingSource={chunk_packing_source} "
            f"chunkRetries={chunk_audio_retries} concurrency={chunk_concurrency} "
            f"v3BatchSize={v3_batch_size} batchSizeSource={batch_size_source}"
        )

        whisper_preload_ms = 0
        if mode == "v3turbo":
            whisper_preload_ms = _preload_faster_whisper_for_v3(req.alignment_method, warnings)
            if whisper_preload_ms:
                _log_aligned(f"[perf] whisperPreload={whisper_preload_ms}ms")

        request_silence_policy = _silence_policy_config() if mode == "v3turbo" else None
        effective_sampler = _resolve_effective_sampler(
            mode=mode,
            cfg_id=cfg["id"],
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
        )
        tts_start = _now()
        wav_chunks, chunk_perf, synth_metadata = _synthesize_tts_chunks(
            tts=tts,
            voice=voice,
            mode=mode,
            sample_rate=sample_rate,
            tts_chunks=tts_chunks,
            chunk_reports=chunk_reports,
            temperature=float(effective_sampler["temperature"]),
            top_k=int(effective_sampler["top_k"]),
            top_p=effective_sampler.get("top_p"),
            repetition_penalty=effective_sampler.get("repetition_penalty"),
            max_chars=int(chunking_plan["maxCharsUsed"]),
            max_tokens=req.max_tokens,
            concurrency=chunk_concurrency,
            chunk_audio_retries=chunk_audio_retries,
            silence_policy=request_silence_policy,
            v3_batch_size=v3_batch_size,
            batch_size_source=batch_size_source,
        )
        for synth_warning in synth_metadata.get("warnings") or []:
            if synth_warning not in warnings:
                warnings.append(str(synth_warning))
        tts_synth_ms = _elapsed_ms(tts_start)
        if len(wav_chunks) != len(tts_chunks):
            raise HTTPException(
                status_code=500,
                detail=f"TTS returned {len(wav_chunks)} non-empty chunks for {len(tts_chunks)} requested chunks",
            )
        slowest_chunk = max(chunk_perf, key=lambda item: int(item.get("durationMs", 0)), default=None)
        if slowest_chunk:
            _log_aligned(
                "[perf] ttsSynth="
                f"{tts_synth_ms}ms slowestChunk={slowest_chunk['index']} "
                f"slowestMs={slowest_chunk['durationMs']} risk={slowest_chunk.get('riskLevel')}"
            )
        else:
            _log_aligned(f"[perf] ttsSynth={tts_synth_ms}ms")
        if synth_metadata.get("synthStrategy") == "v3_batch_cuda":
            _log_aligned(
                f"[perf] batchSynth={synth_metadata.get('batchSynthMs')}ms "
                f"batchSize={synth_metadata.get('v3BatchSize')} "
                f"batchCount={synth_metadata.get('v3BatchCount')} "
                f"jobCount={synth_metadata.get('v3BatchJobCount')}"
            )

        concat_export_start = _now()
        audio = join_audio_chunks(
            wav_chunks,
            sample_rate,
            silence_p=float(chunking_plan.get("interChunkSilenceSeconds", 0.12)),
        )
        if req.apply_watermark:
            apply_watermark = getattr(tts, "_apply_watermark", None)
            if callable(apply_watermark):
                audio = apply_watermark(audio)
            else:
                warnings.append("Watermark requested but active TTS backend has no watermark hook")
        audio = _apply_speed(audio, req.speed)
        if audio is None or len(audio) == 0:
            raise HTTPException(status_code=500, detail="TTS returned empty audio")

        duration = float(len(audio)) / float(sample_rate)
        if duration <= 0:
            raise HTTPException(status_code=500, detail="TTS returned zero-duration audio")

        _save_wav(audio, audio_path, sample_rate)
        concat_export_ms = _elapsed_ms(concat_export_start)
        _log_aligned(f"[perf] concatExport={concat_export_ms}ms")

        whisper_start = _now()
        aligned_words, method_used, fallback_reason, alignment_report = _run_whisper_alignment(
            audio_path,
            req.alignment_method,
            warnings,
        )
        fallback_used = False
        segments, low_confidence, low_confidence_reason = _timeline_from_whisper_words(
            clean_segments,
            aligned_words,
            duration,
        )
        if low_confidence:
            fallback_used = True
            fallback_reason = fallback_reason or low_confidence_reason or "low confidence or no matching text"
            method_used = "word_ratio_fallback"
            segments = _timeline_from_word_ratio(clean_segments, duration)
            warnings.append(f"Alignment fallback used: {fallback_reason}")
        whisper_align_ms = _elapsed_ms(whisper_start)
        alignment_report["durationMs"] = whisper_align_ms
        alignment_report["methodUsed"] = method_used
        alignment_report["fallbackUsed"] = fallback_used
        alignment_report["fallbackReason"] = fallback_reason if fallback_used else alignment_report.get("fallbackReason")
        _log_aligned(f"[perf] whisperAlign={whisper_align_ms}ms method={method_used}")

        if not segments or any(float(segment["duration"]) <= 0 for segment in segments):
            raise HTTPException(status_code=500, detail="Failed to produce valid non-empty segment timings")

        silence_policy = synth_metadata.get("silencePolicy") or _silence_policy_config()
        silence_summary = _summarize_silence_policy(chunk_perf, silence_policy)
        chunk_audio_qa = {
            "enabled": True,
            "minChunkChars": min_chunk_chars,
            "maxChunkChars": max_chunk_chars,
            "chunkAudioRetries": chunk_audio_retries,
            "silencePolicy": silence_summary["policy"],
            "silenceSummary": silence_summary,
            "chunkCount": len(chunk_perf),
            "retryCount": sum(max(0, int(((item.get("chunkAudioQa") or {}).get("attemptCount")) or 1) - 1) for item in chunk_perf),
            "invalidCount": sum(1 for item in chunk_perf if not bool((item.get("chunkAudioQa") or {}).get("valid", True))),
            "maxDurationSeconds": round(
                max((float((item.get("chunkAudioQa") or {}).get("durationSeconds", 0.0)) for item in chunk_perf), default=0.0),
                3,
            ),
            "maxSilenceSeconds": round(
                max((float((item.get("chunkAudioQa") or {}).get("longestSilenceSeconds", 0.0)) for item in chunk_perf), default=0.0),
                3,
            ),
            "suspiciousChunks": [
                {
                    "index": item["index"],
                    "reason": (item.get("chunkAudioQa") or {}).get("reason"),
                    "silenceSeverity": (item.get("chunkAudioQa") or {}).get("silenceSeverity"),
                    "silenceRetryRequired": (item.get("chunkAudioQa") or {}).get("silenceRetryRequired"),
                    "silenceRetryReason": (item.get("chunkAudioQa") or {}).get("silenceRetryReason"),
                    "attemptCount": (item.get("chunkAudioQa") or {}).get("attemptCount"),
                    "durationSeconds": (item.get("chunkAudioQa") or {}).get("durationSeconds"),
                    "longestSilenceSeconds": (item.get("chunkAudioQa") or {}).get("longestSilenceSeconds"),
                    "textPreview": item.get("textPreview"),
                    "sourceSegmentIds": item.get("sourceSegmentIds"),
                    "wasMergedMicroSentence": item.get("wasMergedMicroSentence"),
                }
                for item in chunk_perf
                if not bool((item.get("chunkAudioQa") or {}).get("valid", True))
            ],
            "volumeConsistency": synth_metadata.get(
                "volumeConsistency",
                {
                    "enabled": False,
                    "outlierCount": 0,
                    "retriedVolumeOutlierCount": 0,
                    "warningVolumeOutlierCount": 0,
                    "worstLocalDropDb": 0,
                },
            ),
        }

        total_before_artifacts_ms = _elapsed_ms(request_start)
        cpu_sample_summary = cpu_sampler.stop()
        gpu_sample_summary = gpu_sampler.stop()
        cpu_tuning = {
            "cpuCount": cpu_thread_config["cpuCount"],
            "ttsThreads": cpu_thread_config.get("resolved"),
            "requestedTtsThreads": cpu_thread_config.get("raw"),
            "torchNumThreads": torch_thread_info.get("torchNumThreads"),
            "torchInteropThreads": torch_thread_info.get("torchInteropThreads"),
            "onnxIntraOpThreads": cpu_thread_config.get("resolved"),
            "onnxInterOpThreads": max(1, min(2, int(cpu_thread_config["resolved"]))) if cpu_thread_config.get("resolved") else None,
            "ggufThreads": cpu_thread_config.get("resolved"),
            "alignmentCpuThreads": _env_int(
                "VIE_TTS_ALIGNMENT_CPU_THREADS",
                int(cpu_thread_config.get("resolved") or 0),
                0,
                os.cpu_count() or 1,
            ) or None,
            "threadEnv": thread_env,
            **cpu_sample_summary,
        }
        _log_aligned(
            "[perf] cpu="
            f"avg={cpu_tuning.get('cpuAvgPercent')}% "
            f"peak={cpu_tuning.get('cpuPeakPercent')}% "
            f"memPeakMb={cpu_tuning.get('memoryPeakMb')}"
        )
        performance = {
            "requestId": req.request_id or output_suffix,
            "mode": _public_mode(mode),
            "internalMode": mode,
            "model": cfg["id"],
            "synthStrategy": synth_metadata.get("synthStrategy", "sequential"),
            "hfSamplerDefault": bool(effective_sampler.get("hfSamplerDefault")),
            "sampler": synth_metadata.get("sampler"),
            "frameCap": synth_metadata.get("frameCap"),
            "v3BatchSize": int(synth_metadata.get("v3BatchSize") or v3_batch_size),
            "batchSizeSource": synth_metadata.get("batchSizeSource") or batch_size_source,
            "timelineMode": effective_timeline_mode,
            "requestedTimelineMode": requested_timeline_mode,
            "segmentCount": len(clean_segments),
            "chunkCount": len(tts_chunks),
            "chunkPackingStrategy": chunk_packing_strategy,
            "chunkPackingSource": chunk_packing_source,
            "chunkConcurrency": chunk_concurrency,
            "requestedChunkConcurrency": requested_chunk_concurrency,
            "audioDurationSeconds": round(duration, 3),
            "totalDurationMs": total_before_artifacts_ms,
            "requestParseValidationMs": parse_validation_ms,
            "modelLoadVoiceMs": model_load_voice_ms,
            "preprocessMs": preprocess_ms,
            "chunkingMs": chunking_ms,
            "ttsSynthMs": tts_synth_ms,
            "batchSynthMs": int(synth_metadata.get("batchSynthMs") or 0),
            "ttsSynthChunkSumMs": sum(int(item.get("durationMs", 0)) for item in chunk_perf),
            "concatExportMs": concat_export_ms,
            "whisperPreloadMs": whisper_preload_ms,
            "whisperAlignMs": whisper_align_ms,
            "artifactWriteMs": 0,
            "realtimeSpeed": round(duration / (total_before_artifacts_ms / 1000.0), 3) if total_before_artifacts_ms else None,
            "cpuTuning": cpu_tuning,
            "batchSynth": synth_metadata,
            "chunks": chunk_perf,
            "chunkAudioQa": chunk_audio_qa,
            "alignment": alignment_report,
            "gpuTuning": gpu_sample_summary,
        }
        response = {
            "status": "ok",
            "request_id": req.request_id,
            "audio_path": str(audio_path),
            "text_path": str(text_path),
            "timings_path": str(timings_path),
            "sample_rate": sample_rate,
            "duration": round(duration, 3),
            "model": cfg["id"],
            "mode": _public_mode(mode),
            "internal_mode": mode,
            "device": cfg["device"],
            "ref_audio": req.ref_audio,
            "timeline_mode": effective_timeline_mode,
            "requested_timeline_mode": requested_timeline_mode,
            "alignment_method": method_used,
            "requested_alignment_method": req.alignment_method,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason if fallback_used else None,
            "segments": segments,
            "warnings": warnings,
            **chunking_plan,
            "chunkAudioQa": chunk_audio_qa,
            "verificationDisabled": True,
            "patching": {
                "enabled": False,
            },
            "textMutationAllowed": False,
            "textChangedForTts": False,
            "pronunciationAliasApplied": False,
            "pauseMarkerApplied": False,
            "synthStrategy": synth_metadata.get("synthStrategy", "sequential"),
            "v3BatchSize": int(synth_metadata.get("v3BatchSize") or v3_batch_size),
            "batchSizeSource": synth_metadata.get("batchSizeSource") or batch_size_source,
            "performance": performance,
            "alignment": alignment_report,
        }

        artifact_start = _now()
        text_path.write_text(full_text + "\n", encoding="utf-8")
        response["performance"]["artifactWriteMs"] = _elapsed_ms(artifact_start)
        response["performance"]["totalDurationMs"] = _elapsed_ms(request_start)
        timings_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
        _log_aligned(f"[perf] artifactWrite={response['performance']['artifactWriteMs']}ms")
        _log_aligned(f"[perf] total={response['performance']['totalDurationMs']}ms")
        return response
    except HTTPException:
        cpu_sampler.stop()
        gpu_sampler.stop()
        raise
    except Exception as exc:
        cpu_sampler.stop()
        gpu_sampler.stop()
        _log_aligned("[synthesize-aligned] exception\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


def main():
    host = "127.0.0.1"
    port = 8002
    print(f"Voice Clone API: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
