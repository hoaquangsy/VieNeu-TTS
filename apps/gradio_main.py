import gradio as gr
print("⏳ Đang khởi động VieNeu-TTS... Vui lòng chờ...")
import soundfile as sf
import tempfile
from vieneu import Vieneu
import os
import sys
import time
import numpy as np
import queue
import threading
import yaml
import json
import uuid
import re
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks, env_bool, split_into_chunks_v2, get_silence_duration_v2
from vieneu_utils.phonemize_text import phonemize_with_dict
from sea_g2p import Normalizer
from functools import lru_cache
import gc

# --- CONSTANTS & CONFIG ---
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
USER_VOICES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_voices.json")
PRONUNCIATION_RULES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_pronunciation_rules.txt")
MODEL_SELECTION_STATE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_selection_state.json")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}
except Exception as e:
    raise RuntimeError(f"Không thể đọc config.yaml: {e}")

BACKBONE_CONFIGS = _config.get("backbone_configs", {})
CODEC_CONFIGS = _config.get("codec_configs", {})
DEFAULT_PRONUNCIATION_RULES = "TTS => ti tít\nAI => ây ai\nWAV => quây"

# Refilter and Simplify Configs per requirements
HAS_GPU = False
try:
    import torch
    HAS_GPU = torch.cuda.is_available() or (sys.platform == "darwin" and torch.backends.mps.is_available())
except ImportError:
    pass

filtered_backbones = {}
if HAS_GPU:
    filtered_backbones["VieNeu-TTS-v2 (GPU)"] = {
        "repo": "pnnbao-ump/VieNeu-TTS-v2",
        "supports_streaming": False,
        "description": "VieNeu-TTS Version 2 - hỗ trợ song ngữ (Anh-Việt) và chế độ podcast"
    }
    filtered_backbones["VieNeu-TTS (GPU)"] = {
        "repo": "pnnbao-ump/VieNeu-TTS",
        "supports_streaming": False,
        "description": "VieNeu-TTS Version 1 - ổn định, production-ready"
    }
    filtered_backbones["VieNeu-TTS-0.3B-ngoc-huyen (GPU)"] = {
        "repo": "pnnbao-ump/VieNeu-TTS-0.3B-ngoc-huyen",
        "supports_streaming": False,
        "description": "VieNeu-TTS-0.3B - Ngọc Huyền"
    }

filtered_backbones["VieNeu-TTS-v2 (CPU)"] = {
    "repo": "pnnbao-ump/VieNeu-TTS-v2",
    "gguf_filename": "VieNeu-TTS-v2-Q4-K-M.gguf",
    "supports_streaming": False,
    "description": "VieNeu-TTS-v2 (CPU) - GGUF Q4_K_M, hỗ trợ song ngữ & podcast"
}

filtered_backbones["VieNeu-TTS-v2-Turbo (CPU)"] = {
    "repo": "pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF",
    "supports_streaming": True,
    "description": "VieNeu-TTS-v2-Turbo - Siêu nhanh, tối ưu tuyệt đối cho CPU & Thiết bị yếu"
}

BACKBONE_CONFIGS = filtered_backbones

filtered_codecs = {
    "NeuCodec (Distill)": {
        "repo": "neuphonic/distill-neucodec",
        "description": "Codec mặc định cho model GPU",
        "use_preencoded": False
    },
    "NeuCodec (ONNX)": {
        "repo": "neuphonic/neucodec-onnx-decoder-int8",
        "description": "Codec siêu nhẹ, tối ưu cho CPU (ONNX)",
        "use_preencoded": False
    },
    "VieNeu-Codec": {
        "repo": "pnnbao-ump/VieNeu-Codec",
        "description": "Codec tối ưu cho Turbo v2 (ONNX)",
        "use_preencoded": False
    }
}
CODEC_CONFIGS = filtered_codecs

_text_settings = _config.get("text_settings", {})
MAX_CHARS_PER_CHUNK = _text_settings.get("max_chars_per_chunk", 256)
MAX_TOTAL_CHARS_STREAMING = _text_settings.get("max_total_chars_streaming", 3000)

if not BACKBONE_CONFIGS or not CODEC_CONFIGS:
    raise ValueError("config.yaml thiếu backbone_configs hoặc codec_configs")

# --- 1. MODEL CONFIGURATION ---
# Global model instance
tts = None
current_backbone = None
current_codec = None
current_device_choice = None
current_force_lmdeploy = False
current_custom_model_id = ""
current_custom_base_model = ""
model_loaded = False
using_lmdeploy = False
PRESET_VOICES_CACHE = []  # List of all voices (tuples or strings)
CONV_VOICES_CACHE = []    # Filtered list for conversation (podcast=True)
MAX_SPEAKERS = 8          # Max concurrent speakers in conversation tab

# Normalizer (module-level singleton)
_text_normalizer = Normalizer()

# --- CANCELLATION ---
# threading.Event is a mutable object: never reassigned, always the same reference.
# All threads share the exact same object — no scoping/serialization issues.
_STOP_EVENT = threading.Event()

# Cache for reference texts
_ref_text_cache = {}

def get_default_model_selection() -> dict:
    if "VieNeu-TTS-v2 (GPU)" in BACKBONE_CONFIGS:
        default_backbone = "VieNeu-TTS-v2 (GPU)"
    elif "VieNeu-TTS-v2-Turbo (CPU)" in BACKBONE_CONFIGS:
        default_backbone = "VieNeu-TTS-v2-Turbo (CPU)"
    else:
        default_backbone = list(BACKBONE_CONFIGS.keys())[0]

    if "Turbo" in default_backbone:
        default_codec = "VieNeu-Codec"
    elif "(CPU)" in default_backbone:
        default_codec = "NeuCodec (ONNX)"
    else:
        default_codec = "NeuCodec (Distill)" if "NeuCodec (Distill)" in CODEC_CONFIGS else list(CODEC_CONFIGS.keys())[0]

    base_model_choices = [k for k in BACKBONE_CONFIGS.keys() if "turbo" not in k.lower() and k != "Custom Model"]
    return {
        "backbone_choice": default_backbone,
        "codec_choice": default_codec,
        "device_choice": "Auto",
        "force_lmdeploy": True,
        "custom_model_id": "",
        "custom_base_model": base_model_choices[0] if base_model_choices else "",
    }

def load_model_selection_state() -> dict:
    state = get_default_model_selection()
    if not os.path.exists(MODEL_SELECTION_STATE_PATH):
        return state

    try:
        with open(MODEL_SELECTION_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        return state

    if data.get("backbone_choice") in BACKBONE_CONFIGS or data.get("backbone_choice") == "Custom Model":
        state["backbone_choice"] = data.get("backbone_choice")
    if data.get("codec_choice") in CODEC_CONFIGS:
        state["codec_choice"] = data.get("codec_choice")
    if data.get("device_choice") in get_available_devices():
        state["device_choice"] = data.get("device_choice")
    state["force_lmdeploy"] = bool(data.get("force_lmdeploy", state["force_lmdeploy"]))
    state["custom_model_id"] = str(data.get("custom_model_id", "") or "")

    base_model_choices = [k for k in BACKBONE_CONFIGS.keys() if "turbo" not in k.lower() and k != "Custom Model"]
    if data.get("custom_base_model") in base_model_choices:
        state["custom_base_model"] = data.get("custom_base_model")
    return state

def save_model_selection_state(
    backbone_choice: str,
    codec_choice: str,
    device_choice: str,
    force_lmdeploy: bool,
    custom_model_id: str = "",
    custom_base_model: str = "",
):
    payload = {
        "backbone_choice": backbone_choice,
        "codec_choice": codec_choice,
        "device_choice": device_choice,
        "force_lmdeploy": bool(force_lmdeploy),
        "custom_model_id": (custom_model_id or "").strip(),
        "custom_base_model": custom_base_model or "",
    }
    try:
        with open(MODEL_SELECTION_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"   ⚠️ Could not save model selection state: {e}")

def is_repo_cached_locally(repo_id: str, filename: str | None = None) -> bool:
    if not repo_id:
        return False
    if os.path.exists(repo_id):
        return True

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        if filename:
            hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
        else:
            snapshot_download(repo_id=repo_id, local_files_only=True)
        return True
    except Exception:
        return False

def is_model_selection_cached(
    backbone_choice: str,
    codec_choice: str,
    custom_model_id: str = "",
) -> bool:
    if backbone_choice == "Custom Model":
        model_id = (custom_model_id or "").strip()
        if not model_id:
            return False
        return is_repo_cached_locally(model_id)

    backbone_config = BACKBONE_CONFIGS.get(backbone_choice)
    codec_config = CODEC_CONFIGS.get(codec_choice)
    if not backbone_config or not codec_config:
        return False

    backbone_cached = is_repo_cached_locally(
        backbone_config["repo"],
        filename=backbone_config.get("gguf_filename"),
    )
    codec_cached = is_repo_cached_locally(codec_config["repo"])
    return backbone_cached and codec_cached

def is_current_model_selection(
    backbone_choice: str,
    codec_choice: str,
    device_choice: str,
    force_lmdeploy: bool,
    custom_model_id: str = "",
    custom_base_model: str = "",
) -> bool:
    return (
        model_loaded
        and tts is not None
        and current_backbone == backbone_choice
        and current_codec == codec_choice
        and current_device_choice == device_choice
        and bool(current_force_lmdeploy) == bool(force_lmdeploy)
        and (current_custom_model_id or "") == ((custom_model_id or "").strip())
        and (current_custom_base_model or "") == (custom_base_model or "")
    )

def unload_loaded_model():
    global tts, current_backbone, current_codec, current_device_choice, current_force_lmdeploy
    global current_custom_model_id, current_custom_base_model, model_loaded, using_lmdeploy
    try:
        if tts is not None:
            close = getattr(tts, "close", None)
            if callable(close):
                close()
    except Exception:
        pass

    tts = None
    current_backbone = None
    current_codec = None
    current_device_choice = None
    current_force_lmdeploy = False
    current_custom_model_id = ""
    current_custom_base_model = ""
    model_loaded = False
    using_lmdeploy = False
    cleanup_gpu_memory()

def build_selection_state_response(message: str, allow_load: bool, clear_voices: bool = False):
    slot_updates = [gr.update(choices=[]) if clear_voices else gr.update() for _ in range(MAX_SPEAKERS)]
    voice_update = gr.update(choices=[], value=None, interactive=False) if clear_voices else gr.update()
    return (
        message,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=allow_load),
        gr.update(interactive=False),
        voice_update,
        gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(),
        *slot_updates
    )

def maybe_auto_load_selected_model(
    backbone_choice: str,
    codec_choice: str,
    device_choice: str,
    force_lmdeploy: bool,
    custom_model_id: str = "",
    custom_base_model: str = "",
    custom_hf_token: str = "",
):
    is_cached = is_model_selection_cached(backbone_choice, codec_choice, custom_model_id)

    if is_cached:
        if is_current_model_selection(
            backbone_choice,
            codec_choice,
            device_choice,
            force_lmdeploy,
            custom_model_id,
            custom_base_model,
        ):
            msg = get_model_status_message() + "\n\n📦 Model đã có sẵn trong máy và đã tự nạp vào RAM."
            yield (
                msg,
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(),
                *([gr.update()] * MAX_SPEAKERS)
            )
            return

        for item in load_model(
            backbone_choice,
            codec_choice,
            device_choice,
            force_lmdeploy,
            custom_model_id,
            custom_base_model,
            custom_hf_token,
        ):
            msg = item[0] if item else ""
            updated = list(item)
            updated[3] = gr.update(interactive=False if not str(msg).startswith("❌") else True)
            yield tuple(updated)
        return

    if not is_current_model_selection(
        backbone_choice,
        codec_choice,
        device_choice,
        force_lmdeploy,
        custom_model_id,
        custom_base_model,
    ):
        unload_loaded_model()

    yield build_selection_state_response(
        "📥 Model này chưa có sẵn trong máy. Bạn có thể bấm `Tải model` để tải và nạp model này.",
        allow_load=True,
        clear_voices=True,
    )

def get_available_devices() -> list[str]:
    """Get list of available devices for current platform."""
    devices = ["Auto", "CPU"]
    
    try:
        import torch
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            devices.append("MPS")
        elif torch.cuda.is_available():
            devices.append("CUDA")
    except ImportError:
        pass

    return devices

def get_model_status_message() -> str:
    """Reconstruct status message from global state"""
    global model_loaded, tts, using_lmdeploy, current_backbone, current_codec
    if not model_loaded or tts is None:
        return "⏳ Chưa tải model."
    
    if "v2-Turbo" in (current_backbone or ""):
        backend_name = "⚡ Turbo (v2)"
    elif using_lmdeploy:
        backend_name = "🚀 LMDeploy (Optimized)"
    else:
        backend_name = "📦 Standard"
    
    # We don't track the exact device strings perfectly in global state, so we estimate
    try:
        import torch
        has_mps = torch.backends.mps.is_available()
        has_cuda = torch.cuda.is_available()
    except:
        has_mps = has_cuda = False

    device_info = "GPU (CUDA)" if (using_lmdeploy or "CUDA" in (current_backbone or "")) else ("MPS (Metal)" if has_mps else "Auto")
    
    if "v2-Turbo" in (current_backbone or ""):
        codec_device = "GPU/MPS" if (has_cuda or has_mps) else "CPU"
    elif "ONNX" in (current_codec or ""):
        codec_device = "CPU"
    else:
        codec_device = "GPU/MPS" if (has_cuda or has_mps) else "CPU"

    preencoded_note = ""    
    opt_info = ""
    if using_lmdeploy and hasattr(tts, 'get_optimization_stats'):
        stats = tts.get_optimization_stats()
        opt_info = (
            f"\n\n🔧 Tối ưu hóa:"
            f"\n  • Triton: {'✅' if stats['triton_enabled'] else '❌'}"
            f"\n  • Max Batch Size (Default): {stats.get('max_batch_size', 'N/A')}"
            f"\n  • Reference Cache: {stats['cached_references']} voices"
            f"\n  • Prefix Caching: ❌"
        )

    return (
        f"✅ Model đã tải thành công!\n\n"
        f"🔧 Backend: {backend_name}\n"
        f" Parrot: {current_backbone} on {device_info}\n"
        f"🎵 Codec: {current_codec} on {codec_device}{preencoded_note}{opt_info}"
    )

def restore_ui_state():
    """Update UI components based on persistence"""
    global model_loaded
    msg = get_model_status_message()
    return (
        msg, 
        gr.update(interactive=model_loaded), # btn_generate
        gr.update(interactive=model_loaded), # btn_generate_conv
        gr.update(interactive=False)         # btn_stop
    )

def should_use_lmdeploy(backbone_choice: str, device_choice: str) -> bool:
    """Determine if we should use LMDeploy backend."""
    # LMDeploy not supported on macOS
    if sys.platform == "darwin":
        return False

    if "gguf" in backbone_choice.lower() or "v2-turbo" in backbone_choice.lower():
        return False
    
    try:
        import torch
        if device_choice == "Auto":
            has_gpu = torch.cuda.is_available()
        elif device_choice == "CUDA":
            has_gpu = torch.cuda.is_available()
        else:
            has_gpu = False
        return has_gpu
    except ImportError:
        return False

@lru_cache(maxsize=32)
def get_ref_text_cached(text_path: str) -> str:
    """Cache reference text loading"""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()

def cleanup_gpu_memory():
    """Aggressively cleanup GPU memory"""
    if 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    gc.collect()

def _voice_codes_to_list(codes):
    if codes is None:
        return []
    if isinstance(codes, list):
        return codes
    if isinstance(codes, np.ndarray):
        return codes.flatten().tolist()
    try:
        import torch
        if isinstance(codes, torch.Tensor):
            return codes.detach().cpu().flatten().tolist()
    except Exception:
        pass
    if hasattr(codes, "tolist"):
        data = codes.tolist()
        return data if isinstance(data, list) else [data]
    return list(codes)

def _safe_voice_id(name: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_-]+", "_", name.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or f"Voice_{uuid.uuid4().hex[:8]}"

def _default_user_voices_payload(default_voice_id: str | None = None) -> dict:
    return {
        "meta": {
            "spec": "vieneu.voice.presets",
            "spec_version": "1.0",
            "source": "local_user_saved",
        },
        "default_voice": default_voice_id or "",
        "presets": {},
    }

def load_user_voices_data(default_voice_id: str | None = None) -> dict:
    base = _default_user_voices_payload(default_voice_id)
    if not os.path.exists(USER_VOICES_PATH):
        return base

    try:
        if os.path.getsize(USER_VOICES_PATH) == 0:
            return base
        with open(USER_VOICES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except json.JSONDecodeError:
        backup_path = f"{USER_VOICES_PATH}.corrupt.{int(time.time())}.bak"
        try:
            os.replace(USER_VOICES_PATH, backup_path)
            print(f"   ⚠️ user_voices.json invalid, backed up to: {backup_path}")
        except Exception:
            pass
        return base
    except Exception:
        return base

    if not isinstance(data, dict):
        return base

    data.setdefault("meta", base["meta"])
    data.setdefault("default_voice", default_voice_id or data.get("default_voice", ""))
    data.setdefault("presets", {})
    if not isinstance(data["presets"], dict):
        data["presets"] = {}
    return data

def load_user_voices_into_tts():
    if tts is None or not os.path.exists(USER_VOICES_PATH):
        return
    try:
        data = load_user_voices_data()
        presets = data.get("presets", {})
        if presets:
            tts._preset_voices.update(presets)
            print(f"   ✅ Loaded {len(presets)} user voices")
    except Exception as e:
        print(f"   ⚠️ Could not load user voices: {e}")

def save_current_clone_voice(voice_name: str, voice_description: str, custom_audio, custom_text: str):
    global PRESET_VOICES_CACHE
    if not model_loaded or tts is None:
        return gr.update(), "⚠️ Vui lòng tải model trước khi lưu voice."
    if custom_audio is None:
        return gr.update(), "⚠️ Vui lòng upload audio giọng mẫu trước."
    if not custom_text or not custom_text.strip():
        return gr.update(), "⚠️ Vui lòng nhập transcript đúng của audio mẫu trước khi lưu."
    if not voice_name or not voice_name.strip():
        return gr.update(), "⚠️ Vui lòng nhập tên voice cần lưu."

    try:
        voice_id = _safe_voice_id(voice_name)
        existing = set(getattr(tts, "_preset_voices", {}).keys())
        base_voice_id = voice_id
        idx = 2
        while voice_id in existing:
            voice_id = f"{base_voice_id}_{idx}"
            idx += 1

        ref_codes = tts.encode_reference(custom_audio)
        voice_entry = {
            "description": voice_description.strip() or voice_name.strip(),
            "text": custom_text.strip(),
            "codes": _voice_codes_to_list(ref_codes),
            "source": "user_saved",
            "podcast": True,
        }

        data = load_user_voices_data(default_voice_id=voice_id)

        data.setdefault("presets", {})[voice_id] = voice_entry
        if not data.get("default_voice"):
            data["default_voice"] = voice_id
        with open(USER_VOICES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        tts._preset_voices[voice_id] = voice_entry
        voices = tts.list_preset_voices()
        PRESET_VOICES_CACHE = voices
        return gr.update(choices=voices, value=voice_id, interactive=True), f"✅ Đã lưu voice '{voice_id}' vào hệ thống."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return gr.update(), f"❌ Lỗi lưu voice: {e}"

def apply_pronunciation_replacements(text: str, replacement_rules: str) -> str:
    if not replacement_rules or not replacement_rules.strip():
        return text

    result = text
    for raw_line in replacement_rules.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" in line:
            source, target = line.split("=>", 1)
        elif "=" in line:
            source, target = line.split("=", 1)
        else:
            continue

        source = source.strip()
        target = target.strip()
        if not source:
            continue
        result = re.sub(re.escape(source), target, result, flags=re.IGNORECASE)
    return result

def load_pronunciation_rules() -> str:
    if not os.path.exists(PRONUNCIATION_RULES_PATH):
        return DEFAULT_PRONUNCIATION_RULES
    try:
        with open(PRONUNCIATION_RULES_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content or DEFAULT_PRONUNCIATION_RULES
    except Exception:
        return DEFAULT_PRONUNCIATION_RULES

def save_pronunciation_rules(rules_text: str):
    value = (rules_text or "").strip() or DEFAULT_PRONUNCIATION_RULES
    try:
        with open(PRONUNCIATION_RULES_PATH, "w", encoding="utf-8") as f:
            f.write(value + "\n")
        return "Da luu tu dien phat am."
    except Exception as e:
        return f"Khong luu duoc tu dien phat am: {e}"

def reset_pronunciation_rules():
    return DEFAULT_PRONUNCIATION_RULES, "Da khoi phuc tu dien mac dinh."

def prepare_short_text_for_tts(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return value

    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r" *\n+ *", "\n", value)
    plain_value = value.replace("\n", " ").strip()
    if len(plain_value) <= 48 and plain_value[-1] not in ".!?…":
        value = value + "."
    return value

def normalize_leading_fillers_for_tts(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return value

    # "Ừm" at the beginning is often treated as a hesitation and skipped.
    # "Ờm" phonemizes more reliably while preserving the intended filler sound.
    value = re.sub(r"^\s*[\u1eea\u1eeb]m\s*[,，.。…]*\s*", "Ờm, ", value, count=1, flags=re.IGNORECASE)
    value = re.sub(r"^\s*um+\s*[,，.。…]*\s*", "Ờm, ", value, count=1, flags=re.IGNORECASE)
    value = re.sub(r"^\s*uhm+\s*[,，.。…]*\s*", "Ờm, ", value, count=1, flags=re.IGNORECASE)
    value = re.sub(r"^\s*h\u1ea3\s*[,，!！?？.。…]*\s*", "H\u1ea3?\n", value, count=1, flags=re.IGNORECASE)
    return value

def safe_normalize_text_for_tts(text: str) -> str:
    source = re.sub(r"[ \t\r\f\v]+", " ", str(text or "").strip())
    source = re.sub(r" *\n+ *", "\n", source)
    if not source:
        return source

    try:
        normalized = re.sub(r"[ \t\r\f\v]+", " ", _text_normalizer.normalize(source).strip())
        normalized = re.sub(r" *\n+ *", "\n", normalized)
    except Exception:
        return source

    if not normalized:
        return source

    source_has_vietnamese = bool(re.search(r"[ăâđêôơưĂÂĐÊÔƠƯàáạảãèéẹẻẽìíịỉĩòóọỏõùúụủũỳýỵỷỹÀÁẠẢÃÈÉẸẺẼÌÍỊỈĨÒÓỌỎÕÙÚỤỦŨỲÝỴỶỸ]", source))
    suspicious = (
        normalized.startswith("?")
        or normalized.count("?") >= 1
        or len(normalized) < max(4, int(len(source) * 0.6))
    )

    if source_has_vietnamese and suspicious:
        return source

    return normalized

def estimate_chunk_max_tokens(chunk_text: str) -> int:
    length = len(str(chunk_text).strip())
    if length <= 12:
        return 112
    if length <= 24:
        return 144
    if length <= 48:
        return 176
    if length <= 80:
        return 240
    if length <= 140:
        return 320
    if length <= 220:
        return 448
    return min(896, max(384, int(length * 4.0)))

def estimate_chunk_generation_budget(
    tts_obj,
    chunk_text: str,
    ref_codes,
    ref_text_raw: str,
) -> int:
    text_value = str(chunk_text or "").strip()
    if not text_value:
        return 128

    fallback_budget = estimate_chunk_max_tokens(text_value)

    try:
        chunk_phonemes = phonemize_with_dict(text_value, skip_normalize=True)
        phoneme_units = max(1, len(chunk_phonemes.split()))
        phoneme_chars = max(1, len(chunk_phonemes.replace(" ", "")))
        text_chars = max(1, len(text_value))
        speech_budget = max(
            160,
            int(phoneme_units * 8.0 + 64),
            int(phoneme_chars * 1.9 + 48),
            int(text_chars * 4.8),
        )

        max_context = int(getattr(tts_obj, "max_context", 2048))
        context_headroom = None

        if getattr(tts_obj, "_is_quantized_model", False) and hasattr(tts_obj, "backbone") and hasattr(tts_obj.backbone, "tokenize"):
            ref_phonemes = tts_obj.get_ref_phonemes(ref_text_raw) if ref_text_raw else ""
            prompt = tts_obj._format_prompt(
                ref_codes,
                ref_text_raw or "",
                text_value,
                ref_phonemes=ref_phonemes,
                input_phonemes=chunk_phonemes,
                use_chat_format=getattr(tts_obj, "use_chat_format", False),
                emotion_tag=getattr(tts_obj, "default_emotion", None),
            )
            prompt_tokens = len(tts_obj.backbone.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))
            context_headroom = max(128, max_context - prompt_tokens - 32)
        elif getattr(tts_obj, "tokenizer", None) is not None and hasattr(tts_obj, "_apply_chat_template"):
            ref_phonemes = tts_obj.get_ref_phonemes(ref_text_raw) if ref_text_raw else ""
            prompt_ids = tts_obj._apply_chat_template(
                ref_codes,
                ref_phonemes,
                chunk_phonemes,
                emotion_tag=getattr(tts_obj, "default_emotion", None),
            )
            context_headroom = max(128, max_context - len(prompt_ids) - 16)

        if context_headroom is not None:
            speech_budget = min(speech_budget, context_headroom)

        return int(max(160, min(1024, speech_budget)))
    except Exception:
        return fallback_budget

def adjust_temperature_for_short_text(chunk_text: str, temperature: float) -> float:
    length = len(str(chunk_text).strip())
    if length <= 24:
        return min(float(temperature), 0.45)
    if length <= 48:
        return min(float(temperature), 0.55)
    return float(temperature)

def append_output_tail_silence(audio: np.ndarray, sr: int, duration_s: float = 0.12) -> np.ndarray:
    if audio is None or len(audio) == 0:
        return audio
    pad_samples = max(1, int(sr * duration_s))
    tail = np.zeros(pad_samples, dtype=np.float32)
    return np.concatenate([np.asarray(audio, dtype=np.float32), tail])

def split_chunk_for_retry(chunk_text: str, max_chars: int) -> list[str]:
    text = re.sub(r"\s+", " ", str(chunk_text or "").strip())
    if not text:
        return []

    smaller_max_chars = max(24, min(max_chars, max(48, len(text) // 2)))
    pieces = split_text_into_chunks(text, max_chars=smaller_max_chars)
    if len(pieces) > 1:
        return pieces

    minor_parts = [part.strip() for part in re.split(r"(?<=[,;:])\s+", text) if part.strip()]
    if len(minor_parts) > 1:
        return minor_parts

    words = text.split()
    if len(words) > 2:
        mid = max(1, len(words) // 2)
        return [" ".join(words[:mid]), " ".join(words[mid:])]
    return [text]

def analyze_tts_text_length(text: str) -> dict:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    words = re.findall(r"\w+", value, flags=re.UNICODE)
    sentence_count = len([p for p in re.split(r"[.!?]+", value) if p.strip()])
    return {
        "text": value,
        "chars": len(value),
        "words": len(words),
        "sentences": max(1, sentence_count),
    }

def count_text_words(text: str) -> int:
    return len(re.findall(r"\w+", str(text or ""), flags=re.UNICODE))

def get_effective_chunk_chars(max_chars_chunk: int, is_v2_turbo: bool, full_text: str = "") -> int:
    requested = int(max_chars_chunk or MAX_CHARS_PER_CHUNK)
    if is_v2_turbo:
        return requested

    info = analyze_tts_text_length(full_text)
    chars = info["chars"]
    words = info["words"]

    if chars <= 0:
        return max(48, min(requested, 90))

    if chars <= 90 and words <= 18:
        return max(48, min(requested, 110))

    if chars <= 180:
        target_words_per_chunk = 14
        hard_cap = 90
    elif chars <= 360:
        target_words_per_chunk = 14
        hard_cap = 82
    else:
        target_words_per_chunk = 12
        hard_cap = 72

    desired_chunks = max(1, int(np.ceil(max(1, words) / target_words_per_chunk)))
    chars_by_length = int(np.ceil(chars / desired_chunks)) + 8
    return max(56, min(requested, hard_cap, chars_by_length))

def rebalance_text_chunks(chunks: list[str], max_chars: int) -> list[str]:
    clean_chunks = [re.sub(r"\s+", " ", str(chunk or "").strip()) for chunk in chunks]
    clean_chunks = [chunk for chunk in clean_chunks if chunk]
    if len(clean_chunks) <= 1:
        return clean_chunks

    soft_cap = max(max_chars + 18, int(max_chars * 1.35))
    dangling_words = {
        "và", "với", "của", "cho", "từ", "đến", "trên", "dưới",
        "trong", "ngoài", "cả", "mà", "nhưng", "nên", "rồi"
    }

    merged: list[str] = []
    idx = 0
    while idx < len(clean_chunks):
        chunk = clean_chunks[idx]
        words = chunk.split()
        is_short = len(chunk) < 34 or count_text_words(chunk) <= 4
        has_dangling_edge = bool(words) and (
            words[-1].strip(",;:.!?").lower() in dangling_words
            or words[0].strip(",;:.!?").lower() in dangling_words
        )

        if (is_short or has_dangling_edge) and idx + 1 < len(clean_chunks):
            combined = f"{chunk} {clean_chunks[idx + 1]}".strip()
            if len(combined) <= soft_cap:
                merged.append(combined)
                idx += 2
                continue

        if (is_short or has_dangling_edge) and merged:
            combined = f"{merged[-1]} {chunk}".strip()
            if len(combined) <= soft_cap:
                merged[-1] = combined
                idx += 1
                continue

        merged.append(chunk)
        idx += 1

    if len(merged) == len(clean_chunks):
        return merged
    return rebalance_text_chunks(merged, max_chars)

def split_text_for_standard_tts(text: str, max_chars: int) -> list[str]:
    chunks = split_text_into_chunks(text, max_chars=max_chars)
    return rebalance_text_chunks(chunks, max_chars)

def estimate_min_audio_seconds_for_text(text: str) -> float:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if not value:
        return 0.0

    words = re.findall(r"\w+", value, flags=re.UNICODE)
    word_count = len(words)
    char_count = len(value)
    return min(18.0, max(0.65, word_count * 0.24, char_count * 0.045))

def is_probably_truncated_audio(wav, text: str, sr: int = 24000) -> bool:
    if wav is None or len(wav) == 0:
        return True

    value = re.sub(r"\s+", " ", str(text or "").strip())
    if len(value) < 24:
        return False

    duration = float(len(wav)) / float(sr)
    return duration < estimate_min_audio_seconds_for_text(value)

def synthesize_chunk_with_retry(
    tts_obj,
    chunk_text: str,
    ref_codes,
    ref_text_raw: str,
    temperature: float,
    max_chars_chunk: int,
    is_v2_turbo: bool,
):
    chunk_value = str(chunk_text or "").strip()
    if not chunk_value:
        return []

    base_temp = adjust_temperature_for_short_text(chunk_value, temperature)
    token_budget = estimate_chunk_generation_budget(tts_obj, chunk_value, ref_codes, ref_text_raw)
    attempts: list[dict] = []
    if is_v2_turbo:
        attempts = [
            {
                "text": chunk_value,
                "kwargs": {
                    "ref_codes": ref_codes,
                    "temperature": base_temp,
                    "max_chars": max_chars_chunk,
                    "skip_normalize": True,
                    "skip_phonemize": True,
                },
            },
            {
                "text": chunk_value,
                "kwargs": {
                    "ref_codes": ref_codes,
                    "temperature": min(base_temp, 0.35),
                    "max_chars": max_chars_chunk,
                    "skip_normalize": True,
                    "skip_phonemize": True,
                },
            },
        ]
    else:
        attempts = [
            {
                "text": chunk_value,
                "kwargs": {
                    "ref_codes": ref_codes,
                    "ref_text": ref_text_raw,
                    "temperature": base_temp,
                    "max_chars": max_chars_chunk,
                    "max_tokens": token_budget,
                    "skip_normalize": True,
                },
            },
            {
                "text": chunk_value,
                "kwargs": {
                    "ref_codes": ref_codes,
                    "ref_text": ref_text_raw,
                    "temperature": min(base_temp, 0.35),
                    "max_chars": max_chars_chunk,
                    "max_tokens": min(1024, int(token_budget * 1.3)),
                    "skip_normalize": True,
                },
            },
        ]

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            wav = tts_obj.infer(attempt["text"], **attempt["kwargs"])
            if wav is not None and len(wav) > 0 and not is_probably_truncated_audio(wav, attempt["text"]):
                return [wav]
        except Exception as exc:
            last_error = exc

    retry_parts = split_chunk_for_retry(chunk_value, max_chars_chunk)
    if len(retry_parts) > 1:
        wavs = []
        for part in retry_parts:
            sub_wavs = synthesize_chunk_with_retry(
                tts_obj,
                part,
                ref_codes,
                ref_text_raw,
                min(base_temp, 0.4),
                max(32, min(max_chars_chunk, len(part) + 16)),
                is_v2_turbo,
            )
            wavs.extend(sub_wavs)
        if wavs:
            return wavs

    if last_error is not None:
        raise last_error
    raise ValueError(f"Khong tong hop duoc doan text: {chunk_value}")

def apply_audio_style_controls(audio: np.ndarray, sr: int, speaking_rate: float, pitch_steps: float) -> np.ndarray:
    if audio is None or len(audio) == 0:
        return audio

    processed = np.asarray(audio, dtype=np.float32)
    if processed.ndim > 1:
        processed = processed.squeeze()

    safe_rate = float(np.clip(float(speaking_rate), 0.85, 1.2))
    safe_pitch = float(np.clip(float(pitch_steps), -2.0, 2.0))

    # torchaudio produces cleaner results here than librosa on these short mono clips.
    try:
        import torch
        import torchaudio

        waveform = torch.from_numpy(processed).to(torch.float32).unsqueeze(0)

        if abs(safe_rate - 1.0) > 1e-6:
            waveform, _ = torchaudio.functional.speed(waveform, orig_freq=sr, factor=safe_rate)

        if abs(safe_pitch) > 1e-6:
            waveform = torchaudio.functional.pitch_shift(
                waveform,
                sample_rate=sr,
                n_steps=int(round(safe_pitch)),
                n_fft=1024,
                hop_length=256,
            )

        processed = waveform.squeeze(0).detach().cpu().numpy()
    except Exception:
        # Fall back to the original audio instead of returning a distorted result.
        processed = np.asarray(audio, dtype=np.float32)

    peak = float(np.max(np.abs(processed))) if len(processed) else 0.0
    if peak > 1.0:
        processed = processed / peak

    return np.asarray(processed, dtype=np.float32)

def load_model(backbone_choice: str, codec_choice: str, device_choice: str, 
               force_lmdeploy: bool, custom_model_id: str = "", custom_base_model: str = "", 
               custom_hf_token: str = ""):
    """Load model with optimizations and max batch size control"""
    global tts, current_backbone, current_codec, current_device_choice, current_force_lmdeploy
    global current_custom_model_id, current_custom_base_model, model_loaded, using_lmdeploy
    lmdeploy_error_reason = None
    model_loaded = False # Ensure we don't try to use a half-loaded model
    
    # Helper for slot updates (initially no change)
    slot_no_updates = [gr.update()] * MAX_SPEAKERS

    yield (
        "⏳ Đang tải model với tối ưu hóa... Lưu ý: Quá trình này sẽ tốn thời gian. Vui lòng kiên nhẫn.",
        gr.update(interactive=False), # btn_generate
        gr.update(interactive=False), # btn_generate_conv
        gr.update(interactive=False), # btn_load
        gr.update(interactive=False), # btn_stop
        gr.update(), # voice_select
        gr.update(), gr.update(), gr.update(), gr.update(), # tab_p, tab_c, tab_sel, mode_state
        gr.update(), # conv_tab
        *slot_no_updates
    )
    
    try:
        # Cleanup before loading new model
        if tts is not None:
            tts = None # Reset instead of del to avoid NameError if load fails
            cleanup_gpu_memory()
        
        # Prepare Backbone Config/Repo
        custom_loading = False
        is_merged_lora = False

        if backbone_choice == "Custom Model":
            custom_loading = True
            if not custom_model_id or not custom_model_id.strip():
                yield (
                    "❌ Lỗi: Vui lòng nhập Model ID cho Custom Model.",
                    gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), # conv_tab
                    *slot_no_updates
                )
                return

            # Check if it is a LoRA to merge
            if "lora" in custom_model_id.lower():
                # Merging mode
                print(f"🔄 Detected LoRA in name. preparing merge with base: {custom_base_model}")
                if custom_base_model not in BACKBONE_CONFIGS:
                    yield (
                        f"❌ Lỗi: Base Model '{custom_base_model}' không hợp lệ.",
                        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), # conv_tab
                        *slot_no_updates
                    )
                    return
                
                base_config = BACKBONE_CONFIGS[custom_base_model]
                backbone_config = {
                    "repo": base_config["repo"], # Load base first
                    "supports_streaming": base_config["supports_streaming"],
                    "description": f"Custom Merged: {custom_model_id} + {custom_base_model}"
                }
                is_merged_lora = True
            else:
                # Normal custom model
                backbone_config = {
                    "repo": custom_model_id.strip(),
                    "supports_streaming": False, # Assume false for unknown
                    "description": f"Custom Model: {custom_model_id}"
                }
        else:
            backbone_config = BACKBONE_CONFIGS[backbone_choice]
            
        codec_config = CODEC_CONFIGS[codec_choice]
        use_lmdeploy = False
        
        # Override LMDeploy if custom
        if custom_loading:
             if "gguf" in backbone_config['repo'].lower() or "v2-turbo" in backbone_config['repo'].lower():
                 # GGUF must use Standard/Turbo backend
                 use_lmdeploy = False
             elif is_merged_lora:
                 # LoRA can use LMDeploy if we merge first (checked logic below) or Standard
                 use_lmdeploy = force_lmdeploy and should_use_lmdeploy(custom_base_model, device_choice)
             else:
                 # Full custom model (e.g. finetune)
                 use_lmdeploy = force_lmdeploy and should_use_lmdeploy("VieNeu-TTS (GPU)", device_choice) # Assume GPU compatible?
        # Use LMDeploy only if Force LMDeploy is set and the model is compatible
        # NOTE: For VieNeu-v2-Turbo, we handle LMDeploy inside TurboGPUVieNeuTTS class, 
        # so we set use_lmdeploy = False here to avoid generic FastVieNeuTTS loading.
        # NOTE: For custom_loading, the block above already decided use_lmdeploy correctly
        # (e.g. False for GGUF repos). Do NOT override that decision here.
        if "v2-Turbo" in backbone_choice:
             should_use_generic_fast = False
        elif custom_loading:
             should_use_generic_fast = False  # already handled above per repo name
        else:
             should_use_generic_fast = force_lmdeploy and should_use_lmdeploy(backbone_choice, device_choice)
             
        if should_use_generic_fast:
            use_lmdeploy = True
        
        if use_lmdeploy:
            lmdeploy_error_reason = None
            print(f"🚀 Using LMDeploy backend with optimizations")
            
            backbone_device = "cuda"
            
            if "ONNX" in codec_choice:
                codec_device = "cpu"
            else:
                try:
                    import torch
                    codec_device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    codec_device = "cpu"
            
            # Special handling for Custom LoRA + LMDeploy -> Merge & Save
            target_backbone_repo = backbone_config["repo"]
            
            if custom_loading and is_merged_lora:
                safe_name = custom_model_id.strip().replace("/", "_").replace("\\", "_").replace(":", "")
                cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "merged_models_cache", safe_name)
                target_backbone_repo = os.path.abspath(cache_dir)
                
                # Check if already merged (and voices.json exists)
                if not os.path.exists(cache_dir) or not os.path.exists(os.path.join(cache_dir, "vocab.json")):
                    print(f"🔄 Merging LoRA for LMDeploy optimization: {cache_dir}")
                    if os.path.exists(cache_dir):
                        print("   ⚠️ Detected incomplete cache, rebuilding...")
                    yield (
                         f"⏳ Đang merge và lưu model LoRA để tối ưu cho LMDeploy (thao tác này chỉ chạy một lần)...",
                         gr.update(interactive=False),
                         gr.update(interactive=False),
                         gr.update(interactive=False),
                         gr.update(interactive=False),
                         gr.update(),
                         gr.update(), gr.update(), gr.update(), gr.update(),
                         gr.update(), # conv_tab
                         *slot_no_updates
                    )
                    
                    try:
                        # Use GPU for merging if available for speed
                        # We use the Base Model specified
                        from vieneu.standard import VieNeuTTS
                        base_repo = BACKBONE_CONFIGS[custom_base_model]["repo"]
                        merge_device = "cuda" if torch.cuda.is_available() else "cpu"
                        
                        print(f"   • Loading base: {base_repo} ({merge_device})")
                        temp_tts = VieNeuTTS(
                            backbone_repo=base_repo,
                            backbone_device=merge_device, 
                            codec_repo=codec_config["repo"],
                            codec_device="cpu", # Codec unused for merging, keep on CPU
                            hf_token=custom_hf_token
                        )
                        
                        print(f"   • Loading Adapter: {custom_model_id}")
                        temp_tts.load_lora_adapter(custom_model_id.strip(), hf_token=custom_hf_token)
                        
                        print(f"   • Merging...")
                        if hasattr(temp_tts.backbone, "merge_and_unload"):
                            temp_tts.backbone = temp_tts.backbone.merge_and_unload()
                        
                        print(f"   • Saving to cache: {cache_dir}")
                        temp_tts.backbone.save_pretrained(cache_dir)
                        temp_tts.tokenizer.save_pretrained(cache_dir)
                        
                        # Fix for LMDeploy: Explicitly save legacy tokenizer files (vocab.json, merges.txt)
                        # because LMDeploy/Transformers might default to slow tokenizer if fast one has issues,
                        # and save_pretrained on fast tokenizer sometimes omits legacy files.
                        try:
                            print("   • Ensuring legacy tokenizer files...")
                            from transformers import AutoTokenizer
                            slow_tokenizer = AutoTokenizer.from_pretrained(base_repo, use_fast=False)
                            slow_tokenizer.save_pretrained(cache_dir)
                        except Exception as e:
                            print(f"   ⚠️ Warning: Could not save slow tokenizer files: {e}")

                        # Save voices.json to cache directory so FastVieNeuTTS can find it
                        print(f"   • Saving voices definition...")
                        import json
                        voices_json_path = os.path.join(cache_dir, "voices.json")
                        voices_content = {
                             "meta": { "note": "Automatically generated during LoRA merge" },
                             "default_voice": temp_tts._default_voice,
                             "presets": temp_tts._preset_voices
                        }
                        with open(voices_json_path, 'w', encoding='utf-8') as f:
                             json.dump(voices_content, f, ensure_ascii=False, indent=2)

                        del temp_tts
                        cleanup_gpu_memory()
                        print("   ✅ Merge & Save successfully!")
                        
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise RuntimeError(f"Failed to merge & save LoRA for LMDeploy: {e}")

            print(f"📦 Loading optimized model...")
            print(f"   Backbone: {target_backbone_repo} on {backbone_device}")
            print(f"   Codec: {codec_config['repo']} on {codec_device}")
            print(f"   Triton: Enabled")
            
            try:
                from vieneu.fast import FastVieNeuTTS
                tts = FastVieNeuTTS(
                    backbone_repo=target_backbone_repo,
                    backbone_device=backbone_device,
                    codec_repo=codec_config["repo"],
                    codec_device=codec_device,
                    memory_util=0.3,
                    tp=1,
                    enable_prefix_caching=False,
                    enable_triton=True,
                    hf_token=custom_hf_token
                )
                using_lmdeploy = True
                
                # Legacy caching removed
                print(f"   ✅ Optimized backend initialized")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                
                error_str = str(e)
                if "$env:CUDA_PATH" in error_str:
                    lmdeploy_error_reason = "Không tìm thấy biến môi trường CUDA_PATH. Vui lòng cài đặt NVIDIA GPU Computing Toolkit."
                else:
                    lmdeploy_error_reason = f"{error_str}"
                
                yield (
                    f"⚠️ LMDeploy Init Error: {lmdeploy_error_reason}. Đang loading model với backend mặc định - tốc độ chậm hơn so với lmdeploy...",
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), # conv_tab
                    *slot_no_updates
                )
                time.sleep(1)
                use_lmdeploy = False
                using_lmdeploy = False
        
        if not use_lmdeploy:
            print(f"📦 Using original backend")

            if device_choice == "Auto":
                repo_lower = backbone_config['repo'].lower()
                is_gguf_backbone = "gguf" in repo_lower

                if is_gguf_backbone:
                    # GGUF backbones (llama-cpp-python): Metal on Mac, CUDA on Windows/Linux
                    if sys.platform == "darwin":
                        backbone_device = "gpu"  # llama-cpp-python uses Metal via n_gpu_layers
                    else:
                        try:
                            import torch
                            backbone_device = "gpu" if torch.cuda.is_available() else "cpu"
                        except ImportError:
                            backbone_device = "cpu"
                else:
                    # PyTorch backbones (Standard, Turbo GPU): use native torch device
                    try:
                        import torch
                        if sys.platform == "darwin":
                            backbone_device = "mps" if torch.backends.mps.is_available() else "cpu"
                        else:
                            backbone_device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        backbone_device = "cpu"

                # Codec device
                if "ONNX" in codec_choice:
                    codec_device = "cpu"
                else:
                    try:
                        import torch
                        if sys.platform == "darwin":
                            codec_device = "mps" if torch.backends.mps.is_available() else "cpu"
                        else:
                            codec_device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        codec_device = "cpu"

            elif device_choice == "MPS":
                backbone_device = "mps"
                codec_device = "mps" if "ONNX" not in codec_choice else "cpu"

            else:
                backbone_device = device_choice.lower()
                codec_device = device_choice.lower()

                if "ONNX" in codec_choice:
                    codec_device = "cpu"

            if "gguf" in backbone_config['repo'].lower() and backbone_device == "cuda":
                # Only Llama-cpp (GGUF) uses the 'gpu' string for CUDA
                backbone_device = "gpu"
            
            print(f"📦 Loading model...")
            print(f"   Backbone: {backbone_config['repo']} on {backbone_device}")
            print(f"   Codec: {codec_config['repo']} on {codec_device}")
            
            if "v2-Turbo" in backbone_choice:
                # VieNeu v2 Turbo uses the dedicated backend
                print("   ⚡ Mode: Turbo")
                mode = "turbo_gpu" if "GPU" in backbone_choice else "turbo"
                tts = Vieneu(
                    mode=mode,
                    backbone_repo=backbone_config["repo"],
                    decoder_repo=codec_config["repo"],
                    device=backbone_device,
                    backend="lmdeploy" if force_lmdeploy and "GPU" in backbone_choice else "standard",
                    hf_token=custom_hf_token
                )
            else:
                from vieneu.standard import VieNeuTTS
                tts = VieNeuTTS(
                    backbone_repo=backbone_config["repo"],
                    backbone_device=backbone_device,
                    codec_repo=codec_config["repo"],
                    codec_device=codec_device,
                    hf_token=custom_hf_token,
                    gguf_filename=backbone_config.get("gguf_filename")
                )

            # Perform LoRA Merge if needed (ONLY for Standard Backend)
            # For LMDeploy, we handled it above by saving to disk
            if is_merged_lora and custom_loading and not using_lmdeploy:
                yield (
                    f"🔄 Đang tải và merge LoRA adapter: {custom_model_id}...",
                    gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), # conv_tab
                    *slot_no_updates
                )
                try:
                    # 1. Load Adapter
                    tts.load_lora_adapter(custom_model_id.strip(), hf_token=custom_hf_token)
                    
                    # 2. Merge and Unload
                    # Check if backbone matches expected type for merge
                    if hasattr(tts, 'backbone') and hasattr(tts.backbone, 'merge_and_unload'):
                        print("   🔄 Merging LoRA into backbone...")
                        tts.backbone = tts.backbone.merge_and_unload()
                        
                        # Reset LoRA state so it behaves like a normal model
                        tts._lora_loaded = False 
                        tts._current_lora_repo = None
                        print("   ✅ Merged successfully!")
                    else:
                        print("   ⚠️ Warning: Model does not support merge_and_unload, keeping adapter active.")
                        
                except Exception as e:
                     raise RuntimeError(f"Failed to merge LoRA: {e}")

            using_lmdeploy = False
        
        current_backbone = backbone_choice
        current_codec = codec_choice
        current_device_choice = device_choice
        current_force_lmdeploy = bool(force_lmdeploy)
        current_custom_model_id = custom_model_id.strip() if backbone_choice == "Custom Model" else ""
        current_custom_base_model = custom_base_model or ""
        model_loaded = True
        save_model_selection_state(
            backbone_choice,
            codec_choice,
            device_choice,
            force_lmdeploy,
            current_custom_model_id,
            current_custom_base_model,
        )
        load_user_voices_into_tts()
        
        # Success message with optimization info
        backend_name = "🚀 LMDeploy (Optimized)" if using_lmdeploy else "📦 Standard"
        device_info = "cuda" if use_lmdeploy else (backbone_device if not use_lmdeploy else "N/A")
        
        streaming_support = "✅ Có" if backbone_config['supports_streaming'] else "❌ Không"
        preencoded_note = "\n⚠️ Codec này cần sử dụng pre-encoded codes (.pt files)" if codec_config['use_preencoded'] else ""
        
        opt_info = ""
        if using_lmdeploy and hasattr(tts, 'get_optimization_stats'):
            stats = tts.get_optimization_stats()
            opt_info = (
                f"\n\n🔧 Tối ưu hóa:"
                f"\n  • Triton: {'✅' if stats['triton_enabled'] else '❌'}"
                f"\n  • Max Batch Size (Default): {stats.get('max_batch_size', 'N/A')}"
                f"\n  • Reference Cache: {stats['cached_references']} voices"
                f"\n  • Prefix Caching: ❌"
            )
        
        warning_msg = ""
        if lmdeploy_error_reason:
             warning_msg = (
                 f"\n\n⚠️ **Cảnh báo:** Không thể kích hoạt LMDeploy (Optimized Backend) do lỗi sau:\n"
                 f"👉 {lmdeploy_error_reason}\n"
                 f"💡 Hệ thống đã tự động chuyển về chế độ Standard (chậm hơn)."
             )

        success_msg = get_model_status_message()
        if warning_msg:
            success_msg += warning_msg
            
        # Prepare voice update
        try:
            # Get voices with descriptions for UI from SDK
            voices = tts.list_preset_voices()
        except Exception:
            voices = []

        has_voices = len(voices) > 0
        
        if has_voices:
            default_v = tts._default_voice
            
            # Helper to get values list
            is_tuple = (len(voices) > 0 and isinstance(voices[0], tuple))
            voice_values = [v[1] for v in voices] if is_tuple else voices
            
            if not default_v and voice_values:
                 default_v = voice_values[0]

            # Ensure default_v is in the list and selected correctly
            if default_v and default_v not in voice_values:
                if is_tuple:
                    # Try to find a nice description if possible, else use ID
                    voices.append((default_v, default_v))
                else:
                    voices.append(default_v)
            
            # Sort voices by name/label for better UX
            if is_tuple:
                voices.sort(key=lambda x: str(x[0]))
            else:
                voices.sort()

            voice_update = gr.update(choices=voices, value=default_v, interactive=True)
            
            global PRESET_VOICES_CACHE, CONV_VOICES_CACHE
            PRESET_VOICES_CACHE = voices
            
            # Filter voices for conversation tab (podcast=True)
            # Handle both boolean True/False and string "True"/"False"
            def _check_podcast(v_id):
                val = tts._preset_voices.get(v_id, {}).get('podcast', True)
                if isinstance(val, str):
                    return val.strip().lower() == "true"
                return bool(val)

            CONV_VOICES_CACHE = [v for v in voices if _check_podcast(v[1])]
            
            slot_dd_update = gr.update(choices=CONV_VOICES_CACHE)
            
            # Show Standard Tabs
            tab_p = gr.update(visible=True)
            tab_c = gr.update(visible=True)
            tab_sel = gr.update(selected="preset_mode")
            mode_state = "preset_mode"
        else:
            # Missing voices.json case
            msg = "⚠️ Không tìm thấy file voices.json. Vui lòng dùng Tab Voice Cloning."
            voice_update = gr.update(choices=[msg], value=msg, interactive=False)
            slot_dd_update = gr.update(choices=[])
            
            # Show Preset Tab (to see message) and Custom Tab
            tab_p = gr.update(visible=True)
            tab_c = gr.update(visible=True)
            tab_sel = gr.update(selected="preset_mode")
            mode_state = "preset_mode"

        # Check if v2 for conversation tab
        is_v2 = (backbone_choice == "VieNeu-TTS-v2 (GPU)" or backbone_choice == "VieNeu-TTS-v2 (CPU)")
        conv_tab_update = gr.update(visible=is_v2)

        # Update all MAX_SPEAKERS slot dropdowns
        slot_updates = [slot_dd_update] * MAX_SPEAKERS

        yield (
            success_msg,
            gr.update(interactive=True), # btn_generate
            gr.update(interactive=True), # btn_generate_conv
            gr.update(interactive=False), # btn_load
            gr.update(interactive=False), # btn_stop
            voice_update,
            tab_p, tab_c, tab_sel, mode_state,
            conv_tab_update,
            *slot_updates
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        model_loaded = False
        using_lmdeploy = False

        if "$env:CUDA_PATH" in str(e):
            yield (
                "❌ Lỗi khi tải model: Không tìm thấy biến môi trường CUDA_PATH. Vui lòng cài đặt NVIDIA GPU Computing Toolkit (https://developer.nvidia.com/cuda/toolkit)",
                gr.update(interactive=False),
                gr.update(interactive=False), # btn_generate_conv
                gr.update(interactive=True), # btn_load
                gr.update(interactive=False), # btn_stop
                gr.update(), # voice_select
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), # conv_tab
                *slot_no_updates
            )
        else: 
            yield (
                f"❌ Lỗi khi tải model: {str(e)}",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), # conv_tab
                *slot_no_updates
            )


def resolve_voice_id(v_id: str) -> str:
    """Robustly resolve voice ID, handling both display labels and internal IDs."""
    if not v_id:
        return v_id
    
    global PRESET_VOICES_CACHE
    if not PRESET_VOICES_CACHE:
        return v_id
        
    for item in PRESET_VOICES_CACHE:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            label, value = item[0], item[1]
            if v_id == value or v_id == label:
                return value
        else:
            if v_id == item:
                return item
            
    return v_id

# --- 2. DATA & HELPERS ---

def synthesize_speech(text: str, voice_choice: str, custom_audio, custom_text: str, 
                      mode_tab: str, generation_mode: str, use_batch: bool, max_batch_size_run: int,
                      temperature: float, max_chars_chunk: int, pronunciation_rules: str = "",
                      speaking_rate: float = 1.0, pitch_steps: float = 0.0, session_id: str = None):
    """Synthesis with optimization support and max batch size control"""
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy
    
    _STOP_EVENT.clear()  # Reset for new generation
    
    if not model_loaded or tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"
        return
    
    if not text or text.strip() == "":
        yield None, "⚠️ Vui lòng nhập văn bản!"
        return
    
    raw_text = apply_pronunciation_replacements(text.strip(), pronunciation_rules)
    raw_text = normalize_leading_fillers_for_tts(raw_text)
    raw_text = prepare_short_text_for_tts(raw_text)
    
    codec_config = CODEC_CONFIGS[current_codec]
    use_preencoded = codec_config['use_preencoded']
    
    
    # Setup Reference
    yield None, "📄 Đang xử lý Reference..."
    
    try:
        ref_codes = None
        ref_text_raw = ""
        
        if mode_tab == "preset_mode":
            if not voice_choice:
                raise ValueError("Vui lòng chọn giọng mẫu.")
            if "⚠️" in voice_choice:
                raise ValueError("Không có giọng mẫu khả dụng. Vui lòng chuyển sang Tab Voice Cloning.")
            
            # Use SDK method - handles caching and JSON internally
            v_id = resolve_voice_id(voice_choice)
            voice_data = tts.get_preset_voice(v_id)
            ref_codes = voice_data['codes']
            ref_text_raw = voice_data['text']
        
        elif mode_tab == "custom_mode":
            if custom_audio is None:
                raise ValueError("Vui lòng upload file Audio mẫu (Reference Audio)!")
            
            is_turbo = "v2-Turbo" in (current_backbone or "")
            if not is_turbo and (not custom_text or not custom_text.strip()):
                raise ValueError("Vui lòng nhập nội dung văn bản của Audio mẫu (Reference Text)!")
            
            ref_text_raw = custom_text.strip() if custom_text else ""
            ref_codes = tts.encode_reference(custom_audio)

        # Ensure numpy for inference
        if 'torch' in sys.modules:
            import torch
            if isinstance(ref_codes, torch.Tensor):
                ref_codes = ref_codes.cpu().numpy()

    except Exception as e:
        yield None, f"❌ Lỗi xử lý Reference Audio: {str(e)}"
        return
    
    # === STANDARD MODE ===
    if generation_mode == "Standard (Một lần)":
        backend_name = "LMDeploy" if using_lmdeploy else "Standard"

        normalized_text = safe_normalize_text_for_tts(raw_text)
        is_v2_turbo = "v2-Turbo" in (current_backbone or "")
        effective_max_chars = get_effective_chunk_chars(max_chars_chunk, is_v2_turbo, normalized_text)
        
        if is_v2_turbo:
            # Phoneme-based splitting for accurate progress reporting
            phonemes = phonemize_with_dict(normalized_text, skip_normalize=True)
            text_chunks = split_into_chunks_v2(phonemes, max_chunk_size=effective_max_chars)
        else:
            text_chunks = split_text_for_standard_tts(normalized_text, max_chars=effective_max_chars)
            
        total_chunks = len(text_chunks)

        batch_info = " (Batch Mode)" if use_batch and using_lmdeploy and total_chunks > 1 else ""
        
        # Show batch size info
        batch_size_info = ""
        if use_batch and using_lmdeploy and hasattr(tts, 'max_batch_size'):
            batch_size_info = f" [Max batch: {tts.max_batch_size}]"
        
        yield None, f"🚀 Bắt đầu tổng hợp {backend_name}{batch_info}{batch_size_info} ({total_chunks} đoạn)..."
        
        all_wavs = []
        sr = 24000
        
        start_time = time.time()

        try:
            if is_v2_turbo:
                # Sequential processing with progress updates
                total_chunks = len(text_chunks)
                for i, chunk in enumerate(text_chunks):
                    if _STOP_EVENT.is_set():
                        yield None, "⏹️ Đã dừng tạo giọng nói."
                        return
                    yield None, f"⚡ Turbo v2: Đang xử lý đoạn {i+1}/{total_chunks}..."
                    chunk_wavs = synthesize_chunk_with_retry(
                        tts,
                        chunk.text,
                        ref_codes,
                        ref_text_raw,
                        temperature,
                        effective_max_chars,
                        True,
                    )
                    all_wavs.extend(chunk_wavs)
                    # Add silence between Gradio-level chunks for Turbo
                    if i < total_chunks - 1:
                        sil_dur = get_silence_duration_v2(chunk)
                        sil_wav = np.zeros(int(sr * sil_dur), dtype=np.float32)
                        all_wavs.append(sil_wav)
            
            # Use batch processing if enabled and using LMDeploy (for v1)
            elif use_batch and using_lmdeploy and hasattr(tts, 'infer_batch') and total_chunks > 1:
                # Process in mini-batches to allow cancellation between batches
                num_batches = (total_chunks + max_batch_size_run - 1) // max_batch_size_run
                
                for i in range(0, total_chunks, max_batch_size_run):
                    if _STOP_EVENT.is_set():
                        print("🛑 Synthesis stopped during batch processing.")
                        yield None, "⏹️ Đã dừng tạo giọng nói."
                        return
                    
                    batch_idx = i // max_batch_size_run
                    yield None, f"⚡ Đang xử lý batch {batch_idx+1}/{num_batches} (đoạn {i+1}-{min(i+max_batch_size_run, total_chunks)})..."
                    
                    current_batch = text_chunks[i : i + max_batch_size_run]
                    batch_wavs = tts.infer_batch(
                        current_batch, 
                        ref_codes=ref_codes, 
                        ref_text=ref_text_raw,
                        max_batch_size=max_batch_size_run,
                        temperature=adjust_temperature_for_short_text(" ".join(str(c) for c in current_batch), temperature),
                        max_tokens=max(estimate_chunk_generation_budget(tts, c, ref_codes, ref_text_raw) for c in current_batch),
                        skip_normalize=True
                    )
                    for batch_chunk_text, chunk_wav in zip(current_batch, batch_wavs):
                        if chunk_wav is not None and len(chunk_wav) > 0:
                            all_wavs.append(chunk_wav)
                            continue
                        retry_wavs = synthesize_chunk_with_retry(
                            tts,
                            batch_chunk_text,
                            ref_codes,
                            ref_text_raw,
                            temperature,
                            effective_max_chars,
                            False,
                        )
                        all_wavs.extend(retry_wavs)

            else:
                # Sequential processing (PyTorch or GGUF v1)
                for i, chunk in enumerate(text_chunks):
                    if _STOP_EVENT.is_set():
                        yield None, "⏹️ Đã dừng tạo giọng nói."
                        return
                    yield None, f"⏳ Đang xử lý đoạn {i+1}/{total_chunks}..."
                    chunk_wavs = synthesize_chunk_with_retry(
                        tts,
                        chunk,
                        ref_codes,
                        ref_text_raw,
                        temperature,
                        effective_max_chars,
                        False,
                    )
                    all_wavs.extend(chunk_wavs)
            
            if not all_wavs:
                yield None, "❌ Không sinh được audio nào."
                return
            
            yield None, "💾 Đang ghép file và lưu..."
            
            # Use utility function for joining with silence/crossfade
            # Default silence=0.15s to match SDK
            silence_p = 0.15 if not is_v2_turbo else 0.0 # Turbo adds silence internally
            final_wav = join_audio_chunks(all_wavs, sr=sr, silence_p=silence_p)
            final_wav = apply_audio_style_controls(final_wav, sr, speaking_rate, pitch_steps)
            final_wav = append_output_tail_silence(final_wav, sr)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                output_path = tmp.name
            
            process_time = time.time() - start_time
            backend_info = f" (Backend: {'LMDeploy 🚀' if using_lmdeploy else 'Standard 📦'})"
            speed_info = f", Tốc độ: {len(final_wav)/sr/process_time:.2f}x realtime" if process_time > 0 else ""
            
            
            yield output_path, f"✅ Hoàn tất! (Thời gian: {process_time:.2f}s{speed_info}){backend_info}"
            
            # Cleanup memory
            if using_lmdeploy and hasattr(tts, 'cleanup_memory'):
                tts.cleanup_memory()
            
            cleanup_gpu_memory()
            
        except Exception as e:
            # Check for CUDA OOM specifically if torch is loaded
            if 'torch' in sys.modules:
                import torch
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    cleanup_gpu_memory()
                    yield None, (
                        f"❌ GPU hết VRAM! Hãy thử:\n"
                        f"• Giảm Max Batch Size (hiện tại: {tts.max_batch_size if hasattr(tts, 'max_batch_size') else 'N/A'})\n"
                        f"• Giảm độ dài văn bản\n\n"
                        f"Chi tiết: {str(e)}"
                    )
                    return
            
            import traceback
            traceback.print_exc()
            cleanup_gpu_memory()
            yield None, f"❌ Lỗi Standard Mode: {str(e)}"
            return
    
    # === STREAMING MODE ===
    else:
        sr = 24000
        crossfade_samples = int(sr * 0.03)
        audio_queue = queue.Queue(maxsize=100)
        PRE_BUFFER_SIZE = 3
        
        end_event = threading.Event()
        error_event = threading.Event()
        error_msg = ""
        
        normalized_text = safe_normalize_text_for_tts(raw_text)
        is_v2_turbo = "v2-Turbo" in (current_backbone or "")
        effective_max_chars = get_effective_chunk_chars(max_chars_chunk, is_v2_turbo, normalized_text)
        if is_v2_turbo:
            phonemes = phonemize_with_dict(normalized_text, skip_normalize=True)
            text_chunks = split_into_chunks_v2(phonemes, max_chunk_size=effective_max_chars)
        else:
            text_chunks = split_text_for_standard_tts(normalized_text, max_chars=effective_max_chars)
        
        def producer_thread():
            nonlocal error_msg
            try:
                previous_tail = None
                
                for i, chunk_text in enumerate(text_chunks):
                    if _STOP_EVENT.is_set():
                        break
                    
                    if is_v2_turbo:
                        stream_gen = tts.infer_stream(
                            chunk_text, 
                            ref_codes=ref_codes, 
                            temperature=temperature,
                            max_chars=effective_max_chars,
                            skip_normalize=True,
                            skip_phonemize=True,
                            emotion_tag=""
                        )
                    else:
                        stream_gen = tts.infer_stream(
                            chunk_text, 
                            ref_codes=ref_codes, 
                            ref_text=ref_text_raw,
                            temperature=temperature,
                            max_chars=effective_max_chars,
                            skip_normalize=True,
                            emotion_tag=""
                        )
                    
                    for part_idx, audio_part in enumerate(stream_gen):
                        if _STOP_EVENT.is_set():
                            break
                        if audio_part is None or len(audio_part) == 0:
                            continue
                        
                        if previous_tail is not None and len(previous_tail) > 0:
                            overlap = min(len(previous_tail), len(audio_part), crossfade_samples)
                            if overlap > 0:
                                fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                                fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                                
                                blended = (audio_part[:overlap] * fade_in + 
                                         previous_tail[-overlap:] * fade_out)
                                
                                processed = np.concatenate([
                                    previous_tail[:-overlap] if len(previous_tail) > overlap else np.array([]),
                                    blended,
                                    audio_part[overlap:]
                                ])
                            else:
                                processed = np.concatenate([previous_tail, audio_part])
                            
                            tail_size = min(crossfade_samples, len(processed))
                            previous_tail = processed[-tail_size:].copy()
                            output_chunk = processed[:-tail_size] if len(processed) > tail_size else processed
                        else:
                            tail_size = min(crossfade_samples, len(audio_part))
                            previous_tail = audio_part[-tail_size:].copy()
                            output_chunk = audio_part[:-tail_size] if len(audio_part) > tail_size else audio_part
                        
                        if len(output_chunk) > 0:
                            audio_queue.put((sr, output_chunk))
                            
                    # Add silence between chunks for Turbo v2
                    if is_v2_turbo and i < len(text_chunks) - 1:
                        sil_dur = get_silence_duration_v2(chunk_text)
                        sil_wav = np.zeros(int(sr * sil_dur), dtype=np.float32)
                        audio_queue.put((sr, sil_wav))
                
                if previous_tail is not None and len(previous_tail) > 0:
                    audio_queue.put((sr, previous_tail))
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                error_event.set()
            finally:
                end_event.set()
                audio_queue.put(None)
        
        threading.Thread(target=producer_thread, daemon=True).start()
        
        yield (sr, np.zeros(int(sr * 0.05))), "📄 Đang buffering..."
        
        pre_buffer = []
        while len(pre_buffer) < PRE_BUFFER_SIZE:
            try:
                item = audio_queue.get(timeout=5.0)
                if item is None:
                    break
                pre_buffer.append(item)
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    return
                break
        
        full_audio_buffer = []
        backend_info = "🚀 LMDeploy" if using_lmdeploy else "📦 Standard"
        for sr, audio_data in pre_buffer:
            full_audio_buffer.append(audio_data)
            yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."
        
        while True:
            try:
                item = audio_queue.get(timeout=0.05)
                if item is None:
                    break
                sr, audio_data = item
                full_audio_buffer.append(audio_data)
                yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    break
                if end_event.is_set() and audio_queue.empty():
                    break
                continue
        
        if full_audio_buffer:
            final_wav = np.concatenate(full_audio_buffer)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                
                yield tmp.name, f"✅ Hoàn tất Streaming! ({backend_info})"
            
            # Cleanup memory
            if using_lmdeploy and hasattr(tts, 'cleanup_memory'):
                tts.cleanup_memory()
            
            cleanup_gpu_memory()


# --- 3. CONVERSATION LOGIC ---

def synthesize_conversation(
    script_text: str,
    *args
):
    """
    Synthesizes multi-speaker conversation from a script.

    Gradio passes speaker name boxes and voice dropdowns as individual positional args.
    Layout: args[0..MAX_SPEAKERS-1] = speaker names, args[MAX_SPEAKERS..2*MAX_SPEAKERS-1] = voice IDs,
    args[2*MAX_SPEAKERS] = silence_duration, args[2*MAX_SPEAKERS+1] = temperature,
    args[2*MAX_SPEAKERS+2] = max_chars_chunk, args[2*MAX_SPEAKERS+3] = session_id
    """
    speaker_names     = list(args[:MAX_SPEAKERS])
    speaker_voices    = list(args[MAX_SPEAKERS:MAX_SPEAKERS*2])
    silence_duration  = args[MAX_SPEAKERS * 2]
    temperature       = args[MAX_SPEAKERS * 2 + 1]
    max_chars_chunk   = args[MAX_SPEAKERS * 2 + 2]
    session_id        = args[MAX_SPEAKERS * 2 + 3] if len(args) > MAX_SPEAKERS * 2 + 3 else None

    global tts, model_loaded, using_lmdeploy
    
    _STOP_EVENT.clear()
    
    if not model_loaded or tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"
        return
        
    if not script_text or script_text.strip() == "":
        yield None, "⚠️ Vui lòng nhập kịch bản hội thoại!"
        return

    # 1. Parse Script
    lines = []
    for line in script_text.strip().split('\n'):
        if not line.strip(): continue
        if ':' in line:
            parts = line.split(':', 1)
            lines.append({'speaker': parts[0].strip(), 'text': parts[1].strip()})
        else:
            if lines:
                lines[-1]['text'] += " " + line.strip()
            else:
                lines.append({'speaker': 'Narrator', 'text': line.strip()})

    if not lines:
        yield None, "⚠️ Không tìm thấy lời thoại hợp lệ (định dạng Nhân vật: Lời thoại)!"
        return

    # 2. Build Speaker Mapping from individual slot components
    mapping = {}
    for name, voice in zip(speaker_names, speaker_voices):
        name = str(name).strip() if name else ""
        if not name: continue
        # Use lowercase key for robust matching
        v_id = resolve_voice_id(str(voice)) if voice else ""
        mapping[name.lower()] = {
            'type': 'Preset',
            'voice': v_id,
            'ref_text': ''
        }


    # 3. Process Each Line
    all_wavs = []
    sr = 24000
    total_lines = len(lines)
    
    yield None, f"🎭 Đang khởi tạo hội thoại ({total_lines} câu)..."
    
    start_time = time.time()
    
    try:
        for i, line in enumerate(lines):
            if _STOP_EVENT.is_set():
                yield None, "⏹️ Đã dừng hội thoại."
                return
            spk_name = line['speaker']
            text = line['text']
            
            yield None, f"⏳ [{i+1}/{total_lines}] {spk_name}: {text[:30]}..."
            
            # Determine voice
            ref_codes = None
            ref_text_val = None
            current_voice_obj = None
            
            # Case-insensitive lookup
            config = mapping.get(spk_name.lower())
            
            if not config:
                print(f"  ⚠️ Character '{spk_name}' not found in mapping. Fallback to default.")
                # Fallback to default if speaker not mapped
                try:
                    # Get default voice data
                    default_v_id = tts._default_voice
                    if not default_v_id:
                        dv_list = tts.list_preset_voices()
                        if dv_list:
                            first = dv_list[0]
                            default_v_id = first[1] if isinstance(first, tuple) else first
                    
                    if default_v_id:
                        current_voice_obj = tts.get_preset_voice(default_v_id)
                        ref_codes = current_voice_obj['codes']
                        ref_text_val = current_voice_obj['text']
                except Exception as e:
                    print(f"  ❌ Fallback failed: {e}")
            else:
                try:
                    v_id = config['voice']
                    if config['type'] == "Preset":
                        current_voice_obj = tts.get_preset_voice(v_id)
                        if current_voice_obj and 'codes' in current_voice_obj:
                            ref_codes = current_voice_obj['codes']
                            ref_text_val = current_voice_obj['text']
                        else:
                            print(f"  ❌ Could not find codes for voice '{v_id}'")
                    else: # Custom
                        if v_id and os.path.exists(v_id):
                            ref_codes = tts.encode_reference(v_id)
                            ref_text_val = config.get('ref_text', '')
                            current_voice_obj = {'codes': ref_codes, 'text': ref_text_val}
                            print(f"  🦜 Using custom voice for '{spk_name}'")
                except Exception as e:
                    print(f"  ❌ Lỗi nạp giọng cho {spk_name} (ID: {config.get('voice')}): {e}")
            
            # Ensure numpy for inference
            if 'torch' in sys.modules:
                import torch
                if isinstance(ref_codes, torch.Tensor):
                    ref_codes = ref_codes.cpu().numpy()

            # Infer audio
            try:
                wav = tts.infer(
                    text,
                    voice=current_voice_obj, # Use full voice object
                    ref_codes=ref_codes,     # Fallback if object not supported
                    ref_text=ref_text_val,
                    temperature=temperature,
                    max_chars=max_chars_chunk,
                    emotion_tag="<|emotion_0|>" # Emotion tag for conversation
                )
                
                all_wavs.append(wav)
                
                # Add silence between turns
                if i < total_lines - 1 and silence_duration > 0:
                    silence_len = int(sr * silence_duration)
                    silence = np.zeros(silence_len)
                    all_wavs.append(silence)
                    
            except Exception as e:
                print(f"❌ Lỗi tổng hợp câu {i+1}: {e}")
                continue

        if not all_wavs:
            yield None, "❌ Không thể tạo được âm thanh nào!"
            return

        # 4. Merge and Output
        yield None, "🪄 Đang ghép nối âm thanh..."
        final_wav = np.concatenate(all_wavs)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, final_wav, sr)
            elapsed = time.time() - start_time
            yield tmp.name, f"✅ Hoàn tất hội thoại! ({total_lines} câu, xử lý trong {elapsed:.1f}s)"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, f"❌ Lỗi hệ thống: {str(e)}"

def extract_speakers_from_script(script):
    """Find unique speakers and return gr.update() lists for the 8 slot components."""
    global CONV_VOICES_CACHE
    if not script:
        # Hide all slots
        name_updates = [gr.update(value="", visible=False)] * MAX_SPEAKERS
        dd_updates   = [gr.update(value=None, visible=False)] * MAX_SPEAKERS
        row_updates  = [gr.update(visible=False)] * MAX_SPEAKERS
        return name_updates + dd_updates + row_updates

    speakers = []
    seen = set()
    for line in script.strip().split('\n'):
        if ':' in line:
            s = line.split(':', 1)[0].strip()
            if s and s not in seen:
                seen.add(s)
                speakers.append(s)

    # Auto-match each speaker name to a preset voice
    def _best_match(name):
        if not CONV_VOICES_CACHE:
            return None
        
        name_l = name.lower()
        
        # 0. Manual overrides for specific common names
        overrides = {
            "phương": "Trúc Ly",
            "dũng": "Thanh Bình",
            "hùng": "Thái Sơn"
        }
        if name_l in overrides:
            target = overrides[name_l].lower()
            for v in CONV_VOICES_CACHE:
                label, value = (v[0], v[1]) if isinstance(v, tuple) else (v, v)
                if target in label.lower() or target in value.lower():
                    return value

        # 1. Try to find name in labels or values
        for v in CONV_VOICES_CACHE:
            label, value = (v[0], v[1]) if isinstance(v, tuple) else (v, v)
            if name_l == label.lower() or name_l == value.lower():
                return value
        
        # 2. Fuzzy match (contains)
        for v in CONV_VOICES_CACHE:
            label, value = (v[0], v[1]) if isinstance(v, tuple) else (v, v)
            if name_l in label.lower() or name_l in value.lower() or label.lower() in name_l or value.lower() in name_l:
                return value
        
        # 3. Default to first voice if no match
        first_voice = CONV_VOICES_CACHE[0]
        return first_voice[1] if isinstance(first_voice, tuple) else first_voice

    name_updates, dd_updates, row_updates = [], [], []
    for i in range(MAX_SPEAKERS):
        if i < len(speakers):
            name_updates.append(gr.update(value=speakers[i], visible=True))
            dd_updates.append(gr.update(value=_best_match(speakers[i]), choices=CONV_VOICES_CACHE, visible=True))
            row_updates.append(gr.update(visible=True))
        else:
            name_updates.append(gr.update(value="", visible=False))
            dd_updates.append(gr.update(value=None, choices=CONV_VOICES_CACHE, visible=False))
            row_updates.append(gr.update(visible=False))

    return name_updates + dd_updates + row_updates


# --- 4. UI SETUP ---
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui'],
).set(
    button_primary_background_fill="linear-gradient(90deg, #6366f1 0%, #0ea5e9 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #4f46e5 0%, #0284c7 100%)",
)

css = """
:root {
    --vt-bg: #0b1220;
    --vt-panel: #111827;
    --vt-panel-2: #172033;
    --vt-border: rgba(148, 163, 184, 0.22);
    --vt-text: #e5edf7;
    --vt-muted: #9fb0c7;
    --vt-primary: #38bdf8;
    --vt-accent: #22c55e;
    --vt-warn: #f59e0b;
}
.gradio-container {
    background:
        radial-gradient(circle at 20% 0%, rgba(56, 189, 248, 0.12), transparent 32rem),
        linear-gradient(180deg, #08111f 0%, #0b1220 44%, #101827 100%) !important;
    color: var(--vt-text) !important;
}
.container {
    max-width: 1180px;
    margin: auto;
    padding: 14px 14px 32px;
}
.header-box {
    text-align: left;
    margin-bottom: 12px;
    padding: 14px 18px;
    background: rgba(15, 23, 42, 0.84);
    border: 1px solid var(--vt-border);
    border-radius: 8px;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.24);
    color: white !important;
}
.header-title {
    font-size: 1.45rem;
    line-height: 1.1;
    font-weight: 800;
    color: white !important;
    margin-bottom: 8px;
}
.gradient-text {
    background: -webkit-linear-gradient(45deg, #e2e8f0, #38bdf8 55%, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-icon {
    color: white;
}
.status-box {
    font-weight: 500;
    border: 1px solid var(--vt-border);
    background: rgba(15, 23, 42, 0.7);
    border-radius: 8px;
}
.status-box textarea {
    text-align: left;
    font-family: inherit;
}
.block, .form, .panel {
    border-radius: 8px !important;
}
.block {
    border-color: var(--vt-border) !important;
    background: rgba(17, 24, 39, 0.72) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.16);
}
.form {
    background: transparent !important;
    border-color: transparent !important;
}
label, .wrap label, .block label {
    color: #dbe7f5 !important;
    font-weight: 700 !important;
}
label > span, .wrap label > span, .block label > span {
    background: transparent !important;
    color: #dbe7f5 !important;
    padding: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    border: 0 !important;
}
textarea, input, select {
    background: rgba(8, 15, 28, 0.74) !important;
    border: 1px solid rgba(148, 163, 184, 0.24) !important;
    border-radius: 8px !important;
    color: #eef6ff !important;
}
.container .form > .wrap,
.container .gradio-dropdown,
.container .gradio-radio,
.container .gradio-textbox,
.container .gradio-slider,
.container .gradio-checkbox {
    background: rgba(12, 20, 34, 0.42) !important;
    border: 1px solid rgba(148, 163, 184, 0.16) !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
}
.container .gradio-dropdown input,
.container .gradio-dropdown select {
    min-height: 42px !important;
    font-size: 0.98rem !important;
    border-radius: 6px !important;
}
.model-select {
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
}
.model-select label {
    margin-bottom: 6px !important;
    font-size: 0.84rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #9fb0c7 !important;
}
.model-select select,
.model-select input {
    background: #ecf3fb !important;
    color: #0f172a !important;
    border: 1px solid rgba(56, 189, 248, 0.35) !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}
.model-select:focus-within select,
.model-select:focus-within input {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.16) !important;
}
.device-choices {
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
}
.device-choices > label {
    margin-bottom: 6px !important;
    font-size: 0.84rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #9fb0c7 !important;
}
.device-choices .wrap {
    gap: 8px !important;
}
.device-choices .wrap label {
    margin: 0 !important;
    padding: 10px 14px !important;
    background: rgba(17, 24, 39, 0.86) !important;
    border: 1px solid rgba(148, 163, 184, 0.22) !important;
    border-radius: 999px !important;
    color: #dbe7f5 !important;
}
.device-choices .wrap label *,
.device-choices .wrap label span,
.device-choices .wrap label div {
    color: inherit !important;
}
.device-choices .wrap label:has(input:checked) {
    background: #ecf3fb !important;
    border-color: rgba(56, 189, 248, 0.4) !important;
    color: #0f172a !important;
}
.device-choices .wrap label:has(input:checked) *,
.device-choices .wrap label:has(input:checked) span,
.device-choices .wrap label:has(input:checked) div {
    color: #0f172a !important;
}
.container .gradio-dropdown label,
.container .gradio-radio label {
    margin-bottom: 8px !important;
    font-size: 0.92rem !important;
}
.container .gradio-radio .wrap {
    gap: 8px !important;
}
textarea:focus, input:focus {
    border-color: rgba(56, 189, 248, 0.75) !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.16) !important;
}
button.primary {
    background: linear-gradient(90deg, #0284c7 0%, #16a34a 100%) !important;
    border: 0 !important;
    border-radius: 8px !important;
    min-height: 44px;
    font-weight: 800 !important;
}
button.secondary, button:not(.primary) {
    border-radius: 8px !important;
}
.tabs {
    border-color: rgba(148, 163, 184, 0.18) !important;
}
.tab-nav button {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 700 !important;
}
.tab-nav button.selected {
    color: #e0f2fe !important;
    border-color: #38bdf8 !important;
}
.accordion {
    border-color: var(--vt-border) !important;
    border-radius: 8px !important;
    background: rgba(15, 23, 42, 0.54) !important;
}
.wrap.svelte-p5q82i, .wrap {
    gap: 10px !important;
}
.model-card-content {
    display: none;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 15px;
    font-size: 0.9rem;
    text-align: center;
    color: white !important;
}
.model-card-item {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    color: white !important;
}
.model-card-item strong {
    color: white !important;
}
.model-card-item span {
    color: white !important;
}
.model-card-link {
    color: #60A5FA;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s;
}
.model-card-link:hover {
    color: #22D3EE;
    text-decoration: underline;
}
.warning-banner {
    display: none;
    background: rgba(15, 23, 42, 0.68);
    border: 1px solid rgba(245, 158, 11, 0.24);
    border-left: 3px solid var(--vt-warn);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
}
.warning-banner-title {
    color: #fbbf24;
    font-weight: 700;
    font-size: 1.1rem;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
}
.warning-banner-grid {
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
}
.warning-banner-item {
    flex: 1;
    min-width: 240px;
    background: rgba(30, 41, 59, 0.72);
    padding: 12px;
    border-radius: 8px;
    border: 1px solid rgba(148, 163, 184, 0.18);
}
.warning-banner-item strong {
    color: #bae6fd;
    display: block;
    margin-bottom: 4px;
    font-size: 0.95rem;
}
.warning-banner-content {
    color: var(--vt-muted);
    font-size: 0.9rem;
    line-height: 1.5;
}
.warning-banner-content b {
    color: #f8fafc;
    background: rgba(56, 189, 248, 0.14);
    padding: 1px 4px;
    border-radius: 4px;
}
.script-box textarea {
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
}
.speaker-table {
    margin-top: 10px;
}
.audio-container, audio {
    border-radius: 8px !important;
}
footer {
    display: none !important;
}
.container .block {
    margin-bottom: 12px !important;
}
.workspace-title {
    margin: 4px 0 8px;
    color: #e2e8f0;
    font-weight: 800;
}
"""

EXAMPLES_LIST = [
    ["Về miền Tây không chỉ để ngắm nhìn sông nước hữu tình, mà còn để cảm nhận tấm chân tình của người dân nơi đây.", "Vĩnh (nam miền Nam)"],
    ["Hà Nội những ngày vào thu mang một vẻ đẹp trầm mặc và cổ kính đến lạ thường.", "Bình (nam miền Bắc)"],
]


# Favicon (Parrot Emoji)
head_html = """
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🦜</text></svg>">
"""

with gr.Blocks(theme=theme, css=css, title="VieNeu-TTS", head=head_html) as demo:
    # Session ID for cancellation tracking
    session_id_state = gr.State("")

    with gr.Column(elem_classes="container"):
        gr.HTML("""
<div class="header-box">
    <h1 class="header-title">
        <span class="header-icon">🦜</span>
        <span class="gradient-text">VieNeu-TTS Studio</span>
    </h1>
    <div class="model-card-content">
        <div class="model-card-item">
            <strong>Models:</strong>
            <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS" target="_blank" class="model-card-link">VieNeu-TTS</a>
            <span>•</span>
            <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2" target="_blank" class="model-card-link">VieNeu-TTS-v2</a>
        </div>
        <div class="model-card-item">
            <strong>Repository:</strong>
            <a href="https://github.com/pnnbao97/VieNeu-TTS" target="_blank" class="model-card-link">GitHub</a>
        </div>
        <div class="model-card-item">
            <strong>Tác giả:</strong>
            <a href="https://www.facebook.com/pnnbao97" target="_blank" class="model-card-link">Phạm Nguyễn Ngọc Bảo</a>
        </div>
        <div class="model-card-item">
            <strong>Discord:</strong>
            <a href="https://discord.gg/yJt8kzjzWZ" target="_blank" class="model-card-link">Tham gia cộng đồng</a>
        </div>
    </div>
</div>
        """)
        
        # --- CONFIGURATION ---
        with gr.Accordion("1. Chọn model", open=True):
            with gr.Row():
                # --- DEFAULT VALUES ---
                DEFAULT_TEXT_GPU = "Ừm... mình thử nghe đoạn này một chút nha.\nGiọng này có vẻ ổn hơn rồi đó.\nNhưng mình vẫn muốn nó tự nhiên hơn một chút."
                DEFAULT_TEXT_TURBO = (
                    "Xin chào, đây là bản thử giọng bằng VieNeu TTS.\nMình muốn giọng đọc tự nhiên, rõ ràng, và không bị đều quá."
                )

                saved_model_state = load_model_selection_state()
                default_backbone = saved_model_state["backbone_choice"]
                
                # Default parameters based on backbone
                if "Turbo" in default_backbone:
                    default_codec = "VieNeu-Codec"
                    default_temp = 0.4
                    default_text = DEFAULT_TEXT_TURBO
                elif "(CPU)" in default_backbone:
                    default_codec = "NeuCodec (ONNX)"
                    default_temp = 0.7
                    default_text = DEFAULT_TEXT_GPU
                else:
                    default_codec = "NeuCodec (Distill)" if "NeuCodec (Distill)" in CODEC_CONFIGS else list(CODEC_CONFIGS.keys())[0]
                    default_temp = 0.7
                    default_text = DEFAULT_TEXT_GPU

                if saved_model_state["codec_choice"] in CODEC_CONFIGS:
                    default_codec = saved_model_state["codec_choice"]

                backbone_select = gr.Dropdown(
                    list(BACKBONE_CONFIGS.keys()) + ["Custom Model"], 
                    value=default_backbone,
                    elem_classes="model-select",
                    label="🦜 Backbone"
                )
                codec_select = gr.Dropdown(
                    list(CODEC_CONFIGS.keys()), 
                    value=default_codec,
                    elem_classes="model-select",
                    label="🎵 Codec",
                    interactive=True
                )
                device_choice = gr.Radio(
                    get_available_devices(),
                    value=saved_model_state["device_choice"],
                    label="🖥️ Device",
                    elem_classes="device-choices",
                )
            
            with gr.Row(visible=False) as custom_model_group:
                custom_backbone_model_id = gr.Textbox(
                    label="📦 Custom Model ID",
                    placeholder="pnnbao-ump/VieNeu-TTS-0.3B-lora-ngoc-huyen",
                    info="Nhập HuggingFace Repo ID hoặc đường dẫn local",
                    value=saved_model_state["custom_model_id"],
                    scale=2
                )
                custom_backbone_hf_token = gr.Textbox(
                    label="🔑 HF Token (nếu private)",
                    placeholder="Để trống nếu repo public",
                    type="password",
                    info="Token để truy cập repo private",
                    scale=1
                )
                base_model_choices = [k for k in BACKBONE_CONFIGS.keys() if "turbo" not in k.lower() and k != "Custom Model"]
                custom_backbone_base_model = gr.Dropdown(
                    base_model_choices,
                    label="🔗 Base Model (cho LoRA)",
                    value=saved_model_state["custom_base_model"] if saved_model_state["custom_base_model"] in base_model_choices else (base_model_choices[0] if base_model_choices else None),
                    visible=False,
                    info="Model gốc để merge với LoRA (GPU Only)",
                    scale=1
                )
            
            with gr.Row():
                use_lmdeploy_cb = gr.Checkbox(
                    value=saved_model_state["force_lmdeploy"], 
                    label="🚀 Optimize with LMDeploy (Khuyên dùng cho NVIDIA GPU)",
                    info="Tick nếu bạn dùng GPU để tăng tốc độ tổng hợp đáng kể."
                )
            
            
            gr.Markdown("Chọn model, bấm **Tải model**, sau đó chuyển xuống khu vực nhập nội dung.")
            
            gr.HTML("""
            <div class="warning-banner">
                <div class="warning-banner-title">
                    🦜 Gợi ý tối ưu hiệu năng
                </div>
                <div class="warning-banner-grid">
                    <div class="warning-banner-item">
                        <strong>🐆 Hệ máy GPU</strong>
                        <div class="warning-banner-content">
                            Chế độ podcast và song ngữ Anh Việt đã được hỗ trợ bắt đầu từ phiên bản <b>VieNeu-TTS-v2</b>, tuy nhiên quá trình kiểm thử vẫn đang tiếp tục, có thể sẽ xảy ra lỗi không mong muốn, nếu có lỗi các bạn hãy thông báo với chúng tôi tại: https://discord.com/invite/yJt8kzjzWZ. Trong trường hợp bạn cần sự ổn định hãy sử dụng <b>VieNeu-TTS (GPU)</b>. 
                        </div>
                    </div>
                    <div class="warning-banner-item" style="background: #dcfce7; border-color: #86efac;">
                        <strong style="color: #15803d;">🐢 Hệ máy CPU</strong>
                        <div class="warning-banner-content" style="color: #166534;">
                            Mặc định là <b>VieNeu-TTS-v2-Turbo (CPU)</b> để tốc độ tổng hợp nhanh nhất có thể, tuy nhiên có hạn chế về chất lượng âm thanh. Trong trường hợp bạn cần chất lượng tốt nhất hãy sử dụng <b>VieNeu-TTS-v2 (CPU)</b>.
                        </div>
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 0.85rem; color: #92400e; border-top: 1px dashed #fcd34d; padding-top: 8px;">
                    💡 <b>Mẹo:</b> Nếu máy bạn có GPU mà không thấy các phiên bản GPU hãy xem lại cách cài đặt uv sync --group gpu
                </div>
            </div>
            """)

            btn_load = gr.Button("Tải model", variant="primary", interactive=False)
            model_status = gr.Markdown("⏳ Chưa tải model.")
        
        gr.Markdown("<div class='workspace-title'>2. Soạn nội dung và tạo giọng</div>")
        with gr.Row(elem_classes="container"):
            # --- INPUT ---
            with gr.Column(scale=3):
                with gr.Tabs() as main_input_tabs:
                    # --- TAB 1: SINGLE SPEAKER ---
                    with gr.Tab("🦜 Đọc truyện", id="single_tab") as single_tab:
                        text_input = gr.Textbox(
                            label=f"Văn bản",
                            lines=8,
                            value=default_text,
                        )
                        
                        with gr.Tabs() as tabs:
                            with gr.TabItem("👤 Preset", id="preset_mode") as tab_preset:
                                voice_select = gr.Dropdown(choices=[], value=None, label="Giọng mẫu", allow_custom_value=True)
                            
                            with gr.TabItem("🦜 Voice Cloning", id="custom_mode") as tab_custom:
                                with gr.Group(visible=True) as cloning_elements_group:
                                    custom_audio = gr.Audio(label="Audio giọng mẫu (3-5 giây) (.wav)", type="filepath")
                                    cloning_warning_msg = gr.Markdown(visible=False, elem_id="cloning-warning")
                                    custom_text = gr.Textbox(label="Nội dung audio mẫu - vui lòng gõ đúng nội dung của audio mẫu - kể cả dấu câu vì model rất nhạy cảm với dấu câu (.,?!)")
                                    with gr.Accordion("💾 Lưu voice sau khi nghe thử", open=False):
                                        saved_voice_name = gr.Textbox(
                                            label="Tên voice",
                                            placeholder="Vi du: Giong_Nhi_Nhanh"
                                        )
                                        saved_voice_desc = gr.Textbox(
                                            label="Mô tả voice",
                                            placeholder="Vi du: Giọng nữ tự nhiên, nhí nhảnh"
                                        )
                                        btn_save_voice = gr.Button("➕ Thêm voice vào hệ thống", variant="secondary")
                                        save_voice_status = gr.Markdown("")
                                    gr.Examples(
                                        examples=[
                                            [os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "audio_ref", "example.wav"), "Ví dụ 2. Tính trung bình của dãy số."],
                                            [os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "audio_ref", "example_2.wav"), "Trên thực tế, các nghi ngờ đã bắt đầu xuất hiện."],
                                            [os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "audio_ref", "example_3.wav"), "Cậu có nhìn thấy không?"],
                                            [os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "audio_ref", "example_4.wav"), "Tết là dịp mọi người háo hức đón chào một năm mới với nhiều hy vọng và mong ước."]
                                        ],
                                        inputs=[custom_audio, custom_text],
                                        label="Ví dụ mẫu để thử nghiệm clone giọng"
                                    )
                                    
                                    gr.Markdown("""
                                    **💡 Mẹo nhỏ:** Nếu kết quả Zero-shot Voice Cloning chưa như ý, bạn hãy cân nhắc **Finetune (LoRA)** để đạt chất lượng tốt nhất. 
                                    Hướng dẫn chi tiết có tại file: `finetune/README.md` hoặc xem trên [GitHub](https://github.com/pnnbao97/VieNeu-TTS/tree/main/finetune).
                                    """)
                        
                        generation_mode = gr.Radio(
                            ["Standard (Một lần)"],
                            value="Standard (Một lần)",
                            label="Chế độ sinh"
                        )
                        btn_generate = gr.Button("🎵 Bắt đầu", variant="primary", scale=2, interactive=False)

                    # --- TAB 2: MULTI-SPEAKER CONVERSATION ---
                    with gr.Tab("🎭 Hội thoại", id="conv_tab", visible=False) as conv_tab:
                        conv_script_input = gr.Textbox(
                            label="Kịch bản hội thoại",
                            placeholder="Phương: Chào mọi người, mình là Phương...",
                            lines=10,
                            elem_classes="script-box",
                            value='Phương: Chào mọi người, mình là Phương. Hôm nay team có một announcement cực lớn về VieNeu-TTS Version 2. Đồng hành cùng mình là anh Dũng và Hùng. Hi guys!\n\nDũng: Yo, chào cả nhà. Mình sẽ đi thẳng vào technical side của bản nâng cấp này để mọi người có cái nhìn deep hơn nhé.\n\nHùng: Chào mọi người. Thật sự V2 là một huge milestone. Nó phá vỡ rào cản của những công cụ đọc văn bản khô khan, hướng tới một sự natural communication đúng nghĩa.\n\nPhương: Correct! Và bất ngờ nhất là: nãy giờ mọi người đang nghe bản demo được tạo ra 100% bằng VieNeu-TTS V2 đấy. Tụi mình đều là sản phẩm của AI hết. Amazing, right?\n\nDũng: Đỉnh thật sự! Tiện đây Hùng share thêm về cái nội công bên trong của model này đi.\n\nHùng: Chắc chắn rồi. Model được train trên 10000 hours audio chất lượng cao, nên nó hỗ trợ code-switching Anh Việt cực mượt, tự nhiên như podcast. Đặc biệt, dự án này hoàn toàn open-source để cộng đồng cùng phát triển.\n\nDũng: Về hiệu năng thì khỏi bàn. Khi test trên GPU quốc dân RTX 3060, tốc độ sinh audio nhanh gấp 10 lần realtime. Và đừng lo, nếu bạn không có card đồ hỏa xịn, tụi mình có sẵn bản CPU version để ai cũng có thể tiếp cận được.\n\nPhương: Tốc độ cực nhanh, hỗ trợ đa nền tảng và hoàn toàn miễn phí. Mọi người hãy cùng trải nghiệm nhé!'
                        )
                        
                        with gr.Row():
                            btn_detect_speakers = gr.Button("🔍 Quét nhân vật", size="sm", variant="secondary")
                            silence_slider = gr.Slider(minimum=0, maximum=3, value=0.1, step=0.1, label="⏱️ Khoảng lặng (giây)")

                        gr.Markdown("### 🎭 Cấu hình giọng đọc")
                        gr.Markdown("*Nhấn **Quét nhân vật** để tự động phát hiện và ánh xạ giọng đọc. Tải model trước để có danh sách giọng.*")

                        # Pre-build MAX_SPEAKERS speaker slot rows
                        speaker_name_boxes = []
                        speaker_voice_dds  = []
                        speaker_slot_rows  = []

                        for _i in range(MAX_SPEAKERS):
                            # Mặc định cho 3 nhân vật đầu tiên theo yêu cầu
                            _default_name = ""
                            _default_voice = None
                            _row_visible = False
                            
                            if _i == 0:
                                _default_name = "Phương"
                                _default_voice = "Ly"
                                _row_visible = True
                            elif _i == 1:
                                _default_name = "Dũng"
                                _default_voice = "Binh"
                                _row_visible = True
                            elif _i == 2:
                                _default_name = "Hùng"
                                _default_voice = "Sơn"
                                _row_visible = True
                            elif _i < 2:
                                _default_name = f"Nhân vật {_i+1}"
                                _row_visible = True

                            with gr.Row(visible=_row_visible) as _row:
                                _name = gr.Textbox(
                                    value=_default_name,
                                    label="👤 Nhân vật",
                                    interactive=False,
                                    scale=1,
                                    min_width=120
                                )
                                _dd = gr.Dropdown(
                                    choices=PRESET_VOICES_CACHE,
                                    value=_default_voice,
                                    label="🎤 Giọng đọc",
                                    interactive=True,
                                    scale=3,
                                    allow_custom_value=True
                                )
                            speaker_slot_rows.append(_row)
                            speaker_name_boxes.append(_name)
                            speaker_voice_dds.append(_dd)
                        
                        btn_generate_conv = gr.Button("🎭 Bắt đầu hội thoại", variant="primary", interactive=False)

                # Global Generation Settings
                with gr.Row():
                    use_batch = gr.Checkbox(
                        value=False, 
                        label="⚡ Batch Processing",
                        info="Xử lý nhiều đoạn cùng lúc (chỉ áp dụng khi sử dụng GPU và đã cài đặt LMDeploy)"
                    )
                    max_batch_size_run = gr.Slider(
                        minimum=1, 
                        maximum=16, 
                        value=1, 
                        step=1, 
                        label="📊 Batch Size (Generation)",
                        info="Số lượng đoạn văn bản xử lý cùng lúc. Giá trị cao = nhanh hơn nhưng tốn VRAM hơn. Giảm xuống nếu gặp lỗi Out of Memory."
                    )
                
                with gr.Accordion("⚙️ Cài đặt nâng cao (Generation)", open=False):
                    with gr.Row():
                        temperature_slider = gr.Slider(
                            minimum=0.1, maximum=1.5, value=default_temp, step=0.1,
                            label="🌡️ Temperature", 
                            info="Độ sáng tạo. Cao = đa dạng cảm xúc hơn nhưng dễ lỗi. Thấp = ổn định hơn."
                        )
                        max_chars_chunk_slider = gr.Slider(
                            minimum=128, maximum=512, value=256, step=32,
                            label="📝 Max Chars per Chunk",
                            info="Độ dài tối đa mỗi đoạn xử lý."
                        )
                    with gr.Row():
                        speaking_rate_slider = gr.Slider(
                            minimum=0.85, maximum=1.2, value=1.0, step=0.05,
                            label="Toc do doc",
                            info="1.0 = mac dinh. Nen dung 1.0, 1.1 hoac 1.2."
                        )
                        pitch_steps_slider = gr.Slider(
                            minimum=-2.0, maximum=2.0, value=0.0, step=1.0,
                            label="Do cao giong",
                            info="Tang giam tong giong nhe. Nen giu trong khoang -1 den 1."
                        )
                    pronunciation_rules = gr.Textbox(
                        label="🗣️ Từ điển sửa phát âm",
                        lines=4,
                        value=load_pronunciation_rules(),
                        info="Moi dong mot luat: chu goc => cach doc. Neu model doc kieu tieng Anh, hay dung dang phien am tieng Viet. Vi du: TTS => ti tít"
                    )
                    with gr.Row():
                        btn_save_rules = gr.Button("Luu tu dien", variant="secondary", scale=0)
                        btn_reload_rules = gr.Button("Nap da luu", variant="secondary", scale=0)
                        btn_reset_rules = gr.Button("Mac dinh", variant="secondary", scale=0)
                    pronunciation_rules_status = gr.Markdown("")
                
                # State to track current mode
                current_mode_state = gr.State("preset_mode")
                
                with gr.Row():
                    btn_stop = gr.Button("⏹️ Dừng", variant="stop", scale=1, interactive=False)
            
            # --- OUTPUT ---
            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="Kết quả",
                    type="filepath",
                    autoplay=True
                )
                status_output = gr.Textbox(
                    label="Trạng thái", 
                    elem_classes="status-box",
                    lines=2,
                    max_lines=10,
                    show_copy_button=True
                )
                gr.Markdown("<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>🔒 Audio được đóng dấu bản quyền ẩn (Watermarker) để bảo mật và định danh AI.</div>")
        
        # # --- EVENT HANDLERS ---
        # def update_info(backbone: str) -> str:
        #     return f"Streaming: {'✅' if BACKBONE_CONFIGS[backbone]['supports_streaming'] else '❌'}"
        
        # backbone_select.change(update_info, backbone_select, model_status)
        
        # Handler to show/hide Voice Cloning tab
        def on_codec_change(codec: str, current_mode: str):
            is_onnx = "onnx" in codec.lower()
            # If switching to ONNX and we are on custom mode, switch back to preset
            if is_onnx and current_mode == "custom_mode":
                return gr.update(visible=False), gr.update(selected="preset_mode"), "preset_mode"
            return gr.update(visible=not is_onnx), gr.update(), current_mode
        
        codec_select.change(
            on_codec_change, 
            inputs=[codec_select, current_mode_state], 
            outputs=[tab_custom, tabs, current_mode_state]
        )
        codec_select.change(
            fn=maybe_auto_load_selected_model,
            inputs=[backbone_select, codec_select, device_choice, use_lmdeploy_cb,
                    custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token],
            outputs=[model_status, btn_generate, btn_generate_conv, btn_load, btn_stop, voice_select,
                     tab_preset, tab_custom, tabs, current_mode_state,
                     conv_tab,
                     *speaker_voice_dds]
        )
        
        # Bind tab events to update state
        tab_preset.select(lambda: "preset_mode", outputs=current_mode_state)
        tab_custom.select(lambda: "custom_mode", outputs=current_mode_state)
        
        def validate_audio_duration(audio_path):
            if not audio_path:
                return gr.update(visible=False)
            try:
                info = sf.info(audio_path)
                if info.duration > 5.1:
                    return gr.update(
                        value=f"⚠️ **Cảnh báo:** Audio mẫu hiện tại dài {info.duration:.1f} giây. Để có kết quả clone giọng tối ưu, bạn nên sử dụng đoạn audio có độ dài lý tưởng từ **3 đến 5 giây**.",
                        visible=True
                    )
            except Exception:
                pass
            return gr.update(visible=False)

        custom_audio.change(validate_audio_duration, inputs=[custom_audio], outputs=[cloning_warning_msg])
        
        # --- Custom Model Event Handlers ---

        def on_backbone_change(choice):
            is_custom = (choice == "Custom Model")
            print(f"   🔄 Backbone changed to: {choice}")
            
            # 1. Device logic
            # Allow hardware acceleration (MPS/CUDA/Auto) for all GPU models AND Turbo (GGUF) models
            is_hw_accel_supported = "(GPU)" in choice or "v2-Turbo" in choice or is_custom
            
            if is_hw_accel_supported:
                dev_choices = get_available_devices()
                initial_dev = "Auto"
            else:
                dev_choices = ["CPU"]
                initial_dev = "CPU"
            
            # 2. Parameter logic
            if "Turbo" in choice:
                codec_update = gr.update(value="VieNeu-Codec", interactive=False)
                text_update = gr.update(value=DEFAULT_TEXT_TURBO)
                temp_update = gr.update(value=0.4)
            elif "(CPU)" in choice:
                codec_update = gr.update(value="NeuCodec (ONNX)", interactive=True)
                text_update = gr.update(value=DEFAULT_TEXT_GPU)
                temp_update = gr.update(value=0.7)
            else:
                codec_update = gr.update(value="NeuCodec (Distill)", interactive=True)
                text_update = gr.update(value=DEFAULT_TEXT_GPU)
                temp_update = gr.update(value=0.7)
                
            return (
                gr.update(visible=is_custom), 
                codec_update, 
                text_update, 
                temp_update, 
                gr.update(choices=dev_choices, value=initial_dev),
                gr.update(visible=True)
            )

        backbone_change_event = backbone_select.change(
            on_backbone_change,
            inputs=[backbone_select],
            outputs=[
                custom_model_group, 
                codec_select, 
                text_input, 
                temperature_slider, 
                device_choice,
                cloning_elements_group
            ]
        )
        backbone_change_event.then(
            fn=maybe_auto_load_selected_model,
            inputs=[backbone_select, codec_select, device_choice, use_lmdeploy_cb,
                    custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token],
            outputs=[model_status, btn_generate, btn_generate_conv, btn_load, btn_stop, voice_select,
                     tab_preset, tab_custom, tabs, current_mode_state,
                     conv_tab,
                     *speaker_voice_dds]
        )
        
        def on_custom_id_change(model_id):
            # Auto detect LoRA and base model
            if model_id and "lora" in model_id.lower():
                # Detect base model
                if "0.3" in model_id:
                    base_model = "VieNeu-TTS-0.3B (GPU)"
                else:
                    base_model = "VieNeu-TTS (GPU)"
                
                return (
                    gr.update(visible=True, value=base_model),
                    gr.update(), gr.update()
                )
            
            return (
                gr.update(visible=False),
                gr.update(),
                gr.update()
            )
            
        custom_backbone_model_id.change(
            on_custom_id_change,
            inputs=[custom_backbone_model_id],
            outputs=[custom_backbone_base_model, custom_audio, custom_text]
        )
        custom_backbone_model_id.change(
            fn=maybe_auto_load_selected_model,
            inputs=[backbone_select, codec_select, device_choice, use_lmdeploy_cb,
                    custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token],
            outputs=[model_status, btn_generate, btn_generate_conv, btn_load, btn_stop, voice_select,
                     tab_preset, tab_custom, tabs, current_mode_state,
                     conv_tab,
                     *speaker_voice_dds]
        )
        custom_backbone_base_model.change(
            fn=maybe_auto_load_selected_model,
            inputs=[backbone_select, codec_select, device_choice, use_lmdeploy_cb,
                    custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token],
            outputs=[model_status, btn_generate, btn_generate_conv, btn_load, btn_stop, voice_select,
                     tab_preset, tab_custom, tabs, current_mode_state,
                     conv_tab,
                     *speaker_voice_dds]
        )
        device_choice.change(
            fn=maybe_auto_load_selected_model,
            inputs=[backbone_select, codec_select, device_choice, use_lmdeploy_cb,
                    custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token],
            outputs=[model_status, btn_generate, btn_generate_conv, btn_load, btn_stop, voice_select,
                     tab_preset, tab_custom, tabs, current_mode_state,
                     conv_tab,
                     *speaker_voice_dds]
        )
        use_lmdeploy_cb.change(
            fn=maybe_auto_load_selected_model,
            inputs=[backbone_select, codec_select, device_choice, use_lmdeploy_cb,
                    custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token],
            outputs=[model_status, btn_generate, btn_generate_conv, btn_load, btn_stop, voice_select,
                     tab_preset, tab_custom, tabs, current_mode_state,
                     conv_tab,
                     *speaker_voice_dds]
        )

        btn_load.click(
            fn=load_model,
            inputs=[backbone_select, codec_select, device_choice, use_lmdeploy_cb,
                    custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token],
            outputs=[model_status, btn_generate, btn_generate_conv, btn_load, btn_stop, voice_select,
                     tab_preset, tab_custom, tabs, current_mode_state,
                     conv_tab,
                     *speaker_voice_dds]
        )

        btn_save_voice.click(
            fn=save_current_clone_voice,
            inputs=[saved_voice_name, saved_voice_desc, custom_audio, custom_text],
            outputs=[voice_select, save_voice_status]
        )

        btn_save_rules.click(
            fn=save_pronunciation_rules,
            inputs=[pronunciation_rules],
            outputs=[pronunciation_rules_status]
        )
        btn_reload_rules.click(
            fn=lambda: (load_pronunciation_rules(), "Da nap tu dien da luu."),
            outputs=[pronunciation_rules, pronunciation_rules_status]
        )
        btn_reset_rules.click(
            fn=reset_pronunciation_rules,
            outputs=[pronunciation_rules, pronunciation_rules_status]
        )
        
        # --- Conversation Event Handlers ---
        # Scan speakers → update all 8 slot rows/names/dropdowns
        btn_detect_speakers.click(
            fn=extract_speakers_from_script,
            inputs=[conv_script_input],
            outputs=speaker_name_boxes + speaker_voice_dds + speaker_slot_rows
        )
        
        conv_gen_event = btn_generate_conv.click(
            fn=synthesize_conversation,
            inputs=[conv_script_input,
                    *speaker_name_boxes,
                    *speaker_voice_dds,
                    silence_slider, temperature_slider, max_chars_chunk_slider,
                    session_id_state],
            outputs=[audio_output, status_output]
        )
        btn_generate_conv.click(lambda: gr.update(interactive=True), outputs=btn_stop)
        conv_gen_event.then(lambda: gr.update(interactive=False), outputs=btn_stop)

        # --- Auto-adjust Temperature on Tab Switch ---
        conv_tab.select(
            fn=lambda: gr.update(value=1.0),
            outputs=temperature_slider
        )
        single_tab.select(
            fn=lambda: gr.update(value=default_temp),
            outputs=temperature_slider
        )
        
        # --- Standard Generation Handlers ---
        gen_event = btn_generate.click(
            fn=synthesize_speech,
            inputs=[text_input, voice_select, custom_audio, custom_text, current_mode_state, 
                    generation_mode, use_batch, max_batch_size_run,
                    temperature_slider, max_chars_chunk_slider, pronunciation_rules, speaking_rate_slider, pitch_steps_slider, session_id_state],
            outputs=[audio_output, status_output]
        )
        btn_generate.click(lambda: gr.update(interactive=True), outputs=btn_stop)
        gen_event.then(lambda: gr.update(interactive=False), outputs=btn_stop)

        # --- Stop Button ---
        def request_stop():
            print("🛑 STOP REQUESTED via button click.")
            _STOP_EVENT.set()
            return None, "⏹️ Đã dừng tạo giọng nói.", gr.update(interactive=False)

        # Handler: set stop event + update UI
        # Note: We avoid cancels= here to prevent internal Gradio KeyError crashes,
        # relying instead on the frequent _STOP_EVENT.is_set() checks in the code.
        btn_stop.click(fn=request_stop, outputs=[audio_output, status_output, btn_stop])

        # Startup: auto-load cached model if available, otherwise enable manual load
        demo.load(
            fn=maybe_auto_load_selected_model,
            inputs=[backbone_select, codec_select, device_choice, use_lmdeploy_cb,
                    custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token],
            outputs=[model_status, btn_generate, btn_generate_conv, btn_load, btn_stop, voice_select,
                     tab_preset, tab_custom, tabs, current_mode_state,
                     conv_tab,
                     *speaker_voice_dds]
        )

def main():
    # Cho phép override từ biến môi trường (hữu ích cho Docker)
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    # Check running in Colab
    is_on_colab = os.getenv("COLAB_RELEASE_TAG") is not None

    # Default:
    # - Colab: share=True (convenient)
    # - Docker/local: share=False (safe)
    share = env_bool("GRADIO_SHARE", default=is_on_colab)
    
    # If server_name is "0.0.0.0" and GRADIO_SHARE is not set, disable sharing
    if server_name == "0.0.0.0" and os.getenv("GRADIO_SHARE") is None:
        share = False

    demo.queue().launch(server_name=server_name, server_port=server_port, share=share)

if __name__ == "__main__":
    main()
