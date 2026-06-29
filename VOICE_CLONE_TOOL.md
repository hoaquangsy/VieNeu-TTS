# VieNeu-TTS Voice Clone Tool

This repo includes a focused CLI, local HTTP API, and AutoVid bridge for Vietnamese voice cloning.

## Setup

```powershell
uv sync
```

On Windows CPU, if `llama-cpp-python` install fails, use the project wheel index from the README:

```powershell
uv pip install llama-cpp-python==0.3.16 --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/
```

## Clone a Voice

Turbo mode is the default and does not require a transcript for the reference audio:

```powershell
.\run_voice_clone.bat --ref-audio examples\audio_ref\example.wav --text "Xin chao, day la giong noi duoc clone bang VieNeu-TTS." --output outputs\clone.wav
```

For best Vietnamese quality, provide UTF-8 text:

```powershell
.\run_voice_clone.bat --ref-audio examples\audio_ref\example.wav --text "Xin chao, day la giong noi duoc clone bang VieNeu-TTS." --output outputs\clone.wav
```

## Standard Mode

Standard mode requires the exact transcript of the reference audio:

```powershell
.\run_voice_clone.bat --mode standard --ref-audio examples\audio_ref\example.wav --ref-text-file examples\audio_ref\example.txt --text "Day la ban clone bang standard mode." --output outputs\standard_clone.wav
```

## Preset Voices

List built-in voices:

```powershell
.\run_voice_clone.bat --list-voices
```

Use a built-in voice:

```powershell
.\run_voice_clone.bat --preset-voice "Ly" --text "Xin chao, toi dang dung giong co san." --output outputs\preset.wav
```

## Direct Python Entrypoint

After setup, the same tool is available as:

```powershell
uv run vieneu-clone --help
```

## Web UI

Run the focused voice clone interface:

```powershell
.\run_voice_clone_ui.bat
```

Open:

```text
http://127.0.0.1:7861
```

## Main Launcher

Launcher chinh cho workspace nay:

```powershell
.\run_project.bat
```

No se restart va mo:

```text
GUI goc:          http://127.0.0.1:7860
Voice Clone API: http://127.0.0.1:8002
```

Khi can chay truc tiep trong workspace, uu tien dung:

```powershell
.\.venv\Scripts\python.exe apps\run_original_ui.py
.\.venv\Scripts\python.exe apps\voice_clone_api.py
```

## AutoVid Bridge

AutoVid co the goi VieNeu theo 2 cach.

### 1. Legacy local command

File:

```text
D:\VieNeu-TTS\autovid_tts.bat
D:\VieNeu-TTS\autovid_tts_wrapper.py
```

Wrapper nhan:

```text
autovid_tts_wrapper.py <text-file> <output-wav>
```

Env vars ho tro:

```text
LOCAL_TTS_MODE              turbo | turbo_gpu | standard | remote
LOCAL_TTS_DEVICE            cpu | cuda | gpu
LOCAL_TTS_SPEED             1.0 | 1.1 | 1.2
LOCAL_TTS_EMOTION           natural | storytelling
LOCAL_TTS_PRESET_VOICE      preset voice id, vi du Ly, Binh, Son co dau neu API tra ve
LOCAL_TTS_PRESET_VOICE_B64  preset voice id dang base64 de giu Unicode
LOCAL_TTS_REF_AUDIO         duong dan audio tham chieu
LOCAL_TTS_REF_TEXT          transcript cho standard mode
LOCAL_TTS_REF_TEXT_FILE     file transcript cho standard mode
LOCAL_TTS_API_BASE          remote API base neu mode=remote
LOCAL_TTS_MODEL             remote model name neu mode=remote
```

### 2. Aligned full narration API

Duong dang dung cho AutoVid chat luong cao:

```http
POST http://127.0.0.1:8002/synthesize-aligned
```

AutoVid gui nhieu segment vao mot request. VieNeu tach sentence-aware chunks, tao mot full WAV duy nhat bang dung text goc, chay Whisper chi de lay timing, roi tra timeline.

Artifacts nam tai:

```text
D:\VieNeu-TTS\outputs\api\full_<request_id>.wav
D:\VieNeu-TTS\outputs\api\full_<request_id>.txt
D:\VieNeu-TTS\outputs\api\full_<request_id>.timings.json
```

Kiem tra nhanh VieNeu da nhan request AutoVid chua:

```powershell
Get-NetTCPConnection -LocalPort 8002 -ErrorAction SilentlyContinue
Get-ChildItem D:\VieNeu-TTS\outputs\api -File |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 10 Name,LastWriteTime,Length
```

Neu AutoVid dang chay nhung `outputs\api` khong co file moi va port `8002` khong co connection active, thi AutoVid dang o buoc khac, khong phai VieNeu TTS.

## Current Local Notes

- File `CODEX_RESTORED_CONTEXT.md` la noi bo, khong commit.
- API mac dinh chon `vieneu-v2-cpu` cho aligned TTS.
- `/synthesize-aligned` voi `full_then_align` la duong uu tien cho AutoVid; `full_then_align_verified` chi con la alias tuong thich nguoc.
- Case AutoVid gan nhat da pass voi request id `autovid_20260605154345_2fcf3c6080bf`; neu rerender tu `data.json`, AutoVid co the khong goi VieNeu nua.
