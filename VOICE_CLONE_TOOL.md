# VieNeu-TTS Voice Clone Tool

This repo now includes a focused CLI for Vietnamese voice cloning.

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
.\run_voice_clone.bat --ref-audio examples\audio_ref\example.wav --text "Xin chào, đây là giọng nói được clone bằng VieNeu-TTS." --output outputs\clone.wav
```

## Standard Mode

Standard mode requires the exact transcript of the reference audio:

```powershell
.\run_voice_clone.bat --mode standard --ref-audio examples\audio_ref\example.wav --ref-text-file examples\audio_ref\example.txt --text "Đây là bản clone bằng standard mode." --output outputs\standard_clone.wav
```

## Preset Voices

List built-in voices:

```powershell
.\run_voice_clone.bat --list-voices
```

Use a built-in voice:

```powershell
.\run_voice_clone.bat --preset-voice "Ly" --text "Xin chào, tôi đang dùng giọng có sẵn." --output outputs\preset.wav
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
