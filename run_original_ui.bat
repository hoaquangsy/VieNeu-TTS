@echo off
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set "PY_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PY_EXE%" (
  echo Python virtual environment not found. Run uv sync first.
  exit /b 1
)
"%PY_EXE%" apps\run_original_ui.py
