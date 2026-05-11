@echo off
setlocal
cd /d "%~dp0"
set "UV_EXE=uv"
where uv >nul 2>nul
if errorlevel 1 (
  if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
  ) else (
    echo uv is not installed or is not in PATH.
    echo Install it with:
    echo powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    exit /b 1
  )
)
"%UV_EXE%" run python apps\voice_clone_api.py
