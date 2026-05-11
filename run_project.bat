@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
title VieNeu-TTS Launcher

set "PY_EXE=%~dp0.venv\Scripts\python.exe"
set "GUI_PORT=7860"
set "API_PORT=8002"
set "GUI_URL=http://127.0.0.1:%GUI_PORT%"
set "API_URL=http://127.0.0.1:%API_PORT%"

if not exist "%PY_EXE%" (
  cls
  echo ========================================
  echo          VieNeu-TTS Launcher
  echo ========================================
  echo.
  echo [ERROR] Khong tim thay virtual environment.
  echo Hay chay: uv sync
  echo.
  pause
  exit /b 1
)

cls
echo ========================================
echo          VieNeu-TTS Launcher
echo ========================================
echo.
echo Workspace: %~dp0
echo.

call :ensure_service "GUI goc" "%GUI_PORT%" "apps\run_original_ui.py"
call :ensure_service "Voice Clone API" "%API_PORT%" "apps\voice_clone_api.py"

echo.
echo ----------------------------------------
echo San sang
echo ----------------------------------------
echo GUI : %GUI_URL%
echo API : %API_URL%
echo.

start "" "%GUI_URL%"
echo Da mo giao dien trong trinh duyet mac dinh.
echo.
echo Nhan phim bat ky de dong cua so nay.
pause >nul
exit /b 0

:ensure_service
set "SERVICE_NAME=%~1"
set "SERVICE_PORT=%~2"
set "SERVICE_SCRIPT=%~3"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$conns = @(Get-NetTCPConnection -LocalPort %SERVICE_PORT% -State Listen -ErrorAction SilentlyContinue); foreach ($conn in $conns) { try { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue } catch {} }; if ($conns.Count -gt 0) { Start-Sleep -Seconds 1 }; exit 20" >nul 2>nul

start "" /min "%PY_EXE%" "%SERVICE_SCRIPT%"
echo [START] %SERVICE_NAME% -> cong %SERVICE_PORT%
goto :eof
