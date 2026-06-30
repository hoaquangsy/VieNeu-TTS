@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
title VieNeu Voice API LAN

set "PY_EXE=%~dp0.venv\Scripts\python.exe"
set "API_PORT=8002"

if not exist "%PY_EXE%" (
  echo [ERROR] Khong tim thay virtual environment: %PY_EXE%
  echo Hay setup moi truong truoc khi chay API.
  pause
  exit /b 1
)

for /f "usebackq tokens=*" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "(Get-NetIPAddress -AddressFamily IPv4 ^| Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } ^| Select-Object -First 1 -ExpandProperty IPAddress)"`) do set "LAN_IP=%%I"

if "%LAN_IP%"=="" set "LAN_IP=<PC-LAN-IP>"

echo VieNeu Voice API LAN
echo --------------------
echo Local : http://127.0.0.1:%API_PORT%
echo LAN   : http://%LAN_IP%:%API_PORT%
echo Health: http://%LAN_IP%:%API_PORT%/health
echo.
echo Neu laptop khong goi duoc, mo Windows Firewall cho TCP port %API_PORT%.
echo.

"%PY_EXE%" -m uvicorn apps.voice_clone_api:app --host 0.0.0.0 --port %API_PORT%
