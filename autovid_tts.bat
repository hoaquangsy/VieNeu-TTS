@echo off
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" "autovid_tts_wrapper.py" "%~1" "%~2"
exit /b %ERRORLEVEL%
