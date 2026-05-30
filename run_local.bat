@echo off
REM ─────────────────────────────────────────────────────────────────
REM  run_local.bat  —  Start API + Bot together for local development
REM  FastAPI  → http://localhost:8000
REM  Bot      → polls Telegram
REM ─────────────────────────────────────────────────────────────────

set SCRIPT_DIR=%~dp0

echo [INFO] Starting FastAPI server on http://localhost:8000 ...
start "FastAPI - OCR API" cmd /k "cd /d %SCRIPT_DIR% && .venv\Scripts\activate && uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

REM Give the API a moment to bind before the bot tries to call it
timeout /t 3 /nobreak >nul

echo [INFO] Starting Telegram bot ...
start "Telegram Bot" cmd /k "cd /d %SCRIPT_DIR% && .venv\Scripts\activate && python initbot.py"

echo.
echo ✅  Both services launched in separate windows.
echo    FastAPI docs: http://localhost:8000/docs
echo    Press any key to exit this launcher window.
pause >nul
