@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Chay python -m venv .venv truoc.
  exit /b 1
)
echo Mo http://localhost:8000 sau khi server khoi dong.
".venv\Scripts\python.exe" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
