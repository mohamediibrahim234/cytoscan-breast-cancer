@echo off
setlocal
cd /d "%~dp0"
title CytoScan - Breast Cancer Diagnosis API

rem ---- 1. create the project's own environment on first run ----
if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating .venv ...
    where uv >nul 2>nul && (uv venv .venv --python 3.12) || (py -3.12 -m venv .venv || python -m venv .venv)
    if not exist ".venv\Scripts\python.exe" (
        echo [error] Could not create a Python environment. Install Python 3.12 or uv first.
        pause & exit /b 1
    )
    echo [setup] Installing requirements ...
    where uv >nul 2>nul && (uv pip install --python .venv\Scripts\python.exe -r requirements.txt) || (.venv\Scripts\python.exe -m pip install -r requirements.txt)
)
set "PY=.venv\Scripts\python.exe"

rem ---- 2. pick a port: %PORT% or 8000, moving up if it is already taken ----
if "%PORT%"=="" set "PORT=8000"
for /f %%p in ('%PY% -c "import socket;s=%PORT%;print(next(p for p in range(s,s+50) if socket.socket().connect_ex(('127.0.0.1',p))!=0))"') do set "PORT=%%p"
set "URL=http://127.0.0.1:%PORT%"

rem ---- 3. open the browser once /health answers (runs in the background) ----
start "" /b powershell -NoProfile -WindowStyle Hidden -Command ^
  "for($i=0;$i -lt 300;$i++){try{Invoke-RestMethod '%URL%/health' -TimeoutSec 2 | Out-Null; Start-Process '%URL%/'; break}catch{Start-Sleep -Milliseconds 500}}"

echo.
echo  Starting server on %URL%   (docs: %URL%/docs)
echo  The browser opens automatically when the model is loaded. Close this window to stop.
echo.
%PY% -m uvicorn app.main:app --host 127.0.0.1 --port %PORT%
pause
