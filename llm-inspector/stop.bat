@echo off
setlocal

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "PORT=8000"
set "PID_FILE=%ROOT%\.inspector.pid"

echo ====================================
echo Stopping LLM Inspector (BAT only)...
echo ====================================

if exist "%PID_FILE%" (
  for /f %%P in (%PID_FILE%) do (
    echo [INFO] Killing pid from file: %%P
    taskkill /PID %%P /F >nul 2>nul
  )
  del /f /q "%PID_FILE%" >nul 2>nul
)

for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
  echo [INFO] Killing pid on port %PORT%: %%P
  taskkill /PID %%P /F >nul 2>nul
)

echo [OK] Stop completed.
pause
endlocal
