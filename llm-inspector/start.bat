@echo off
echo ====================================
echo LLM Inspector v7.0
echo ====================================

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
    echo [OK] Virtual environment activated
) else (
    echo [Warning] .venv not found, using system Python
)

REM Enable asyncio-native concurrent pipeline (Phase P3)
REM Set to 0 to fall back to ThreadPoolExecutor mode
SET ASYNCIO_MODE=1

REM Change to backend directory
cd backend

REM Check and install dependencies
echo.
echo ====================================
echo Checking Dependencies...
echo ====================================
python scripts\setup_dependencies.py --skip-optional
if %errorlevel% neq 0 (
    echo [WARN] Some optional dependencies failed to install
    echo [INFO] Continuing with core functionality...
)

REM Start server using Python startup script
echo.
echo ====================================
echo Starting Server...
echo ====================================
python start.py --port 8000

pause
