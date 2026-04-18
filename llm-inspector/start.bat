@echo off
echo ====================================
echo LLM Inspector v13.0
echo ====================================

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
    echo [OK] Virtual environment activated
) else (
    echo [Warning] .venv not found, using system Python
)

REM Enable asyncio-native concurrent pipeline
REM Set to 0 to fall back to ThreadPoolExecutor mode
SET ASYNCIO_MODE=1

REM Change to backend directory
cd backend

REM Check and install dependencies (idempotent: skips already-installed packages)
echo.
echo ====================================
echo Checking Dependencies...
echo ====================================
python -c "import yaml, numpy, scipy, cryptography" 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Installing missing core dependencies...
    python scripts\setup_dependencies.py --skip-optional
    if %errorlevel% neq 0 (
        echo [WARN] Some dependencies failed to install
        echo [INFO] Continuing with available functionality...
    )
) else (
    echo [OK] Core dependencies already satisfied
)

REM Verify SOURCES.yaml provenance registry
echo.
echo ====================================
echo Verifying Data Provenance...
echo ====================================
python start.py --verify-sources
if %errorlevel% neq 0 (
    echo [WARN] Provenance check found issues (non-fatal in dev mode)
)

REM Start server using Python startup script
echo.
echo ====================================
echo Starting Server...
echo ====================================
python start.py --port 8000

pause
