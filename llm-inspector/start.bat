@echo off
echo ====================================
echo Starting LLM Inspector...
echo ====================================

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo [Warning] .venv not found.
)

REM Enable asyncio-native concurrent pipeline (Phase P3)
REM Set to 0 to fall back to ThreadPoolExecutor mode
SET ASYNCIO_MODE=1

REM Change to backend directory and start server
cd backend
echo Serving on http://localhost:8000
python -m app.main

pause
