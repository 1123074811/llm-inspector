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

REM Change to backend directory and start server
cd backend
echo Serving on http://localhost:8000
python -m app.main

pause
