# =============================================================================
# LLM Inspector - Windows One-Click Startup Script (PowerShell)
# Usage: .\setup.ps1 [-Port 8000] [-NoBrowser] [-Stop]
# =============================================================================

param(
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$Stop,
    [switch]$NoPause
)

# Allow script execution (if needed) - ignore errors if policy is already set
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force -ErrorAction SilentlyContinue 2>$null

$ErrorActionPreference = "Stop"

# Helper function for prompts
function Test-Interactive {
    return -not $NoPause
}

# ── Color Functions ──────────────────────────────────────────────────────────
function Write-Info  { param($msg) Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Write-Ok    { param($msg) Write-Host "[OK]    $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Err   { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }
function Write-Header { param($msg) Write-Host "`n== $msg ==" -ForegroundColor Blue }

# ── Path Definitions ─────────────────────────────────────────────────────────
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $ScriptDir "backend"
$VenvDir    = Join-Path $ScriptDir ".venv"
$PidFile    = Join-Path $ScriptDir ".inspector.pid"
$LogFile    = Join-Path $ScriptDir "inspector.log"
$EnvFile    = Join-Path $ScriptDir ".env"
$EnvExample = Join-Path $ScriptDir ".env.example"

# ── Stop Mode ────────────────────────────────────────────────────────────────
if ($Stop) {
    if (Test-Path $PidFile) {
        $OldPid = Get-Content $PidFile -ErrorAction SilentlyContinue
        if ($OldPid) {
            try {
                $proc = Get-Process -Id $OldPid -ErrorAction Stop
                Write-Warn "Stopping LLM Inspector (PID=$OldPid)..."
                Stop-Process -Id $OldPid -Force
                Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
                Write-Ok "Service stopped"
            } catch {
                Write-Warn "Process $OldPid does not exist, may have already stopped"
                Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
            }
        }
    } else {
        # Try to find by port
        $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if ($conn) {
            $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
            if ($proc) {
                Write-Warn "Found process on port $Port (PID=$($proc.Id)), stopping..."
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                Write-Ok "Service stopped"
            }
        } else {
            Write-Info "No running LLM Inspector instance found"
        }
    }
    exit 0
}

# ── Banner ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  LLM Inspector  v1.0" -ForegroundColor Cyan
Write-Host "  LLM API Behavior Detection & Identity Tool" -ForegroundColor Cyan
Write-Host ""

# ── Check for Existing Instance ──────────────────────────────────────────────
if (Test-Path $PidFile) {
    $OldPid = Get-Content $PidFile -ErrorAction SilentlyContinue
    if ($OldPid) {
        $proc = Get-Process -Id $OldPid -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Warn "Detected running instance (PID=$OldPid), stopping..."
            Stop-Process -Id $OldPid -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 1
        }
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

# ── Step 1: Check Python ─────────────────────────────────────────────────────
Write-Header "Step 1 / 4  Checking Python Environment"

$PythonCmd = $null
$PythonCandidates = @("python", "python3", "py")

foreach ($cmd in $PythonCandidates) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -eq 3 -and $minor -ge 10) {
                $PythonCmd = $cmd
                Write-Ok "Found Python $major.$minor ($cmd)"
                break
            }
        }
    } catch { continue }
}

if (-not $PythonCmd) {
    Write-Err "Python 3.10+ not found. Please install Python first."
    Write-Host ""
    Write-Host "  Official download: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "  winget:   winget install Python.Python.3.12" -ForegroundColor Yellow
    Write-Host "  Note: Check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    Write-Host ""
    if (Test-Interactive) { Read-Host "Press Enter to exit" }
    exit 1
}

# ── Step 2: Create Virtual Environment and Install Dependencies ──────────────
Write-Header "Step 2 / 4  Installing Dependencies"

if (-not (Test-Path $VenvDir)) {
    Write-Info "Creating virtual environment..."
    & $PythonCmd -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to create virtual environment"
        if (Test-Interactive) { Read-Host "Press Enter to exit" }
        exit 1
    }
    Write-Ok "Virtual environment created: $VenvDir"
} else {
    Write-Ok "Virtual environment already exists, skipping creation"
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip    = Join-Path $VenvDir "Scripts\pip.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Err "Virtual environment is corrupted. Please delete .venv directory and retry."
    if (Test-Interactive) { Read-Host "Press Enter to exit" }
    exit 1
}

# Upgrade pip - suppress warnings and errors
try {
    & $VenvPip install --upgrade pip --quiet 2>&1 | Out-Null
} catch {
    Write-Warn "Failed to upgrade pip, continuing with existing version"
}

# Install required packages
Write-Info "Checking and installing dependencies..."
$packages = @("cryptography>=41", "numpy>=1.24", "scikit-learn>=1.3")
$needInstall = @()

# Get list of installed packages
$installedPackages = & $VenvPip list --format=freeze 2>$1 | Out-String

foreach ($pkg in $packages) {
    $pkgName = ($pkg -split "[>=<]")[0]
    # Check if package is in the installed list
    if ($installedPackages -notmatch "^$pkgName==") {
        $needInstall += $pkg
    }
}

if ($needInstall.Count -gt 0) {
    Write-Info "Installing: $($needInstall -join ', ')"
    & $VenvPip install $needInstall
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Dependency installation failed. Please check network connection."
        if (Test-Interactive) { Read-Host "Press Enter to exit" }
        exit 1
    }
    Write-Ok "Dependencies installed"
} else {
    Write-Ok "All dependencies satisfied, no installation needed"
}

# ── Step 3: Initialize Configuration ─────────────────────────────────────────
Write-Header "Step 3 / 4  Initializing Configuration"

if (-not (Test-Path $EnvFile)) {
    if (Test-Path $EnvExample) {
        Copy-Item $EnvExample $EnvFile
    } else {
        # Create minimal .env if example is missing
        @"
APP_ENV=development
HOST=0.0.0.0
PORT=$Port
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:$Port
DATABASE_URL=sqlite:///./llm_inspector.db
ENCRYPTION_KEY=
API_KEY_TTL_HOURS=72
DEFAULT_REQUEST_TIMEOUT_SEC=60
MAX_STREAM_CHUNKS=512
INTER_REQUEST_DELAY_MS=500
RAW_RESPONSE_TTL_DAYS=7
STREAM_CHUNKS_TTL_DAYS=3
USE_CELERY=false
PREDETECT_CONFIDENCE_THRESHOLD=0.85
"@ | Set-Content $EnvFile -Encoding UTF8
    }

    # Auto-generate encryption key
    $encKey = & $VenvPython -c "import secrets,base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"
    (Get-Content $EnvFile) -replace "^ENCRYPTION_KEY=.*", "ENCRYPTION_KEY=$encKey" | Set-Content $EnvFile -Encoding UTF8
    # Set port
    (Get-Content $EnvFile) -replace "^PORT=.*", "PORT=$Port" | Set-Content $EnvFile -Encoding UTF8
    Write-Ok ".env configuration file created (encryption key auto-generated)"
} else {
    Write-Ok ".env configuration file already exists"
    # Update port
    (Get-Content $EnvFile) -replace "^PORT=.*", "PORT=$Port" | Set-Content $EnvFile -Encoding UTF8
}

# ── Step 4: Start Service ────────────────────────────────────────────────────
Write-Header "Step 4 / 4  Starting Service"

# Self-test
Write-Info "Running self-test..."
$selfTest = & $VenvPython -c @"
import sys
sys.path.insert(0, r'$BackendDir')
from app.core.db import init_db
from app.tasks.seeder import seed_all
from app.core.security import get_key_manager
init_db()
seed_all()
km = get_key_manager()
enc, h = km.encrypt('test')
assert km.decrypt(enc) == 'test'
print('SELFTEST_OK')
"@ 2>&1

if ($selfTest -notmatch "SELFTEST_OK") {
    Write-Err "Self-test failed. Output:"
    Write-Host $selfTest
    if (Test-Interactive) { Read-Host "Press Enter to exit" }
    exit 1
}
Write-Ok "Self-test passed"

# Start server as background process
Write-Info "Starting server (port $Port)..."

$envVars = @{
    "PORT"        = "$Port"
    "PYTHONPATH"  = $BackendDir
}

$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName               = $VenvPython
$startInfo.Arguments              = "app/main.py"
$startInfo.WorkingDirectory       = $BackendDir
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError  = $true
$startInfo.UseShellExecute        = $false
$startInfo.CreateNoWindow         = $true

# Pass environment variables
foreach ($kv in $envVars.GetEnumerator()) {
    $startInfo.Environment[$kv.Key] = $kv.Value
}
# Inherit current PATH so Python can find DLLs
$startInfo.Environment["PATH"] = $env:PATH

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $startInfo

# Async log capture
$logContent = [System.Collections.Concurrent.ConcurrentQueue[string]]::new()
$stdoutJob = Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action {
    if ($EventArgs.Data) {
        Add-Content -Path $using:LogFile -Value $EventArgs.Data -Encoding UTF8
    }
}
$stderrJob = Register-ObjectEvent -InputObject $process -EventName ErrorDataReceived -Action {
    if ($EventArgs.Data) {
        Add-Content -Path $using:LogFile -Value $EventArgs.Data -Encoding UTF8
    }
}

$null = $process.Start()
$process.BeginOutputReadLine()
$process.BeginErrorReadLine()

$ServerPid = $process.Id
$ServerPid | Set-Content $PidFile -Encoding UTF8

# Wait for server ready
Write-Info "Waiting for service to be ready..."
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:$Port/api/v1/health" `
                                  -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch { continue }
}

if (-not $ready) {
    Write-Err "Server startup timeout. Please check logs: $LogFile"
    if (Test-Path $LogFile) {
        Write-Host (Get-Content $LogFile -Tail 20 | Out-String)
    }
    $process.Kill()
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    if (Test-Interactive) { Read-Host "Press Enter to exit" }
    exit 1
}

# ── Startup Successful ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║         LLM Inspector Started Successfully!      ║" -ForegroundColor Green
Write-Host "╠══════════════════════════════════════════════════╣" -ForegroundColor Green
Write-Host "║  URL:         http://localhost:$Port" -ForegroundColor Green
Write-Host "║  PID:         $ServerPid" -ForegroundColor Green
Write-Host "║  Log File:    $LogFile" -ForegroundColor Green
Write-Host "║" -ForegroundColor Green
Write-Host "║  Stop Service: .\setup.ps1 -Stop" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

if (-not $NoBrowser) {
    Start-Process "http://localhost:$Port"
}

Write-Host "Service is running in background. Closing this window will not stop the service." -ForegroundColor Cyan
Write-Host "Use '.\setup.ps1 -Stop' to stop the service." -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to close this window (service continues running)"
