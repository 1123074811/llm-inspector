#!/bin/bash
# LLM Inspector v16.0 - Unix/Linux/Mac Startup Script
# Idempotent dependency check + provenance validation + server start

set -e

echo "===================================="
echo "LLM Inspector v16.0"
echo "===================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}[OK]${NC} Virtual environment activated"
else
    echo -e "${YELLOW}[Warning]${NC} .venv not found, using system Python"
fi

# Enable asyncio mode
export ASYNCIO_MODE=1

# Change to backend directory
cd backend

# Idempotent dependency check: only install if core imports fail
echo ""
echo "===================================="
echo "Checking Dependencies..."
echo "===================================="
if python -c "import yaml, numpy, scipy, cryptography, certifi, tiktoken" 2>/dev/null; then
    echo -e "${GREEN}[OK]${NC} Core dependencies already satisfied"
else
    echo "[INFO] Installing missing core dependencies..."
    python scripts/setup_dependencies.py --skip-optional || {
        echo -e "${YELLOW}[WARN]${NC} Some dependencies failed to install, continuing..."
    }
fi

# Verify SOURCES.yaml provenance registry
echo ""
echo "===================================="
echo "Verifying Data Provenance..."
echo "===================================="
python start.py --verify-sources || {
    echo -e "${YELLOW}[WARN]${NC} Provenance check found issues (non-fatal in dev mode)"
}

# Start server
echo ""
echo "===================================="
echo "Starting Server..."
echo "===================================="
python start.py --port 9999
