#!/bin/bash
# LLM Inspector v9.0 - Unix/Linux/Mac Startup Script
# Automatically checks and installs dependencies before starting

set -e

echo "===================================="
echo "LLM Inspector v9.0"
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

# Check and install dependencies
echo ""
echo "===================================="
echo "Checking Dependencies..."
echo "===================================="
python scripts/setup_dependencies.py --skip-optional
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}[WARN]${NC} Some optional dependencies failed to install"
    echo -e "${YELLOW}[INFO]${NC} Continuing with core functionality..."
fi

# Start server
echo ""
echo "===================================="
echo "Starting Server..."
echo "===================================="
python start.py --port 8000 --strict-provenance
