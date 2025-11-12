#!/bin/bash

# Skills4Deca v2.4 Server Startup Script
# This script ensures a clean start of the recommendation server

echo "========================================"
echo "Skills4Deca v2.4 Server Startup"
echo "========================================"
echo ""

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with your DEEPINFRA_API_KEY"
    exit 1
fi

# Kill any existing instances on port 5000
echo "🔍 Checking for existing server instances..."
if lsof -ti:5000 > /dev/null 2>&1; then
    echo "⚠️  Port 5000 is in use. Stopping existing server..."
    lsof -ti:5000 | xargs kill -9 2>/dev/null
    sleep 2
    echo "✅ Existing server stopped"
else
    echo "✅ Port 5000 is free"
fi

# Kill any orphaned app.py processes
if pgrep -f "python.*app.py" > /dev/null 2>&1; then
    echo "⚠️  Found orphaned app.py processes. Cleaning up..."
    pkill -f "python.*app.py"
    sleep 1
    echo "✅ Cleanup complete"
fi

# Verify Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found!"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

echo ""
echo "🚀 Starting Skills4Deca v2.4 server..."
echo "   - Multi-Query Analysis (rule-based)"
echo "   - Hybrid Search (70% semantic + 30% keyword)"
echo "   - Qwen3-Reranker-4B (cross-encoder)"
echo "   - GLM-4.6 LLM (validation + explanations)"
echo ""
echo "📡 Server will be available at:"
echo "   - Local:   http://localhost:5000"
echo "   - Network: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "📋 Logs will be written to: logs/hybrid_rag.log"
echo ""
echo "⚡ Press Ctrl+C to stop the server"
echo ""
echo "========================================"
echo ""

# Start the server
python3 app.py

# If server exits, show message
echo ""
echo "========================================"
echo "Server stopped"
echo "========================================"
