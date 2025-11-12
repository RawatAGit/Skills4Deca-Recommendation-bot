#!/bin/bash

# Skills4Deca v2.4 Server Stop Script
# This script cleanly stops all running server instances

echo "========================================"
echo "Skills4Deca v2.4 Server Shutdown"
echo "========================================"
echo ""

# Check if server is running on port 5000
if lsof -ti:5000 > /dev/null 2>&1; then
    echo "🔍 Found server running on port 5000"
    echo "⏹️  Stopping server..."
    lsof -ti:5000 | xargs kill -9 2>/dev/null
    sleep 1

    # Verify it's stopped
    if lsof -ti:5000 > /dev/null 2>&1; then
        echo "❌ Failed to stop server on port 5000"
        exit 1
    else
        echo "✅ Server on port 5000 stopped"
    fi
else
    echo "ℹ️  No server running on port 5000"
fi

# Check for any orphaned app.py processes
if pgrep -f "python.*app.py" > /dev/null 2>&1; then
    echo "🔍 Found orphaned app.py processes"
    echo "⏹️  Cleaning up..."
    pkill -f "python.*app.py"
    sleep 1

    # Verify cleanup
    if pgrep -f "python.*app.py" > /dev/null 2>&1; then
        echo "⚠️  Some processes may still be running"
        echo "   Run: ps aux | grep app.py"
    else
        echo "✅ All app.py processes stopped"
    fi
else
    echo "ℹ️  No orphaned app.py processes found"
fi

echo ""
echo "========================================"
echo "✅ Shutdown complete"
echo "========================================"
echo ""
echo "To restart the server, run:"
echo "  ./start_server.sh"
echo ""
