#!/bin/bash
set -e

# Check if we're in production (skip local-only helpers)
if [ "$1" = "production" ]; then
    echo "--- Production Build ---"
    pip install --force-reinstall -r requirements-production.txt
else
    echo "--- Development Build ---"
    pip install --force-reinstall -r requirements.txt
    pip install --upgrade "httpx>=0.25.0"

    echo "--- Generating visualization dataset ---"
    if ! python generate_visualization_data.py; then
        echo "WARNING: Visualization dataset generation failed.\nEnsure course_metadata_cache.json exists and rerun the script manually."
    fi
fi

# Install gunicorn (not in requirements.txt)
pip install gunicorn

# Build the React frontend
cd react-frontend
npm install
npm run build
cd ..

# Make start script executable
chmod +x start.sh

echo "--- Build complete! ---"
if [ "$1" = "production" ]; then
    # Check if precomputed assets exist
    echo "Checking for precomputed assets..."

    if [ ! -f "course_metadata_cache.json" ] || [ ! -f "bm25_index.pkl" ]; then
        echo "⚠️  Precomputed assets not found, generating..."
        python precompute_assets.py
        echo "✅ Assets precomputed for deployment"
    else
        echo "✅ Course metadata cache: $(ls -lh course_metadata_cache.json | awk '{print $5}')"
        echo "✅ BM25 index: $(ls -lh bm25_index.pkl | awk '{print $5}')"
    fi

    if [ ! -f "visualization_data_precomputed.json" ]; then
        echo "⚠️  Precomputed visualization not found, generating fallback..."
        python deployment_fallback.py
        echo "✅ Fallback visualization generated for deployment"
    else
        echo "✅ Precomputed visualization data found: $(ls -lh visualization_data_precomputed.json | awk '{print $5}')"
    fi
else
    echo "Development build: Ready with precomputed visualization data"
fi
