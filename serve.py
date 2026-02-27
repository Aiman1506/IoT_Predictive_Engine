"""
FactoryGuard AI - Production API Server Entry Point
=====================================================
Run with: python serve.py
Or with gunicorn: gunicorn -w 4 -b 0.0.0.0:5000 serve:app
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from api.app import app, load_model_artifact

# Load model once at module level (needed for gunicorn multi-worker)
load_model_artifact()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
