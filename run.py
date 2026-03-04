"""
AI Interior Design Studio - Main Entry Point

A Flask-based API for AI-powered interior design visualization.
"""

import os
import sys

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.config import Config

app = create_app()

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    debug = os.getenv('FLASK_DEBUG', '0') == '1'
    
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║          🏠 AI Interior Design Studio                     ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Server running at: http://{host}:{port}                   ║
    ║  Debug mode: {'ON' if debug else 'OFF'}                                        ║
    ║  GPU available: {'Yes' if Config.USE_GPU else 'No'}                                      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    app.run(host=host, port=port, debug=debug)
