#!/usr/bin/env python
"""
Start the Keyframe Detection Web Application
=============================================

This script starts the Flask web application from the project root.
It handles path setup and launches the webapp.

Usage:
    python run_webapp.py
    
Then navigate to: http://localhost:5000
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Change to project root directory
os.chdir(project_root)

# Import and run the Flask app
from webapp.app import app

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 KEYFRAME DETECTION WEB APPLICATION")
    print("="*70)
    print(f"Project Root: {project_root}")
    print(f"Server: http://localhost:5000")
    print("="*70 + "\n")
    
    # Run Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)
