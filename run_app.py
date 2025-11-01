#!/usr/bin/env python3
"""
Quick launcher for the Reddit Scraper Dashboard
Run this file to start the Streamlit app instantly
"""

import subprocess
import sys
import os

def run_streamlit_app():
    """Launch the Streamlit dashboard"""
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    print(f"🚀 Starting Streamlit app: {app_path}")
    print("📱 Dashboard will open at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop\n")
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

if __name__ == "__main__":
    run_streamlit_app()
