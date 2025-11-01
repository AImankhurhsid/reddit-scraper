#!/usr/bin/env python3
"""
Quick launcher for the Reddit Scraper
Run this file to scrape Reddit posts
"""

import subprocess
import sys
import os

def run_scraper():
    """Launch the scraper with default settings"""
    scraper_path = os.path.join(os.path.dirname(__file__), "scrape_srm.py")
    print("🕷️  Starting Reddit Scraper...")
    print("📍 Subreddit: r/SRMUNIVERSITY")
    print("🌐 Language: English (en)")
    print("⏹️  Press Ctrl+C to stop\n")
    
    # Default: subreddit=SRMUNIVERSITY, lang=en
    subprocess.run([sys.executable, scraper_path, "--subreddit", "SRMUNIVERSITY", "--lang", "en"])

if __name__ == "__main__":
    run_scraper()
