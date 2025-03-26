import json
import os
import sys

DEFAULT_CONFIG = {
    "url_file": "tools/urls_to_scrape.txt",
    "output_dir": "tools/scraped_data",
    "url_cache_file": "tools/scraped_urls.pkl",
    "log_file": "tools/scraping.log",
    "max_pages_per_url": 200,
    "rate_limit": 2,  # seconds between requests
    "max_retries": 3,
    "backoff_factor": 2,
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def load_config(config_file="tools/config.json"):
    """Load configuration from file, or create default if it doesn't exist"""
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Merge with defaults for any missing values
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                return config
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
            print("Using default configuration")
            return DEFAULT_CONFIG
    else:
        # Create default config file
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        print(f"Created default configuration file at {config_file}")
        return DEFAULT_CONFIG 