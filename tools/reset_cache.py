#!/usr/bin/env python3
"""
Reset Cache - A utility script to clear the scraped URLs cache for the Wiki Scraper.

This script deletes the URL cache file to start fresh with URL tracking.
"""

import os
import argparse
import sys


def clear_url_cache(cache_file: str) -> None:
    """Clear the URL cache file."""
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            print(f"✅ URL cache cleared: {cache_file}")
            
            # Also check for the DBM version of the cache
            dbm_file = f"{cache_file}.db"
            if os.path.exists(dbm_file):
                os.remove(dbm_file)
                print(f"✅ DBM cache also cleared: {dbm_file}")
            
            return True
        except Exception as e:
            print(f"❌ Error clearing cache: {str(e)}")
            return False
    else:
        print(f"ℹ️ No URL cache found at {cache_file}")
        return True


def main():
    """Main function to handle command-line arguments and reset the cache."""
    parser = argparse.ArgumentParser(
        description="Reset the URL cache for the Wiki Scraper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--cache-file",
        default="tools/scraped_urls.pkl",
        help="Path to the URL cache file"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reset the cache without confirmation"
    )
    
    args = parser.parse_args()
    
    # Ask for confirmation unless --force is used
    if not args.force:
        response = input(f"Are you sure you want to reset the URL cache at {args.cache_file}? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            return
    
    success = clear_url_cache(args.cache_file)
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 