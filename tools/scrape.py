#!/usr/bin/env python3
"""
Wiki Scraper - An elegant tool for scraping wiki content.

This script downloads and extracts content from wiki pages, respecting
robots.txt rules and implementing proper caching to avoid duplicate requests.
"""

import logging
import time
import json
import os
import sys
import pickle
import dbm
import urllib.robotparser
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Set, Optional, Union, Any

import requests
from bs4 import BeautifulSoup

# --- Configuration ---

DEFAULT_CONFIG = {
    "url_file": "tools/urls_to_scrape.txt",
    "output_dir": "tools/scraped_data",
    "url_cache_file": "tools/scraped_urls.pkl",
    "log_file": "tools/scraping.log",
    "max_pages_per_url": 200,
    "parallel": True,  # Enable parallelism by default
    "rate_limit": 2,
    "max_retries": 3,
    "backoff_factor": 2,
    "max_workers": 5,
    "max_depth": 2,  # How deep to follow links (1 = just direct links, 2 = links of links, etc.)
    "user_agent": "WikiScraper/1.0 (https://github.com/yourusername/wiki-scraper)"
}


# --- Utilities ---

class DBMBackedSet:
    """A set-like object that stores data in a DBM file to reduce memory usage."""
    
    def __init__(self, dbm_file: str):
        self.dbm_file = dbm_file
        
    def __contains__(self, item: str) -> bool:
        """Check if the item exists in the set."""
        with dbm.open(self.dbm_file, 'r') as db:
            return item.encode('utf-8') in db
            
    def add(self, item: str) -> None:
        """Add an item to the set."""
        with dbm.open(self.dbm_file, 'c') as db:
            db[item.encode('utf-8')] = b'1'
    
    def __len__(self) -> int:
        """Return the number of items in the set."""
        with dbm.open(self.dbm_file, 'r') as db:
            return len(db)


class RateLimiter:
    """Manages rate limiting for different domains to avoid overloading servers."""
    
    def __init__(self, default_rate: float = 2.0):
        self.default_rate = default_rate
        self.domain_rates: Dict[str, float] = {}
        self.last_request: Dict[str, float] = {}
        self.logger = logging.getLogger('wiki_scraper')
    
    def wait(self, domain: str) -> None:
        """Wait appropriate time based on domain-specific rate limits."""
        now = time.time()
        rate = self.domain_rates.get(domain, self.default_rate)
        
        if domain in self.last_request:
            elapsed = now - self.last_request[domain]
            wait_time = max(0, rate - elapsed)
            
            if wait_time > 0:
                self.logger.debug(f"Rate limiting: Waiting {wait_time:.2f}s for {domain}")
                time.sleep(wait_time)
        
        self.last_request[domain] = time.time()
    
    def adjust_rate(self, domain: str, success: bool = True) -> None:
        """Adjust rate limit based on success or failure."""
        current_rate = self.domain_rates.get(domain, self.default_rate)
        
        if success:
            # Slightly decrease wait time on success (be cautious)
            new_rate = max(0.5, current_rate * 0.95)
        else:
            # Increase wait time on failure
            new_rate = min(30, current_rate * 1.5)
        
        self.domain_rates[domain] = new_rate
        self.logger.debug(f"Adjusted rate for {domain}: {current_rate:.2f}s → {new_rate:.2f}s")


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Set up logging to both file and console."""
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create directory for log file if needed
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('wiki_scraper')
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Add handlers
    if log_file:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# --- Core Scraper ---

class WikiScraper:
    """
    A powerful and elegant wiki scraper that downloads and extracts content from wiki pages.
    
    Features:
    - Respects robots.txt rules
    - Implements caching to avoid duplicate requests
    - Handles rate limiting and retries
    - Organizes content in a structured format
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the scraper with the given configuration."""
        self.config = config
        self.base_url = None
        self.output_dir = config["output_dir"]
        self.url_cache_file = config["url_cache_file"]
        
        # Set up logging
        self.logger = setup_logging(config.get("log_file"))
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize URL cache
        self.visited_urls = self._load_visited_urls()
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(config.get("rate_limit", 2))
        
        # Initialize robot parsers
        self.robot_parsers = {}
        
        # Track statistics
        self.stats = {
            "pages_visited": 0,
            "pages_skipped": 0,
            "errors": 0,
            "content_saved": 0
        }
    
    def _load_visited_urls(self) -> Union[Set[str], DBMBackedSet]:
        """Load previously visited URLs from cache file with memory efficiency."""
        if os.path.exists(self.url_cache_file):
            try:
                # For very large URL sets, use a file-backed set to reduce memory usage
                if os.path.getsize(self.url_cache_file) > 10 * 1024 * 1024:  # > 10MB
                    self.logger.info("Large URL cache detected, using disk-backed storage")
                    # Convert pickle file to dbm if needed
                    if not os.path.exists(f"{self.url_cache_file}.db"):
                        with open(self.url_cache_file, 'rb') as f:
                            urls = pickle.load(f)
                        with dbm.open(f"{self.url_cache_file}.db", 'c') as db:
                            for url in urls:
                                db[url.encode('utf-8')] = b'1'
                    return DBMBackedSet(f"{self.url_cache_file}.db")
                else:
                    with open(self.url_cache_file, 'rb') as f:
                        return pickle.load(f)
            except Exception as e:
                self.logger.error(f"Error loading URL cache: {str(e)}")
                return set()
        return set()
    
    def save_visited_urls(self) -> None:
        """Save visited URLs to cache file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.url_cache_file), exist_ok=True)
            
            # If it's a DBMBackedSet, it's already saved on disk
            if isinstance(self.visited_urls, DBMBackedSet):
                self.logger.info(f"URL cache is maintained on disk")
                return
                
            with open(self.url_cache_file, 'wb') as f:
                pickle.dump(self.visited_urls, f)
            self.logger.info(f"Saved {len(self.visited_urls)} visited URLs to {self.url_cache_file}")
        except Exception as e:
            self.logger.error(f"Error saving URL cache: {str(e)}")
    
    def get_robot_parser(self, base_url: str) -> urllib.robotparser.RobotFileParser:
        """Get or create a robot parser for the given base URL."""
        if base_url not in self.robot_parsers:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(urljoin(base_url, "/robots.txt"))
            try:
                rp.read()
                self.logger.info(f"Read robots.txt for {base_url}")
            except Exception as e:
                self.logger.warning(f"Failed to read robots.txt for {base_url}: {str(e)}")
            self.robot_parsers[base_url] = rp
        return self.robot_parsers[base_url]
    
    def can_fetch(self, url: str) -> bool:
        """Check if the URL can be fetched according to robots.txt."""
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        rp = self.get_robot_parser(base_url)
        
        user_agent = self.config.get("user_agent", "WikiScraper/1.0")
        can_fetch = rp.can_fetch(user_agent, url)
        
        if not can_fetch:
            self.logger.warning(f"Robots.txt disallows fetching: {url}")
        
        return can_fetch
    
    def get_soup_from_url(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch a URL and return a BeautifulSoup object with retries and validation."""
        if url in self.visited_urls:
            self.logger.info(f"Skipping already visited URL: {url}")
            self.stats["pages_skipped"] += 1
            return None
        
        self.visited_urls.add(url)
        
        # Check robots.txt first
        if not self.can_fetch(url):
            self.stats["pages_skipped"] += 1
            return None
        
        # Apply rate limiting by domain
        domain = urlparse(url).netloc
        self.rate_limiter.wait(domain)
        
        headers = {
            'User-Agent': self.config.get("user_agent", "WikiScraper/1.0"),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': self.base_url,
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        max_retries = self.config.get("max_retries", 3)
        backoff_factor = self.config.get("backoff_factor", 2)
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Fetching {url} (attempt {attempt+1}/{max_retries})")
                
                response = requests.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    # Check content type to ensure it's HTML
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'text/html' not in content_type:
                        self.logger.warning(f"Skipping non-HTML content: {url} (Content-Type: {content_type})")
                        self.stats["errors"] += 1
                        return None
                    
                    self.stats["pages_visited"] += 1
                    self.rate_limiter.adjust_rate(domain, success=True)
                    return BeautifulSoup(response.text, 'html.parser')
                
                elif response.status_code == 429:  # Too Many Requests
                    wait_time = 60 * (attempt + 1)  # Progressive waiting
                    self.logger.warning(f"Rate limited (429). Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    self.rate_limiter.adjust_rate(domain, success=False)
                    continue
                
                else:
                    self.logger.error(f"Failed to fetch {url}: Status code {response.status_code}")
                    self.stats["errors"] += 1
                    
                    # Don't retry for certain status codes
                    if response.status_code in [404, 403, 410]:
                        break
            
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout fetching {url}, attempt {attempt+1}/{max_retries}")
                self.rate_limiter.adjust_rate(domain, success=False)
            
            except requests.exceptions.ConnectionError:
                self.logger.warning(f"Connection error fetching {url}, attempt {attempt+1}/{max_retries}")
                self.rate_limiter.adjust_rate(domain, success=False)
            
            except Exception as e:
                self.logger.error(f"Error fetching {url}: {str(e)}")
                self.stats["errors"] += 1
            
            # Don't wait after the last attempt
            if attempt < max_retries - 1:
                wait_time = backoff_factor * (attempt + 1)
                self.logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        return None
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename to prevent path traversal attacks."""
        # Remove directory components 
        filename = os.path.basename(filename)
        
        # Remove any null bytes (potential security issue)
        filename = filename.replace('\0', '')
        
        # Replace potentially problematic characters
        filename = ''.join(c for c in filename if c.isalnum() or c in '._- ')
        
        # Ensure it's not empty
        if not filename:
            filename = 'unnamed'
        
        return filename
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing extra whitespace, etc."""
        if not text:
            return ""
        
        # Replace multiple whitespace with a single space
        text = ' '.join(text.split())
        
        # Remove common wiki editing markers
        text = text.replace("[edit]", "")
        
        return text
    
    def extract_article_content(self, soup: BeautifulSoup, url: str) -> Optional[Dict[str, Any]]:
        """Extract structured content from a wiki article page."""
        try:
            # Extract title (typically in h1 with page-header__title class in Fandom wikis)
            title_element = soup.find('h1', class_='page-header__title')
            title = title_element.text.strip() if title_element else 'Unknown Title'
            
            # Get the main content div (for Fandom wikis, typically in mw-parser-output class)
            content_div = soup.find('div', {'class': 'mw-parser-output'})
            
            if not content_div:
                self.logger.warning(f"No main content found for {url}")
                return None
            
            # Remove navigation, edit links, etc.
            for element in content_div.select('.navigation, .toc, .editsection'):
                if element:
                    element.decompose()
            
            # Extract text paragraphs and clean them
            paragraphs = []
            for p in content_div.find_all('p'):
                text = self.clean_text(p.text)
                if text:
                    paragraphs.append(text)
            
            # Extract lists
            lists = []
            for ul in content_div.find_all(['ul', 'ol']):
                items = []
                for li in ul.find_all('li'):
                    text = self.clean_text(li.text)
                    if text:
                        items.append(text)
                if items:
                    lists.append(items)
            
            # Extract tables
            tables = []
            for table in content_div.find_all('table'):
                rows = []
                for tr in table.find_all('tr'):
                    cells = [self.clean_text(td.text) for td in tr.find_all(['td', 'th'])]
                    if any(cells):
                        rows.append(cells)
                if rows:
                    tables.append(rows)
            
            # Extract images
            images = []
            for img in content_div.find_all('img'):
                if 'src' in img.attrs:
                    image_url = img['src']
                    if not image_url.startswith(('http://', 'https://')):
                        image_url = urljoin(self.base_url, img['src'])
                    images.append({
                        'url': image_url,
                        'alt': img.get('alt', ''),
                        'title': img.get('title', '')
                    })
            
            # Extract section headers for better content structure
            sections = []
            current_section = {'title': None, 'level': 0, 'content': []}
            
            for element in content_div.children:
                if element.name and element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    level = int(element.name[1])
                    if current_section['content']:
                        sections.append(current_section)
                    current_section = {
                        'title': self.clean_text(element.text),
                        'level': level,
                        'content': []
                    }
                elif current_section['title'] is not None:
                    if element.name == 'p' and element.text.strip():
                        current_section['content'].append({
                            'type': 'paragraph',
                            'text': self.clean_text(element.text)
                        })
            
            if current_section['content']:
                sections.append(current_section)
            
            return {
                'title': title,
                'url': url,
                'paragraphs': paragraphs,
                'lists': lists,
                'tables': tables,
                'images': images,
                'sections': sections
            }
        
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {str(e)}")
            return None
    
    def extract_all_pages_links(self, soup: BeautifulSoup, url: str) -> List[str]:
        """Extract links from a Special:AllPages page or similar listing page."""
        links = []
        
        try:
            # The main content is usually in the mw-allpages-body class on AllPages
            content_area = soup.find('div', {'class': 'mw-allpages-body'})
            
            if not content_area:
                # For Dark Souls Fandom wiki, links appear to be in a list
                content_area = soup.find('ul')
            
            if content_area:
                # Find all links in the content area
                for a in content_area.find_all('a', href=True):
                    href = a['href']
                    # Make sure it's a full URL
                    full_url = urljoin(url, href)
                    if full_url not in self.visited_urls:
                        links.append(full_url)
            
            # Check if there's a next page
            next_page = soup.find('a', text=lambda t: t and 'Next page' in t)
            if next_page and 'href' in next_page.attrs:
                next_url = urljoin(url, next_page['href'])
                if next_url not in self.visited_urls:
                    links.append(next_url)
            
            self.logger.info(f"Found {len(links)} links on {url}")
            return links
        
        except Exception as e:
            self.logger.error(f"Error extracting links from {url}: {str(e)}")
            return []
    
    def create_output_folder_for_url(self, url: str) -> str:
        """Create an appropriate output folder for a given URL."""
        # Parse the URL to get the domain and path
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path
        
        # Create a folder structure based on the domain and path
        folder_name = domain.replace('.', '_')
        
        # If it's a special page, extract that information for the folder name
        if 'Special:' in path:
            special_part = path.split('Special:')[1].split('/')[0]
            folder_name = f"{folder_name}_Special_{special_part}"
        
        # Create the full folder path
        folder_path = os.path.join(self.output_dir, folder_name)
        
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            self.logger.info(f"Created output folder: {folder_path}")
        
        return folder_path
    
    def save_content_to_file(self, content: Dict[str, Any], url: str, folder_path: Optional[str] = None) -> Optional[str]:
        """Save extracted content to a JSON file in the appropriate folder."""
        if not content:
            return None
        
        try:
            # If no folder path is provided, use the default output directory
            if folder_path is None:
                folder_path = self.output_dir
                
            # Generate a safe filename from the URL or title
            if 'title' in content and content['title'] != 'Unknown Title':
                safe_filename = self.sanitize_filename(content['title'].replace(' ', '_'))
            else:
                # Extract the last part of the URL path
                safe_filename = self.sanitize_filename(url.split('/')[-1].replace(' ', '_'))
            
            # Ensure the filename is safe and add .json extension
            safe_filename = ''.join(c for c in safe_filename if c.isalnum() or c in '_-')
            json_filename = f"{safe_filename}.json"
            
            # Save the content
            file_path = os.path.join(folder_path, json_filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved content to {file_path}")
            self.stats["content_saved"] += 1
            return file_path
        
        except Exception as e:
            self.logger.error(f"Error saving content: {str(e)}")
            return None
    
    def scrape_entry_point_only(self, entry_url: str, max_pages: int = 100) -> None:
        """Scrape only the pages directly linked from the entry point URL."""
        try:
            self.logger.info(f"Starting to scrape entry point: {entry_url}")
            
            # Set base URL for this scraping session
            parsed_url = urlparse(entry_url)
            self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Create a folder for this entry URL
            folder_path = self.create_output_folder_for_url(entry_url)
            self.logger.info(f"Saving data for {entry_url} in folder: {folder_path}")
            
            # Create progress file path
            progress_file = os.path.join(folder_path, ".progress.json")
            
            # Check for existing progress
            processed_urls = set()
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress_data = json.load(f)
                        processed_urls = set(progress_data.get('processed_urls', []))
                        self.logger.info(f"Resuming from previous session. {len(processed_urls)} URLs already processed.")
                except Exception as e:
                    self.logger.error(f"Error loading progress file: {str(e)}")
            
            # First, get the entry page
            soup = self.get_soup_from_url(entry_url)
            if not soup:
                self.logger.error(f"Failed to fetch entry point: {entry_url}")
                return
                
            # Extract links from the entry page
            direct_links = self.extract_all_pages_links(soup, entry_url)
            self.logger.info(f"Found {len(direct_links)} direct links from entry point")
            
            # Process each direct link, but don't follow links from those pages
            processed_count = 0
            for i, url in enumerate(direct_links):
                # Skip already processed URLs from a previous run
                if url in processed_urls:
                    self.logger.info(f"Already processed in previous run: {url}")
                    processed_count += 1
                    continue
                    
                if processed_count >= max_pages:
                    self.logger.info(f"Reached maximum number of pages ({max_pages})")
                    break
                
                if url in self.visited_urls and url != entry_url:  # Allow revisiting the entry point
                    self.logger.info(f"Skipping already visited: {url}")
                    continue
                
                soup = self.get_soup_from_url(url)
                if not soup:
                    continue
                
                # Only extract content if it's a regular wiki page
                if 'Special:' not in url:
                    content = self.extract_article_content(soup, url)
                    self.save_content_to_file(content, url, folder_path)
                
                processed_count += 1
                processed_urls.add(url)
                
                # Save progress periodically (every 5 pages)
                if processed_count % 5 == 0:
                    self._save_progress(progress_file, processed_urls)
                    self.save_visited_urls()
                    
                self.logger.info(f"Processed {processed_count}/{len(direct_links)} direct links ({i+1} attempted)")
            
            # Final progress save
            self._save_progress(progress_file, processed_urls)
            
            # Save the updated visited URLs
            self.save_visited_urls()
            
            # Log statistics
            self.logger.info(f"Scraping complete for {entry_url}")
            self.logger.info(f"Stats: {json.dumps(self.stats, indent=2)}")
        
        except Exception as e:
            self.logger.error(f"Error during scraping of {entry_url}: {str(e)}")
    
    def _save_progress(self, progress_file: str, processed_urls: Set[str]) -> None:
        """Save progress information to a file."""
        try:
            with open(progress_file, 'w') as f:
                json.dump({
                    'processed_urls': list(processed_urls),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f)
            self.logger.debug(f"Progress saved to {progress_file}")
        except Exception as e:
            self.logger.error(f"Error saving progress: {str(e)}")
    
    def scrape_urls_parallel(self, urls: List[str], max_workers: int = 5, max_pages_per_url: int = 100) -> None:
        """Scrape multiple URLs in parallel using threads."""
        self.logger.info(f"Starting parallel scraping of {len(urls)} URLs with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all URLs to the executor
            futures = [executor.submit(self.scrape_entry_point_only, url, max_pages_per_url) for url in urls]
            
            # Wait for all tasks to complete
            for i, future in enumerate(futures):
                try:
                    future.result()  # This will re-raise any exception that occurred
                    self.logger.info(f"Completed task {i+1}/{len(futures)}")
                except Exception as e:
                    self.logger.error(f"Task {i+1}/{len(futures)} failed: {str(e)}")
        
        self.logger.info("Parallel scraping complete")

    def scrape_recursive(self, entry_url: str, max_pages: int = 100, max_depth: int = 2) -> None:
        """
        Recursively scrape pages starting from the entry point URL, following links to a specified depth.
        
        Args:
            entry_url: The starting URL to scrape
            max_pages: Maximum number of pages to scrape in total
            max_depth: How deep to follow links (1 = just direct links, 2 = links of links, etc.)
        """
        try:
            self.logger.info(f"Starting recursive scraping from: {entry_url} (max depth: {max_depth})")
            
            # Set base URL for this scraping session
            parsed_url = urlparse(entry_url)
            self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Create a folder for this entry URL
            folder_path = self.create_output_folder_for_url(entry_url)
            self.logger.info(f"Saving data for {entry_url} in folder: {folder_path}")
            
            # Create progress file path
            progress_file = os.path.join(folder_path, ".progress.json")
            
            # Check for existing progress
            processed_urls = set()
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress_data = json.load(f)
                        processed_urls = set(progress_data.get('processed_urls', []))
                        self.logger.info(f"Resuming from previous session. {len(processed_urls)} URLs already processed.")
                except Exception as e:
                    self.logger.error(f"Error loading progress file: {str(e)}")
            
            # Create a queue for BFS traversal with (url, depth) tuples
            from collections import deque
            queue = deque([(entry_url, 0)])  # Start with entry URL at depth 0
            
            # Track all URLs to be processed to avoid duplicates in the queue
            queued_urls = {entry_url}
            
            # Process pages in breadth-first order
            processed_count = 0
            
            while queue and processed_count < max_pages:
                # Get the next URL and its depth from the queue
                current_url, current_depth = queue.popleft()
                
                # Skip if already processed in a previous run
                if current_url in processed_urls:
                    self.logger.info(f"Already processed in previous run: {current_url}")
                    continue
                
                # Skip if already visited in this run (except entry point)
                if current_url in self.visited_urls and current_url != entry_url:
                    self.logger.info(f"Skipping already visited: {current_url}")
                    continue
                
                # Fetch the page
                soup = self.get_soup_from_url(current_url)
                if not soup:
                    continue
                
                # Extract and save content if it's a regular wiki page
                if 'Special:' not in current_url:
                    content = self.extract_article_content(soup, current_url)
                    self.save_content_to_file(content, current_url, folder_path)
                    processed_count += 1
                    processed_urls.add(current_url)
                    
                    # Save progress periodically
                    if processed_count % 5 == 0:
                        self._save_progress(progress_file, processed_urls)
                        self.save_visited_urls()
                        self.logger.info(f"Processed {processed_count} pages, queue size: {len(queue)}")
                
                # If we haven't reached max depth, extract links and add to queue
                if current_depth < max_depth:
                    # If it's the entry point, use extract_all_pages_links, otherwise use a general link extractor
                    if current_url == entry_url:
                        links = self.extract_all_pages_links(soup, current_url)
                    else:
                        links = self.extract_regular_links(soup, current_url)
                    
                    self.logger.info(f"Found {len(links)} links at depth {current_depth} on {current_url}")
                    
                    # Add new links to the queue
                    for link in links:
                        if link not in queued_urls:
                            queue.append((link, current_depth + 1))
                            queued_urls.add(link)
                            
                            # Safety check to avoid excessive queuing
                            if len(queued_urls) >= max_pages * 10:
                                self.logger.warning(f"Queue size limit reached ({len(queued_urls)} URLs)")
                                break
            
            # Final progress save
            self._save_progress(progress_file, processed_urls)
            self.save_visited_urls()
            
            # Log statistics
            self.logger.info(f"Recursive scraping complete for {entry_url}")
            self.logger.info(f"Processed {processed_count} pages out of {len(queued_urls)} discovered")
            self.logger.info(f"Stats: {json.dumps(self.stats, indent=2)}")
        
        except Exception as e:
            self.logger.error(f"Error during recursive scraping of {entry_url}: {str(e)}")

    def extract_regular_links(self, soup: BeautifulSoup, url: str) -> List[str]:
        """Extract links from a regular wiki page, including galleries and thumbnails."""
        links = []
        found_links = set()  # Track unique links
        
        try:
            # Get the main content
            content_div = soup.find('div', {'class': 'mw-parser-output'})
            
            if not content_div:
                self.logger.warning(f"No content div found on {url}")
                return []
            
            # Method 1: Extract all direct links in content
            for a in content_div.find_all('a', href=True):
                href = a['href']
                
                # Skip external links, special pages, etc.
                if href.startswith('/wiki/') and not href.startswith('/wiki/Special:') and not href.startswith('/wiki/File:'):
                    full_url = urljoin(url, href)
                    
                    # Only add unique URLs we haven't visited yet
                    if full_url not in self.visited_urls and full_url not in found_links:
                        links.append(full_url)
                        found_links.add(full_url)
            
            # Method 2: Find all gallery sections with various possible class names
            gallery_containers = content_div.find_all(['div', 'section', 'ul'], 
                                                      class_=lambda c: c and any(gallery_class in str(c).lower() 
                                                                              for gallery_class in ['gallery', 'thumbnails', 'thumb']))
            
            for container in gallery_containers:
                self.logger.debug(f"Found gallery container with class: {container.get('class', 'None')}")
                
                # Extract links from gallery items
                gallery_items = container.find_all(['li', 'div', 'figure'], 
                                                  class_=lambda c: c and any(item_class in str(c).lower() 
                                                                          for item_class in ['gallery-item', 'thumb', 'image']))
                
                # If no specific gallery items found, look for any links in the container
                if not gallery_items:
                    gallery_items = [container]
                
                for item in gallery_items:
                    # Try to find the link - sometimes in the item, sometimes in a child element
                    link = item.find('a', href=True)
                    if link and 'href' in link.attrs:
                        href = link['href']
                        if href.startswith('/wiki/') and not href.startswith('/wiki/Special:') and not href.startswith('/wiki/File:'):
                            full_url = urljoin(url, href)
                            if full_url not in self.visited_urls and full_url not in found_links:
                                links.append(full_url)
                                found_links.add(full_url)
            
            # Method 3: Look for specific heading sections that often contain important links
            # Find section headings (h2, h3, etc.)
            for heading in content_div.find_all(['h2', 'h3', 'h4']):
                heading_text = heading.get_text().lower()
                
                # If this heading suggests it contains a list of items (gallery, list, types, etc.)
                if any(keyword in heading_text for keyword in ['gallery', 'types', 'shields', 'weapons', 'armor', 'list']):
                    # Look at all elements after this heading until the next heading
                    element = heading.find_next()
                    while element and element.name not in ['h2', 'h3', 'h4']:
                        # Process links in this section
                        for a in element.find_all('a', href=True) if hasattr(element, 'find_all') else []:
                            href = a['href']
                            if href.startswith('/wiki/') and not href.startswith('/wiki/Special:') and not href.startswith('/wiki/File:'):
                                full_url = urljoin(url, href)
                                if full_url not in self.visited_urls and full_url not in found_links:
                                    links.append(full_url)
                                    found_links.add(full_url)
                        
                        element = element.find_next()
            
            # Method 4: Handle table-based galleries (some wikis use these)
            for table in content_div.find_all('table'):
                # Check if this looks like a gallery table (has images)
                if table.find('img'):
                    for cell in table.find_all(['td', 'th']):
                        for a in cell.find_all('a', href=True):
                            href = a['href']
                            if href.startswith('/wiki/') and not href.startswith('/wiki/Special:') and not href.startswith('/wiki/File:'):
                                full_url = urljoin(url, href)
                                if full_url not in self.visited_urls and full_url not in found_links:
                                    links.append(full_url)
                                    found_links.add(full_url)
            
            # Method 5: Check for common Fandom gallery patterns (different from standard galleries)
            # This handles cases where galleries are implemented with custom CSS/HTML
            image_wrappers = content_div.find_all(['div', 'span'], class_=lambda c: c and 'image' in str(c).lower())
            for wrapper in image_wrappers:
                link = wrapper.find('a', href=True)
                if link and 'href' in link.attrs:
                    href = link['href']
                    if href.startswith('/wiki/') and not href.startswith('/wiki/Special:') and not href.startswith('/wiki/File:'):
                        # Check if this is a content link, not just an image link
                        # Image links often have File: or Image: in them
                        if '/File:' not in href and '/Image:' not in href:
                            full_url = urljoin(url, href)
                            if full_url not in self.visited_urls and full_url not in found_links:
                                links.append(full_url)
                                found_links.add(full_url)
            
            self.logger.info(f"Found {len(links)} unique links on page {url}")
            return links
        
        except Exception as e:
            self.logger.error(f"Error extracting links from {url}: {str(e)}")
            return []


# --- CLI Interface ---

def load_config(config_file: str = "tools/config.json") -> Dict[str, Any]:
    """Load configuration from file, or create default if it doesn't exist."""
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
            return DEFAULT_CONFIG.copy()
    else:
        # Create default config file
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        print(f"Created default configuration file at {config_file}")
        return DEFAULT_CONFIG.copy()


def read_urls_from_file(filename: str) -> List[str]:
    """Read URLs from a text file, ignoring empty lines and comments."""
    urls = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    urls.append(line)
        return urls
    except Exception as e:
        print(f"Error reading URL file {filename}: {str(e)}")
        return []


def clear_url_cache(cache_file: str = "tools/scraped_urls.pkl") -> None:
    """Clear the URL cache file."""
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"URL cache cleared: {cache_file}")
    else:
        print(f"No URL cache found at {cache_file}")


def create_example_url_file(filename: str) -> None:
    """Create an example URL file if it doesn't exist."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write("# Add URLs to scrape below, one per line\n")
        f.write("# Lines starting with # are ignored\n\n")
        f.write("https://darksouls.fandom.com/wiki/Special:AllPages\n")
    
    print(f"Example URL file created at {filename}. Please edit it and run the script again.")


def parse_arguments() -> Dict[str, Any]:
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Scrape content from wiki pages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        default="tools/config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--url-file", 
        help="Path to file containing URLs to scrape"
    )
    
    parser.add_argument(
        "--output-dir", 
        help="Directory to store scraped content"
    )
    
    parser.add_argument(
        "--max-pages", 
        type=int,
        help="Maximum number of pages to scrape per URL"
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        help="How deep to follow links (1 = direct links only, 2 = links of links, etc.)"
    )
    
    parser.add_argument(
        "--no-parallel", 
        action="store_true",
        help="Disable parallel scraping (run sequentially)"
    )
    
    parser.add_argument(
        "--workers", 
        type=int,
        help="Number of worker threads for parallel scraping"
    )
    
    parser.add_argument(
        "--clear-cache", 
        action="store_true",
        help="Clear the URL cache before scraping"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    return vars(parser.parse_args())


def main() -> None:
    """Main entry point for the scraper."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args["config"])
    
    # Override config with command line arguments
    for key, value in args.items():
        if value is not None and key in config:
            config[key] = value
    
    # Handle the no-parallel option by inverting it for the config
    if args["no_parallel"]:
        config["parallel"] = False
    
    # Set up logging
    log_level = logging.DEBUG if args["debug"] else logging.INFO
    logger = setup_logging(config["log_file"], log_level)
    
    # Clear cache if requested
    if args["clear_cache"]:
        clear_url_cache(config["url_cache_file"])
    
    # Read URLs from file
    urls = read_urls_from_file(config["url_file"])
    
    if not urls:
        logger.warning(f"No URLs found in {config['url_file']}. Creating an example file...")
        create_example_url_file(config["url_file"])
        return
    
    # Create scraper
    scraper = WikiScraper(config)
    
    # Start scraping
    logger.info(f"Starting scraping of {len(urls)} URLs")
    
    # Get max depth from config
    max_depth = config.get("max_depth", 2)
    
    if config.get("parallel", True):
        max_workers = config.get("max_workers", 5)
        logger.info(f"Parallel mode enabled with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all URLs to the executor for recursive scraping
            futures = [
                executor.submit(
                    scraper.scrape_recursive, 
                    url, 
                    config["max_pages_per_url"], 
                    max_depth
                ) for url in urls
            ]
            
            # Wait for all tasks to complete
            for i, future in enumerate(futures):
                try:
                    future.result()  # This will re-raise any exception that occurred
                    logger.info(f"Completed task {i+1}/{len(futures)}")
                except Exception as e:
                    logger.error(f"Task {i+1}/{len(futures)} failed: {str(e)}")
    else:
        logger.info("Sequential mode enabled")
        # Process each URL sequentially
        for url in urls:
            logger.info(f"\nProcessing URL: {url}")
            scraper.scrape_recursive(url, config["max_pages_per_url"], max_depth)
    
    logger.info("Scraping complete")


if __name__ == "__main__":
    main()
