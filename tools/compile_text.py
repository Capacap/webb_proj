#!/usr/bin/env python3
"""
Compile Text - A utility script to convert scraped JSON data to plain text.

This script processes JSON files and extracts their textual content into
concise plain text files optimized for token efficiency in vector databases.
"""

import os
import json
import argparse
import glob
import re
from typing import Dict, List, Any, Optional, Set


def clean_text(text: str) -> str:
    """Clean text to remove redundancies and excessive whitespace."""
    if not text:
        return ""
    
    # Replace multiple spaces, newlines and tabs with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common wiki formatting elements
    text = re.sub(r'\[edit\]|\[source\]|\[citation needed\]', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove redundant punctuation
    text = re.sub(r'[.!?]{2,}', '.', text)
    
    return text.strip()


def deduplicate_content(parts: List[str]) -> List[str]:
    """Remove duplicate or nearly duplicate content."""
    unique_parts = []
    seen_content = set()
    
    for part in parts:
        # Create a simplified version for comparison (lowercase, only alphanumeric)
        simplified = re.sub(r'[^a-z0-9]', '', part.lower())
        
        # Skip very short parts
        if len(simplified) < 10:
            continue
            
        # Skip if we've seen this content already
        if simplified in seen_content:
            continue
            
        # Add significant content parts to results
        if len(simplified) > 20:  # Only keep substantive content
            unique_parts.append(part)
            seen_content.add(simplified)
    
    return unique_parts


def extract_text_from_json(json_data: Dict[str, Any], concise: bool = True) -> str:
    """
    Extract meaningful text content from JSON data while minimizing tokens.
    
    Args:
        json_data: The JSON data to extract text from
        concise: If True, produce more concise output
    
    Returns:
        Extracted text optimized for token efficiency
    """
    text_parts = []
    
    # Content priority extraction - focus on the most important information first
    
    # Title is important - always include if available
    if 'title' in json_data and json_data['title'] and json_data['title'] != 'Unknown Title':
        title = clean_text(json_data['title'])
        text_parts.append(f"Title: {title}")
    
    # Only include URL in non-concise mode
    if not concise and 'url' in json_data and json_data['url']:
        text_parts.append(f"Source: {json_data['url']}")
    
    # Main content extraction - adapt to different possible structures
    content_added = False
    
    # Try to extract from paragraphs (most common in our scraper)
    if 'paragraphs' in json_data and json_data['paragraphs']:
        cleaned_paragraphs = [clean_text(p) for p in json_data['paragraphs'] if clean_text(p)]
        
        # Filter out very short paragraphs for concise mode
        if concise:
            cleaned_paragraphs = [p for p in cleaned_paragraphs if len(p) > 30]
        
        if cleaned_paragraphs:
            text_parts.append("Content:")
            text_parts.extend(cleaned_paragraphs)
            content_added = True
    
    # Try sections if no paragraphs or as additional content
    if 'sections' in json_data and json_data['sections']:
        sections_text = []
        
        for section in json_data['sections']:
            if not section.get('title'):
                continue
                
            section_title = clean_text(section['title'])
            
            # Skip redundant or uninformative sections
            if concise and any(skip in section_title.lower() for skip in ['reference', 'external link', 'see also', 'notes']):
                continue
            
            section_content = []
            
            # Extract section content based on different possible structures
            if 'content' in section:
                if isinstance(section['content'], list):
                    for item in section['content']:
                        if isinstance(item, dict) and 'text' in item:
                            cleaned = clean_text(item['text'])
                            if cleaned:
                                section_content.append(cleaned)
                        elif isinstance(item, str):
                            cleaned = clean_text(item)
                            if cleaned:
                                section_content.append(cleaned)
                elif isinstance(section['content'], str):
                    cleaned = clean_text(section['content'])
                    if cleaned:
                        section_content.append(cleaned)
            
            # Add section if it has meaningful content
            if section_content:
                sections_text.append(f"{section_title}:")
                sections_text.extend(section_content)
        
        if sections_text:
            text_parts.extend(sections_text)
            content_added = True
    
    # Try to extract from 'text' field if nothing else worked
    if not content_added and 'text' in json_data and isinstance(json_data['text'], str):
        cleaned = clean_text(json_data['text'])
        if cleaned:
            text_parts.append(cleaned)
    
    # Extract from lists but format them efficiently - only if we haven't added much content yet
    if (not content_added or len(text_parts) < 3) and 'lists' in json_data and json_data['lists']:
        list_items = []
        
        for list_items_group in json_data['lists']:
            if isinstance(list_items_group, list):
                # Clean and filter list items
                cleaned_items = [clean_text(item) for item in list_items_group if item]
                cleaned_items = [item for item in cleaned_items if item and len(item) > 5]
                
                if cleaned_items:
                    # For token efficiency, join list items with commas
                    if concise and all(len(item) < 40 for item in cleaned_items):
                        list_items.append("Items: " + ", ".join(cleaned_items))
                    else:
                        list_items.extend(cleaned_items)
        
        if list_items:
            text_parts.extend(list_items)
    
    # Include tables only if critical and no other content, as they use many tokens
    if not content_added and 'tables' in json_data and json_data['tables']:
        for table in json_data['tables'][:1]:  # Limit to 1 table in concise mode
            # Convert table to text-only format to save tokens
            rows_text = []
            for row in table:
                if any(cell.strip() for cell in row):  # Skip empty rows
                    row_text = " | ".join(clean_text(cell) for cell in row if cell.strip())
                    if row_text:
                        rows_text.append(row_text)
            
            if rows_text:
                text_parts.extend(rows_text)
    
    # Deduplicate content to save tokens
    text_parts = deduplicate_content(text_parts)
    
    # Limit total output size in concise mode
    if concise and len(text_parts) > 10:
        # Keep title and first 9 content pieces
        text_parts = text_parts[:10]
    
    return "\n".join(text_parts)


def process_json_file(json_path: str, output_dir: str, concise: bool = True, 
                      append_to_single: Optional[str] = None) -> None:
    """Process a single JSON file and convert it to text."""
    try:
        # Read the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON in {json_path}")
                return
        
        # Skip files with no meaningful content
        if not any(key in json_data for key in ['title', 'paragraphs', 'sections', 'text', 'lists']):
            print(f"Skipping {json_path} - no extractable content")
            return
            
        # Extract text content
        text_content = extract_text_from_json(json_data, concise=concise)
        
        # Skip if no meaningful content was extracted
        if not text_content or len(text_content.strip()) < 50:
            print(f"Skipping {json_path} - insufficient content after extraction")
            return
        
        if append_to_single:
            # Append to a single file with a minimal separator
            with open(append_to_single, 'a', encoding='utf-8') as f:
                f.write(f"\n---\n")
                f.write(text_content)
                f.write("\n")
        else:
            # Create output filename based on input filename
            base_name = os.path.basename(json_path)
            file_name = os.path.splitext(base_name)[0] + ".txt"
            output_path = os.path.join(output_dir, file_name)
            
            # Write to individual text file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            print(f"Converted {json_path} to {output_path}")
    
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")


def find_json_files(input_dir: str) -> List[str]:
    """Find all JSON files in the input directory and its subdirectories."""
    json_pattern = os.path.join(input_dir, "**", "*.json")
    return glob.glob(json_pattern, recursive=True)


def main():
    """Main function to handle command-line arguments and process files."""
    parser = argparse.ArgumentParser(
        description="Convert JSON data to plain text optimized for vector databases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input-dir",
        default="tools/scraped_data",
        help="Directory containing JSON files"
    )
    
    parser.add_argument(
        "--output-dir",
        default="tools/text_data",
        help="Directory to store the text files"
    )
    
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Combine all text into a single output file"
    )
    
    parser.add_argument(
        "--output-file",
        default="tools/text_data/combined.txt",
        help="Path for the combined output file (used with --single-file)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include more details in the output (URLs, metadata, etc.)"
    )
    
    parser.add_argument(
        "--min-content-length",
        type=int,
        default=50,
        help="Minimum content length to include (in characters)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all JSON files
    json_files = find_json_files(args.input_dir)
    print(f"Found {len(json_files)} JSON files to process")
    
    # If single file mode, create/clear the output file
    single_output_file = None
    if args.single_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        open(args.output_file, 'w').close()  # Clear file if it exists
        single_output_file = args.output_file
        print(f"Creating combined output file: {args.output_file}")
    
    # Process each file with appropriate conciseness setting
    concise = not args.verbose
    
    for i, json_file in enumerate(json_files):
        if i > 0 and i % 100 == 0:
            print(f"Processed {i}/{len(json_files)} files...")
        process_json_file(json_file, args.output_dir, concise, single_output_file)
    
    print(f"Conversion completed. Processed {len(json_files)} files.")
    if args.single_file:
        print(f"All content combined into {args.output_file}")
    else:
        print(f"Text files saved in {args.output_dir}")


if __name__ == "__main__":
    main() 