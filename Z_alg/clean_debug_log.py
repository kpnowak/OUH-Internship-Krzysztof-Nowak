#!/usr/bin/env python3
"""
Script to clean up debug.log file by removing matplotlib locator debug messages
and other noise while preserving useful debug information.
"""

import re
import os
from datetime import datetime

def clean_debug_log(input_file="debug.log", output_file="debug_cleaned.log"):
    """
    Clean the debug log file by removing matplotlib locator messages and other noise.
    
    Parameters:
    -----------
    input_file : str
        Path to the input debug log file
    output_file : str
        Path to the cleaned output file
    """
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    # Patterns to filter out (these will be removed)
    noise_patterns = [
        r'DEBUG locator: <matplotlib\.ticker\.AutoLocator object at 0x[0-9A-Fa-f]+>',
        r'DEBUG findfont: score\(FontEntry\(',
        r'DEBUG STREAM b\'[^\']*\' \d+ \d+',
        r'DEBUG b\'[^\']*\' \d+ \d+ \(unknown\)',
        r'DEBUG Loaded backend \w+ version',
    ]
    
    # Compile patterns for efficiency
    compiled_patterns = [re.compile(pattern) for pattern in noise_patterns]
    
    lines_read = 0
    lines_kept = 0
    lines_removed = 0
    
    print(f"Cleaning {input_file}...")
    print(f"Output will be saved to {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            lines_read += 1
            
            # Check if line matches any noise pattern
            is_noise = any(pattern.search(line) for pattern in compiled_patterns)
            
            if not is_noise:
                outfile.write(line)
                lines_kept += 1
            else:
                lines_removed += 1
            
            # Progress indicator
            if lines_read % 1000 == 0:
                print(f"Processed {lines_read} lines...")
    
    print(f"\nCleaning completed!")
    print(f"Lines read: {lines_read}")
    print(f"Lines kept: {lines_kept}")
    print(f"Lines removed: {lines_removed}")
    print(f"Reduction: {lines_removed/lines_read*100:.1f}%")
    
    # Show file sizes
    original_size = os.path.getsize(input_file)
    cleaned_size = os.path.getsize(output_file)
    size_reduction = (original_size - cleaned_size) / original_size * 100
    
    print(f"\nFile size reduction:")
    print(f"Original: {original_size:,} bytes")
    print(f"Cleaned:  {cleaned_size:,} bytes")
    print(f"Saved:    {original_size - cleaned_size:,} bytes ({size_reduction:.1f}%)")

def backup_original(input_file="debug.log"):
    """Create a backup of the original debug.log file."""
    if os.path.exists(input_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{input_file}.backup_{timestamp}"
        
        print(f"Creating backup: {backup_file}")
        with open(input_file, 'rb') as src, open(backup_file, 'wb') as dst:
            dst.write(src.read())
        print(f"Backup created successfully!")
        return backup_file
    return None

def replace_original_with_cleaned(cleaned_file="debug_cleaned.log", original_file="debug.log"):
    """Replace the original debug.log with the cleaned version."""
    if os.path.exists(cleaned_file):
        print(f"Replacing {original_file} with cleaned version...")
        
        # Remove original and rename cleaned file
        os.remove(original_file)
        os.rename(cleaned_file, original_file)
        
        print(f"Successfully replaced {original_file} with cleaned version!")
    else:
        print(f"Error: {cleaned_file} not found!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean debug.log file by removing matplotlib noise")
    parser.add_argument("--input", default="debug.log", help="Input debug log file")
    parser.add_argument("--output", default="debug_cleaned.log", help="Output cleaned log file")
    parser.add_argument("--backup", action="store_true", help="Create backup of original file")
    parser.add_argument("--replace", action="store_true", help="Replace original file with cleaned version")
    
    args = parser.parse_args()
    
    # Create backup if requested
    backup_file = None
    if args.backup:
        backup_file = backup_original(args.input)
    
    # Clean the log file
    clean_debug_log(args.input, args.output)
    
    # Replace original if requested
    if args.replace:
        replace_original_with_cleaned(args.output, args.input)
    
    print("\nDone!") 