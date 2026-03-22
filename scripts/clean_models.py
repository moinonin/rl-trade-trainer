#!/usr/bin/env python3
"""
Script to convert fragmented optimization results into valid JSON format.
Reads from stdin or a file and outputs valid JSON.
"""

import json
import re
import sys
from pathlib import Path

def extract_blocks(content):
    """Extract JSON blocks from the mixed content."""
    blocks = []
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for start of JSON block
        if line.startswith('{'):
            # Find the matching closing brace
            brace_count = 0
            block_lines = []
            start_i = i
            
            while i < len(lines):
                current_line = lines[i]
                block_lines.append(current_line)
                
                # Count braces in this line
                for char in current_line:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                
                if brace_count == 0:
                    break
                i += 1
            
            # Extract the JSON block
            block_text = '\n'.join(block_lines)
            try:
                # Parse to ensure it's valid JSON
                block_obj = json.loads(block_text)
                blocks.append(block_obj)
            except json.JSONDecodeError as e:
                # If parsing fails, try to fix common issues
                print(f"Warning: Could not parse block at line {start_i}: {e}", file=sys.stderr)
        
        i += 1
    
    return blocks

def clean_file(file_path, score_threshold=None):
    """Read and clean the file into valid JSON."""
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the lowest alpha value from the beginning
    lowest_alpha = None
    alpha_match = re.search(r'Lowest alpha value found:\s*([-\d.e+-]+)', content)
    if alpha_match:
        lowest_alpha = float(alpha_match.group(1))
    
    # Extract all JSON blocks
    blocks = extract_blocks(content)
    
    if not blocks:
        print("No valid JSON blocks found.", file=sys.stderr)
        return None
    
    # Optionally keep only blocks at/under the requested score threshold.
    if score_threshold is not None:
        filtered_blocks = []
        for block in blocks:
            score = block.get("score")
            if isinstance(score, (int, float)) and score <= score_threshold:
                filtered_blocks.append(block)
        blocks = filtered_blocks

    # Build the final JSON structure
    result = {
        "lowest_alpha_value": lowest_alpha,
        "score_threshold": score_threshold,
        "metrics": {
            "optimization_progress": blocks
        }
    }
    
    return result

def main():
    """Main entry point."""
    score_threshold = None
    if len(sys.argv) > 2:
        try:
            score_threshold = float(sys.argv[2])
        except ValueError:
            print(f"Invalid score threshold: {sys.argv[2]}", file=sys.stderr)
            sys.exit(1)

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Read from stdin
        content = sys.stdin.read()
        # If no file specified, assume content is from stdin
        # We need to create a temporary-like structure
        # For simplicity, we'll just try to process
        if not content:
            print("Usage: python clean_json.py <filename>", file=sys.stderr)
            print("Or pipe content to stdin", file=sys.stderr)
            sys.exit(1)
        
        # Process stdin content
        lowest_alpha = None
        alpha_match = re.search(r'Lowest alpha value found:\s*([-\d.e+-]+)', content)
        if alpha_match:
            lowest_alpha = float(alpha_match.group(1))
        
        blocks = extract_blocks(content)
        
        if not blocks:
            print("No valid JSON blocks found.", file=sys.stderr)
            sys.exit(1)
        
        result = {
            "lowest_alpha_value": lowest_alpha,
            "metrics": {
                "optimization_progress": blocks
            }
        }
        
        print(json.dumps(result, indent=2))
        return
    
    # Process file
    result = clean_file(file_path, score_threshold=score_threshold)
    if result:
        print(json.dumps(result, indent=2))
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
