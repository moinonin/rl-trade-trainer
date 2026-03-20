import os
import json
import sys
from pathlib import Path

def find_alpha_blocks(data, current_path=""):
    """
    Recursively traverse JSON data and yield (path, alpha, block)
    for any object that contains a 'contributions' dict with an 'alpha' key.
    The 'alpha' value must be a number.
    """
    if isinstance(data, dict):
        # Check if this dict has a 'contributions' dict containing 'alpha'
        contributions = data.get('contributions')
        if isinstance(contributions, dict):
            alpha = contributions.get('_raw_alpha')
            if alpha is not None and isinstance(alpha, (int, float)):
                yield current_path, alpha, data   # yield the block that holds contributions

        # Recurse into every value of the dict
        for key, value in data.items():
            new_path = f"{current_path}.{key}" if current_path else key
            yield from find_alpha_blocks(value, new_path)

    elif isinstance(data, list):
        # Recurse into every element of the list
        for idx, item in enumerate(data):
            yield from find_alpha_blocks(item, f"{current_path}[{idx}]")

def search_json_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        print(f"Error: {directory} is not a valid directory.")
        return

    json_files = list(directory.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in {directory}")
        return

    # Store all found (alpha, file, path, block) tuples
    found_entries = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not parse {json_file}: {e}")
            continue

        for path, alpha, block in find_alpha_blocks(data):
            found_entries.append((alpha, json_file, path, block))

    if not found_entries:
        print("No 'alpha' values found inside any 'contributions' object.")
        return

    # Find the minimum alpha value
    min_alpha = min(entry[0] for entry in found_entries)

    # Collect all entries with that minimum value
    min_entries = [entry for entry in found_entries if entry[0] == min_alpha]

    print(f"Lowest alpha value found: {min_alpha}\n")
    for alpha, file, path, block in min_entries:
        print(f"File: {file}")
        print(f"Path: {path}")
        # Show a few details from the block (optional)
        iteration = block.get('iteration', 'N/A')
        print(f"Iteration: {iteration}")
        print("Block (excerpt):")
        # Print only the keys that are likely to be useful, or a compact representation
        # Here we print the whole block but you can limit it if desired
        print(json.dumps(block, indent=2))
        print("-" * 60)

if __name__ == "__main__":
    # Usage: python script.py [directory]
    # If no directory is given, the current working directory is used.
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = "."

    search_json_files(target_dir)