import os
import json
import sys
from pathlib import Path

def find_blocks_with_low_score(data, threshold, current_path=""):
    """
    Recursively traverse JSON data and yield (path, block) for any object
    that has a 'score' key and whose value is < threshold.
    """
    if isinstance(data, dict):
        # If this dict itself has a 'score' key and it's a number < threshold, yield it
        score = data.get('score')
        if score is not None and isinstance(score, (int, float)) and score < threshold:
            yield current_path, data

        # Special handling for known containers
        if 'history' in data and isinstance(data['history'], list):
            for idx, item in enumerate(data['history']):
                yield from find_blocks_with_low_score(item, threshold, f"{current_path}.history[{idx}]")
        if 'metrics' in data and isinstance(data['metrics'], dict):
            metrics = data['metrics']
            if 'optimization_progress' in metrics and isinstance(metrics['optimization_progress'], dict):
                for iter_key, block in metrics['optimization_progress'].items():
                    yield from find_blocks_with_low_score(block, threshold, f"{current_path}.metrics.optimization_progress.{iter_key}")
        # For any other dict, recursively examine its values
        for key, value in data.items():
            yield from find_blocks_with_low_score(value, threshold, f"{current_path}.{key}" if current_path else key)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            yield from find_blocks_with_low_score(item, threshold, f"{current_path}[{idx}]")

def search_json_files(directory, threshold):
    directory = Path(directory)
    if not directory.is_dir():
        print(f"Error: {directory} is not a valid directory.")
        return

    json_files = list(directory.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in {directory}")
        return

    found_any = False
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not parse {json_file}: {e}")
            continue

        for path, block in find_blocks_with_low_score(data, threshold):
            if not found_any:
                print(f"Blocks with score < {threshold}:\n")
                found_any = True
            print(f"File: {json_file}")
            print(f"Path: {path}")
            iteration = block.get('iteration', 'N/A')
            print(f"Iteration: {iteration}")
            print("Block:")
            print(json.dumps(block, indent=2))
            print("-" * 60)

    if not found_any:
        print(f"No blocks with score < {threshold} found.")

if __name__ == "__main__":
    # Usage: python script.py [directory] [threshold]
    # threshold is optional; default = -0.3896 (your current value)
    if len(sys.argv) > 2:
        target_dir = sys.argv[1]
        try:
            threshold = float(sys.argv[2])
        except ValueError:
            print("Threshold must be a number. Using default -0.3896.")
            threshold = 0.04
    elif len(sys.argv) == 2:
        # Second argument might be directory, or threshold if only one argument?
        # We'll be safe: if the argument looks like a number, treat as threshold and use current dir
        try:
            threshold = float(sys.argv[1])
            target_dir = "."
        except ValueError:
            target_dir = sys.argv[1]
            threshold = 0.04   # default
    else:
        target_dir = "."
        threshold = 0.04

    search_json_files(target_dir, threshold)