import os
import json
import sys
from pathlib import Path

def find_blocks_with_low_score(data, threshold, current_path=""):
    """Recursively traverse JSON data and yield blocks with score < threshold."""
    results = []
    
    if isinstance(data, dict):
        score = data.get('score')
        if score is not None and isinstance(score, (int, float)):
            if score < threshold:
                results.append((current_path, data))
            else:
                # Debug: Print why it's not included
                #print(f"DEBUG: Score {score} at {current_path} is NOT < {threshold}", file=sys.stderr)
                pass
        
        # Recursively traverse all values
        for key, value in data.items():
            new_path = f"{current_path}.{key}" if current_path else key
            results.extend(find_blocks_with_low_score(value, threshold, new_path))
            
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            new_path = f"{current_path}[{idx}]"
            results.extend(find_blocks_with_low_score(item, threshold, new_path))
    
    return results

def search_json_files(directory, threshold):
    directory = Path(directory)
    if not directory.is_dir():
        print(f"Error: {directory} is not a valid directory.", file=sys.stderr)
        return

    json_files = list(directory.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in {directory}", file=sys.stderr)
        return

    all_blocks = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not parse {json_file}: {e}", file=sys.stderr)
            continue

        blocks = find_blocks_with_low_score(data, threshold)
        for path, block in blocks:
            all_blocks.append({
                'file': json_file,
                'path': path,
                'block': block,
                'score': block.get('score')
            })

    if not all_blocks:
        print(f"No blocks with score < {threshold} found.", file=sys.stderr)
        return

    # Sort by score (lowest first)
    all_blocks.sort(key=lambda x: x['score'])
    
    # Output only the top result (lowest score)
    best = all_blocks[0]
    print(json.dumps({
        'score': best['score'],
        'raw_alpha': best['block'].get('contributions', {}).get('_raw_alpha', 'N/A'),
        'file': str(best['file']),
        'iteration': best['block'].get('iteration', 'N/A')
    }, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python find_low_score.py <directory> <threshold>", file=sys.stderr)
        sys.exit(1)
    
    target_dir = sys.argv[1]
    threshold = float(sys.argv[2])
    search_json_files(target_dir, threshold)