import json
import sys
from pathlib import Path

def find_highest_win(data, current_path=""):
    """
    Recursively traverse JSON data and return:
        (win_value, path, block_dict)
    for the block with the highest win value encountered.
    A block is a dict with 'score' and 'contributions' where
    'contributions' is a dict containing a 'matrix' list.
    The win is the second‑last element of the second‑last list in 'matrix'.
    """
    best_win = None
    best_path = None
    best_block = None

    if isinstance(data, dict):
        # Check if this dict is a block
        contributions = data.get('contributions')
        if contributions and isinstance(contributions, dict):
            matrix = contributions.get('matrix')
            if matrix and isinstance(matrix, list) and len(matrix) >= 2:
                second_last = matrix[-2]
                if isinstance(second_last, list) and len(second_last) >= 2:
                    win = second_last[-2]  # second‑last element of second‑last list
                    best_win = win
                    best_path = current_path
                    best_block = data

        # Recurse into all values
        for key, value in data.items():
            new_path = f"{current_path}.{key}" if current_path else key
            sub_win, sub_path, sub_block = find_highest_win(value, new_path)
            if sub_win is not None and (best_win is None or sub_win > best_win):
                best_win, best_path, best_block = sub_win, sub_path, sub_block

    elif isinstance(data, list):
        for idx, item in enumerate(data):
            new_path = f"{current_path}[{idx}]"
            sub_win, sub_path, sub_block = find_highest_win(item, new_path)
            if sub_win is not None and (best_win is None or sub_win > best_win):
                best_win, best_path, best_block = sub_win, sub_path, sub_block

    return best_win, best_path, best_block


def search_json_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        print(f"Error: {directory} is not a valid directory.", file=sys.stderr)
        return

    json_files = list(directory.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in {directory}", file=sys.stderr)
        return

    overall_best_win = None
    overall_best_path = None
    overall_best_block = None
    overall_best_file = None

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not parse {json_file}: {e}", file=sys.stderr)
            continue

        win, path, block = find_highest_win(data)
        if win is not None and (overall_best_win is None or win > overall_best_win):
            overall_best_win = win
            overall_best_path = path
            overall_best_block = block
            overall_best_file = json_file

    if overall_best_win is None:
        print("No blocks with a 'matrix' key found.", file=sys.stderr)
        return

    # Output the result
    result = {
        'win': overall_best_win,
        'file': str(overall_best_file),
        'path': overall_best_path,
        'score': overall_best_block.get('score', None),
        'iteration': overall_best_block.get('iteration', None),
        'raw_alpha': overall_best_block.get('contributions', {}).get('_raw_alpha', None)
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_highest_win.py <directory>", file=sys.stderr)
        sys.exit(1)

    target_dir = sys.argv[1]
    search_json_files(target_dir)