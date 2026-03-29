from dataclasses import dataclass
import glob, json, os, time
from datetime import datetime
from pathlib import Path
from typing import Optional
import subprocess

@dataclass
class HyperoptHelper:
    """
    Dataclass to hold metrics for a trading strategy.
    
    Attributes:
    -----------
    alpha_dirs: str
        Glob pattern for finding optimization result directories
    model_dirs: str
        Directory for model files
    initial_alpha: float
        Initial alpha value for comparison (lower is better)
    initial_burke: float
        Initial burke value for comparison (higher is better)
    initial_entropy: float
        Initial entropy value for comparison (lower is better)
    save_intermediate: bool
        Whether to save intermediate results
    """
    alpha_dirs: str = "user_data/optimization_results/*"
    model_dirs: str = "core/rl/pkls/"
    initial_alpha: float = 1  # Lower is better
    initial_burke: float = -1  # Higher is better
    initial_entropy: float = 1  # Lower is better
    save_intermediate: bool = True

    def _resolve_metrics_json_path(self, directory: str) -> Optional[str]:
        """
        Resolve optimization_results.json for a given results directory.
        Supports both direct files and nested alpha_*/optimization_results.json layouts.
        """
        direct_path = os.path.join(directory, 'optimization_results.json')
        if os.path.exists(direct_path):
            return direct_path

        nested_candidates = glob.glob(os.path.join(directory, 'alpha_*', 'optimization_results.json'))
        if nested_candidates:
            # Prefer newest nested result if multiple are present.
            nested_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return nested_candidates[0]

        return None

    def fetch_latest_metrics(self, include_candidates=False):
        """
        Enhanced update with score tracking
        
        Parameters:
        -----------
        include_candidates: bool
            Whether to include candidate models in the search
            
        Returns:
        --------
        dict: Latest metrics or empty dict if none found
        """
        # Define patterns to search
        patterns = [self.alpha_dirs]
        if include_candidates:
            patterns.append("user_data/optimization_results/candidate_*")
            patterns.append("user_data/optimization_results/intermediate_*")
            patterns.append("user_data/optimization_results/alpha_*")
        
        # Get all directories matching the patterns
        all_dirs = []
        for pattern in patterns:
            all_dirs.extend(glob.glob(pattern))
        
        if not all_dirs:
            print("No valid alpha directories found. Proceeding with initialization.")
            return {
                'alpha': self.initial_alpha,
                'metrics': {
                    'burke': self.initial_burke,
                    'entropy': self.initial_entropy
                }
            }

        # Extract timestamps and sort directories
        def extract_timestamp(dir_name):
            parts = dir_name.split('_')
            if len(parts) >= 2:
                return f"{parts[-2]}_{parts[-1]}"  # Combines date and time parts
            return "00000000_000000"  # Default for invalid format
            
        sorted_dirs = sorted(all_dirs, key=extract_timestamp, reverse=True)
        if not sorted_dirs:
            return {
                'alpha': self.initial_alpha,
                'metrics': {
                    'burke': self.initial_burke,
                    'entropy': self.initial_entropy
                }
            }
            
        for latest_dir in sorted_dirs:
            json_path = self._resolve_metrics_json_path(latest_dir)
            if json_path:
                print(f"Using latest directory: {latest_dir}")
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    return data or {}
            else:
                print(f"File not found: {latest_dir}/optimization_results.json (and no nested alpha_* match)")

        # Fallback to model snapshot if no optimization results file found anywhere.
        try:
            with open(os.path.join(self.model_dirs, 'optimization_results.json'), 'r') as f:
                data = json.load(f)
                return data or {}
        except FileNotFoundError:
            print(f"File not found: {os.path.join(self.model_dirs, 'optimization_results.json')}")
            return {}

    def initialize_best_metrics_from_latest(self, include_candidates=False):
        """
        Initialize best metrics from the latest optimization results.
        
        Parameters:
        -----------
        include_candidates: bool
            Whether to include candidate models in the search
            
        Returns:
        --------
        dict: Best metrics from latest results or default values
        """
        try:
            latest_metrics = self.fetch_latest_metrics(include_candidates)
            
            # Extract metrics with fallbacks to initial values
            latest_signed_alpha = latest_metrics.get('alpha', self.initial_alpha)
            
            # Handle potential missing metrics key
            metrics = latest_metrics.get('metrics', {})
            if not metrics:
                print("No metrics found in latest results, using initial values")
                return {
                    'best_alpha': self.initial_alpha,
                    'best_burke': self.initial_burke,
                    'best_entropy': self.initial_entropy
                }
                
            latest_burke = metrics.get('burke', self.initial_burke)
            latest_entropy = metrics.get('entropy', self.initial_entropy)
            
            latest_best_metrics = {
                'best_alpha': latest_signed_alpha,
                'best_burke': latest_burke,
                'best_entropy': latest_entropy
            }
            
            print(f"Initialized metrics - Alpha: {latest_signed_alpha:.6f}, Burke: {latest_burke:.6f}, Entropy: {latest_entropy:.6f}")
            return latest_best_metrics
            
        except Exception as e:
            print(f"Error initializing metrics: {e}")
            print("Using default initial values")
            return {
                'best_alpha': self.initial_alpha,
                'best_burke': self.initial_burke,
                'best_entropy': self.initial_entropy
            }
    
    def save_backup_metrics(self, include_candidates=False):
        """
        Save backup of the latest metrics
        
        Parameters:
        -----------
        include_candidates: bool
            Whether to include candidate models in the search
        """
        try:
            json_path = os.path.join(self.model_dirs, 'optimization_results.json')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            
            # Fetch and save metrics
            metrics_data = self.fetch_latest_metrics(include_candidates)
            with open(json_path, 'w+') as f:
                json.dump(metrics_data, f, indent=4)
                f.write('\n')  # Add newline at end of file
                
            print(f"Backup metrics saved to: {json_path}")
            
            # Also save a timestamped copy for history
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if 'datetime' in globals() else time.strftime("%Y%m%d_%H%M%S")
            history_path = os.path.join(self.model_dirs, f'optimization_results_{timestamp}.json')
            with open(history_path, 'w+') as f:
                json.dump(metrics_data, f, indent=4)
                f.write('\n')
                
            print(f"Historical backup saved to: {history_path}")
            
        except Exception as e:
            print(f"Error saving backup metrics: {e}")

    def get_all_saved_models(self, include_candidates=True):
        """
        Get a list of all saved model directories
        
        Parameters:
        -----------
        include_candidates: bool
            Whether to include candidate models in the search
            
        Returns:
        --------
        list: List of model directory paths
        """
        patterns = [self.alpha_dirs]
        if include_candidates:
            patterns.append("user_data/optimization_results/candidate_*")
            patterns.append("user_data/optimization_results/intermediate_*")
            patterns.append("user_data/optimization_results/alpha_*")
        
        all_dirs = []
        for pattern in patterns:
            all_dirs.extend(glob.glob(pattern))
            
        # Sort by timestamp (newest first)
        def extract_timestamp(dir_name):
            parts = dir_name.split('_')
            if len(parts) >= 2:
                return f"{parts[-2]}_{parts[-1]}"
            return "00000000_000000"
            
        return sorted(all_dirs, key=extract_timestamp, reverse=True)

    def find_production_model(self,
                             alpha_threshold: float = 0,
                             burke_threshold: float = 0.5,
                             entropy_threshold: float = 0.3,
                             alpha_weight: float = 0.3,
                             entropy_weight: float = 0.4,
                             burke_weight: float = 0.2) -> Optional[tuple[str, str]]:
        """
        Find the best model suitable for production deployment based on strict criteria.
        
        Parameters:
        -----------
        alpha_threshold: float
            Maximum acceptable alpha value (lower is better)
        burke_threshold: float
            Minimum acceptable burke ratio (higher is better)
        entropy_threshold: float
            Maximum acceptable entropy value (lower is better)
        alpha_weight: float
            Weight for alpha in composite score (0-1)
        entropy_weight: float
            Weight for entropy in composite score (0-1)
        burke_weight: float
            Weight for burke in composite score (0-1)
            
        Returns:
        --------
        Optional[tuple[str, str]]: Tuple containing:
            - Path to the best production-ready model directory
            - Timestamp string (YYYYMMDD_HHMMSS)
            Returns None if no models meet criteria
        """
        best_score = float('inf')
        best_model_dir = None
        best_timestamp = None
        
        # Only search in main optimization results, not candidates
        all_dirs = glob.glob(self.alpha_dirs)
        # All candidate and main
        '''
        patterns = [
            self.alpha_dirs,
            "user_data/optimization_results/candidate_*",
            "user_data/optimization_results/intermediate_alpha*",
            "user_data/optimization_results/intermediate_run*"
        ]
        
        all_dirs = []
        for pattern in patterns:
            all_dirs.extend(glob.glob(pattern))

        '''

        if not all_dirs:
            print("No models found in optimization directory")
            return None
        
        # Sort by timestamp (newest first)
        def extract_timestamp(dir_name):
            parts = dir_name.split('_')
            if len(parts) >= 2:
                return f"{parts[-2]}_{parts[-1]}"
            return "00000000_000000"
        
        sorted_dirs = sorted(all_dirs, key=extract_timestamp, reverse=True)
        
        for model_dir in sorted_dirs:
            json_path = os.path.join(model_dir, 'optimization_results.json')
            if not os.path.exists(json_path):
                continue
            
            try:
                with open(json_path, 'r') as f:
                    metrics_data = json.load(f)
                
                current_alpha = metrics_data.get('alpha', float('inf'))
                metrics = metrics_data.get('metrics', {})
                current_burke = metrics.get('burke', -float('inf'))
                current_entropy = metrics.get('entropy', float('inf'))
                
                # Production criteria check
                if (current_alpha >= alpha_threshold or
                    current_burke <= burke_threshold or
                    current_entropy >= entropy_threshold):
                    continue
                
                # Calculate composite score (lower is better)
                current_score = (
                    (current_alpha * alpha_weight) +
                    (current_entropy * entropy_weight) -
                    (current_burke * burke_weight)
                )
                
                if current_score < best_score:
                    best_score = current_score
                    best_model_dir = model_dir
                    # Extract timestamp from directory name
                    timestamp_parts = model_dir.split('_')
                    if len(timestamp_parts) >= 2:
                        best_timestamp = f"{timestamp_parts[-2]}_{timestamp_parts[-1]}"
                    print(f"New best production model found:")
                    print(f"  Directory: {model_dir}")
                    print(f"  Timestamp: {best_timestamp}")
                    print(f"  Alpha: {current_alpha:.6f}")
                    print(f"  Burke: {current_burke:.6f}")
                    print(f"  Entropy: {current_entropy:.6f}")
                    print(f"  Score: {current_score:.6f}")
                
            except Exception as e:
                print(f"Error processing {json_path}: {e}")
                continue
        
        if best_model_dir is None:
            print("No models meet production criteria")
            return None
        
        return best_model_dir, best_timestamp

def print_all_metrics():
    base_dir = "user_data/optimization_results"
    patterns = ["candidate*", "intermediate*","alpha_*"]
    
    print("\nSearching for metrics in all optimization directories...")
    print("=" * 80)
    
    for pattern in patterns:
        search_path = os.path.join(base_dir, pattern)
        directories = glob.glob(search_path)
        for directory in directories:
            #print(directory)
            # Find alpha subdirectory
            alpha_dirs = [d for d in os.listdir(directory) if d.startswith('alpha')]
            if not alpha_dirs:
                continue
                
            alpha_dir = alpha_dirs[0]
            json_path = os.path.join(directory, alpha_dir, 'optimization_results.json')
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        opt_score = data.get('optimization_score', 'N/A')
                        alpha = data.get('alpha', 'N/A')
                        metrics = data.get('metrics', {})
                        burke = metrics.get('burke', 'N/A')
                        entropy = metrics.get('entropy', 'N/A')
                        if not all(isinstance(v, (int, float)) for v in [alpha, burke, entropy]):
                            ear = None
                        elif abs(entropy) < 1e-12:
                            ear = None
                        else:
                            ear = alpha * burke / entropy
                        win_rate = metrics.get('win_rate', 'N/A')
                        
                        if ear is not None and ear < 200 and alpha < -0.02 and entropy < 1 and opt_score <= 0 and win_rate >= 0.6:
                        #if entropy <= -0.73 and win_rate > 0.3:
                            print(f"\nDirectory: {os.path.basename(directory)}/{alpha_dir}")
                            print(f"Alpha:   {alpha:.6f}")
                            print(f"Burke:   {burke:.6f}")
                            print(f"Entropy: {entropy:.6f}")
                            print(f"Optimization Score: {opt_score:.6f}")
                            print(f"EAR: {ear:.6f}")
                            print(f"Win Rate: {win_rate:.6f}")
                            print("-" * 80)
                except Exception as e:
                    print(f"Error reading {json_path}: {str(e)}")

if __name__ == "__main__":
    print_all_metrics()
    #HyperoptHelper().save_backup_metrics()
