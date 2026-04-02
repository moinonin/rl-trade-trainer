import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional
import pickle
import logging
import json
import pandas as pd
from datetime import datetime
from core.metrics.nmatrix_hyperopt import calculate_nmatrix, save_optimization_result

# ANSI color codes for colorful terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
BLUE = '\033[94m'
BOLD = '\033[1m'
END = '\033[0m'

@dataclass
class BidsAgent:
    ACTIONS = ('go_long', 'go_short', 'do_nothing')  # Immutable action set
    SPECIAL_RAW_ALPHA_THRESHOLD = -0.1
    
    def __init__(self, model_dir: Optional[str] = None):
        # Learning parameters
        self.config_path = Path(__file__).resolve().parents[1] / "config" / "agent_config.json"
        with open(self.config_path) as f:
            d = json.load(f)
        self.config = dict(d)
        self.learning_rate = d.get('learning_rate')
        self.discount_factor = d.get('discount_factor')
        self.epsilon_start = d.get('epsilon_start')
        self.epsilon = self.epsilon_start
        self.min_epsilon = d.get('min_epsilon')
        self.decay_rate = d.get('decay_rate')
        self.switching_cost = d.get('switching_cost', 0.0)
        self.hold_reward = d.get('hold_reward', 0.0)
        self.state_rebalance = d.get('state_rebalance', {})

        # Model persistence setup
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent
        self.model_dir.mkdir(exist_ok=True)
        
        # Q-table and state management
        self._init_model_structures()
        self.load_model()

    def _init_model_structures(self):
        """Initialize core data structures with proper typing"""
        base_shape = (1024, len(self.ACTIONS))
        # Double Q-learning tables (policy table is derived from these two).
        self.q_table_a = np.zeros(base_shape)
        self.q_table_b = np.zeros(base_shape)
        self.q_table = np.zeros(base_shape)
        self.state_to_index: Dict[Tuple, int] = {}
        self.episode_transitions = []

    def _canonicalize_state(self, state: Tuple) -> Tuple:
        """Normalize state values to reduce sparsity in the tabular state space."""
        if not isinstance(state, tuple):
            state = tuple(state)

        normalized: List[Any] = []
        for value in state:
            if isinstance(value, (np.floating, float)):
                normalized.append(round(float(value), 4))
            elif isinstance(value, (np.integer, int)):
                normalized.append(int(value))
            elif isinstance(value, (np.bool_, bool)):
                normalized.append(int(value))
            else:
                normalized.append(value)
        return tuple(normalized)

    def _refresh_policy_table(self):
        """Policy/read table used by action selection and diagnostics."""
        self.q_table = self.q_table_a + self.q_table_b

    def _ensure_capacity(self, min_size: int):
        """Grow Q-tables without repeating values."""
        if min_size <= self.q_table_a.shape[0]:
            return

        new_size = self.q_table_a.shape[0]
        while new_size < min_size:
            new_size *= 2

        new_a = np.zeros((new_size, len(self.ACTIONS)))
        new_b = np.zeros((new_size, len(self.ACTIONS)))
        old_size = self.q_table_a.shape[0]
        new_a[:old_size] = self.q_table_a
        new_b[:old_size] = self.q_table_b
        self.q_table_a = new_a
        self.q_table_b = new_b
        self._refresh_policy_table()
        logging.info(f"{CYAN}Expanded Q-table to {new_size} states{END}")

    def decay_epsilon(self):
        """Proper epsilon decay mechanism"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

    def load_model(self):
        """Load model components safely"""
        try:
            pkls_dir = self.model_dir/'pkls'
            q_table_a_path = pkls_dir/'q_table_a.pkl'
            q_table_b_path = pkls_dir/'q_table_b.pkl'
            q_table_path = pkls_dir/'q_table.pkl'

            if q_table_a_path.exists() and q_table_b_path.exists():
                with open(q_table_a_path, 'rb') as f:
                    self.q_table_a = pickle.load(f)
                with open(q_table_b_path, 'rb') as f:
                    self.q_table_b = pickle.load(f)
            elif q_table_path.exists():
                # Backward compatibility: old single-table model.
                with open(q_table_path, 'rb') as f:
                    legacy_q = pickle.load(f)
                self.q_table_a = legacy_q / 2.0
                self.q_table_b = legacy_q / 2.0
            self._refresh_policy_table()

            # Try to load state mappings from pkls directory first
            state_path_pkls = pkls_dir/'state_to_index.pkl'
            state_path_root = self.model_dir/'state_to_index.pkl'
            
            # First try pkls directory, then fall back to root directory
            if state_path_pkls.exists():
                with open(state_path_pkls, 'rb') as f:
                    self.state_to_index = pickle.load(f)
                    logging.info("Loaded state mappings from pkls directory")
            elif state_path_root.exists():
                with open(state_path_root, 'rb') as f:
                    self.state_to_index = pickle.load(f)
                    logging.info("Loaded state mappings from root directory")
            
            logging.info(f"{GREEN}Loaded model with {len(self.state_to_index)} known states{END}")
            
        except Exception as e:
            logging.warning(f"{YELLOW}Model loading failed: {e} - Initializing new model{END}")
            self._initialize_new_model()

    def _initialize_new_model(self):
        """Bootstrap initial model state"""
        try:
            # Seed with basic market states
            base_states = [
                ('trend_up', 1.0, 0),
                ('trend_down', -1.0, 1),
                ('neutral', 0.0, 0)
            ]
            
            for state in base_states:
                self._register_state(state)
            
            # Initialize with small positive values to encourage exploration
            self.q_table_a = np.random.uniform(0, 0.05, self.q_table_a.shape)
            self.q_table_b = np.random.uniform(0, 0.05, self.q_table_b.shape)
            self._refresh_policy_table()
            self.save_model()
            
            print(f"{GREEN}✓ Initialized new model with {len(self.state_to_index)} base states{END}")
            
        except Exception as e:
            logging.error(f"{RED}Model initialization failed: {e}{END}")
            raise RuntimeError("Could not initialize new model") from e

    def _register_state(self, state: Tuple) -> int:
        """Safely register new states with Q-table expansion"""
        state = self._canonicalize_state(state)
        if state not in self.state_to_index:
            new_index = len(self.state_to_index)
            self._ensure_capacity(new_index + 1)
            self.state_to_index[state] = new_index
        return self.state_to_index[state]

    def save_model(self):
        """Persist model state safely"""
        try:
            pkls_dir = self.model_dir / 'pkls'
            self._write_model_artifacts(pkls_dir)
            # Also save a backup in the model directory for backward compatibility
            with open(self.model_dir/'state_to_index.pkl', 'wb') as f:
                pickle.dump(self.state_to_index, f)
            
            logging.info(f"{GREEN}Saved model with {len(self.state_to_index)} states{END}")
            
        except Exception as e:
            logging.error(f"{RED}Model save failed: {e}{END}")
            raise RuntimeError("Model persistence failed") from e

    def _write_model_artifacts(self, pkls_dir: Path):
        """Write model artifacts to a target pkls directory."""
        pkls_dir.mkdir(parents=True, exist_ok=True)
        self._refresh_policy_table()

        with open(pkls_dir / 'q_table.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)
        with open(pkls_dir / 'q_table_a.pkl', 'wb') as f:
            pickle.dump(self.q_table_a, f)
        with open(pkls_dir / 'q_table_b.pkl', 'wb') as f:
            pickle.dump(self.q_table_b, f)
        with open(pkls_dir / 'state_to_index.pkl', 'wb') as f:
            pickle.dump(self.state_to_index, f)
        with open(pkls_dir / 'episode_transitions.pkl', 'wb') as f:
            pickle.dump(self.episode_transitions, f)

        self._save_hyperparams_snapshot(pkls_dir)

    def _copy_json_artifacts(self, source_dir: Path, target_dir: Path):
        """Mirror JSON metadata files from a save directory into its pkls directory."""
        if not source_dir.exists():
            return

        import shutil

        target_dir.mkdir(parents=True, exist_ok=True)
        for json_file in source_dir.glob("*.json"):
            shutil.copy2(json_file, target_dir / json_file.name)

    def export_model_artifacts(self, target_dir, metrics_json_path: Optional[Path] = None):
        """Export model artifacts into a dedicated directory with a pkls subdir."""
        export_dir = Path(target_dir)
        pkls_dir = export_dir / 'pkls'
        self._write_model_artifacts(pkls_dir)

        self._copy_json_artifacts(export_dir, pkls_dir)

        if metrics_json_path:
            source_json_path = Path(metrics_json_path)
            if source_json_path.exists() and source_json_path.parent != export_dir:
                import shutil
                shutil.copy2(source_json_path, pkls_dir / source_json_path.name)

        logging.info(f"{GREEN}Exported model artifacts to {export_dir}{END}")

    def _current_hyperparams(self) -> Dict[str, float]:
        payload = dict(self.config)
        payload.update({
            "learning_rate": float(self.learning_rate),
            "discount_factor": float(self.discount_factor),
            "epsilon_start": float(self.epsilon_start),
            "decay_rate": float(self.decay_rate),
            "min_epsilon": float(self.min_epsilon),
            "switching_cost": float(self.switching_cost),
            "hold_reward": float(self.hold_reward),
        })
        return payload

    def _save_hyperparams_snapshot(self, pkls_dir: Path):
        """
        Save latest hyperparameters in pkls and append periodic history snapshots.
        """
        payload = self._current_hyperparams()
        payload["updated_at"] = datetime.now().isoformat()

        latest_path = pkls_dir / "agent_config.json"
        history_path = pkls_dir / "agent_config_history.jsonl"

        with open(latest_path, "w") as f:
            json.dump(payload, f, indent=2)

        with open(history_path, "a") as f:
            f.write(json.dumps(payload) + "\n")

    def persist_params_if_best_raw_alpha(self, raw_alpha: Optional[float]) -> bool:
        """
        Persist agent params to config when a new lowest _raw_alpha is observed.
        Returns True when an update is written.
        """
        if raw_alpha is None:
            return False

        pkls_dir = self.model_dir / "pkls"
        pkls_dir.mkdir(parents=True, exist_ok=True)
        tracker_path = pkls_dir / "best_raw_alpha.json"

        previous_best = self._load_or_bootstrap_best_raw_alpha(pkls_dir, tracker_path)
        candidate_alpha = float(raw_alpha)

        if candidate_alpha >= previous_best:
            return False

        params_payload = self._current_hyperparams()
        with open(self.config_path, "w") as f:
            json.dump(params_payload, f, indent=2)

        tracker_payload = {
            "best_raw_alpha": candidate_alpha,
            "updated_at": datetime.now().isoformat(),
            "agent_config_path": str(self.config_path),
            "params": params_payload,
        }
        with open(tracker_path, "w") as f:
            json.dump(tracker_payload, f, indent=2)

        self._save_hyperparams_snapshot(pkls_dir)
        logging.info(f"{GREEN}Updated agent config from new best _raw_alpha: {candidate_alpha:.10f}{END}")
        return True

    def _load_or_bootstrap_best_raw_alpha(self, pkls_dir: Path, tracker_path: Path) -> float:
        """Read tracked best raw alpha, or bootstrap it from existing optimization artifacts."""
        previous_best = float("inf")
        if tracker_path.exists():
            try:
                with open(tracker_path, "r") as f:
                    previous_best = float(json.load(f).get("best_raw_alpha", previous_best))
                return previous_best
            except Exception:
                previous_best = float("inf")

        def walk_raw_alpha(obj) -> List[float]:
            values: List[float] = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ("_raw_alpha", "alpha") and isinstance(v, (int, float)):
                        values.append(float(v))
                    values.extend(walk_raw_alpha(v))
            elif isinstance(obj, list):
                for item in obj:
                    values.extend(walk_raw_alpha(item))
            return values

        # Search recursively in user_data for all optimization results
        base_search_dir = Path("user_data/optimization_results")
        if base_search_dir.exists():
            for fp in base_search_dir.rglob("optimization_results.json"):
                try:
                    with open(fp, "r") as f:
                        data = json.load(f)
                    values = walk_raw_alpha(data)
                    if values:
                        previous_best = min(previous_best, min(values))
                except Exception:
                    continue

        # Also search in the local pkls directory
        for fp in pkls_dir.glob("optimization_results*.json"):
            try:
                with open(fp, "r") as f:
                    data = json.load(f)
                values = walk_raw_alpha(data)
                if values:
                    previous_best = min(previous_best, min(values))
            except Exception:
                continue

        return previous_best

    def get_state_index(self, state: Tuple) -> int:
        """Get index for state using tuple directly as key"""
        return self._register_state(state)
        
    def expand_q_table(self):
        """Double the size of Q-table when needed"""
        self._ensure_capacity(self.q_table.shape[0] * 2)
        
    def select_action(self, state: Tuple, epsilon: float = None) -> str:
        """Epsilon-greedy action selection with optional epsilon override"""
        # Use provided epsilon if given, otherwise use the agent's epsilon
        use_epsilon = epsilon if epsilon is not None else self.epsilon
        
        if np.random.random() < use_epsilon:
            return np.random.choice(self.ACTIONS)
        
        state_idx = self._register_state(state)
        return self.ACTIONS[np.argmax(self.q_table[state_idx])]
        
    def strategic_action_selection(self, base_state: Tuple) -> Tuple[str, int]:
        """Select the best action across long/short state variants."""
        long_state = base_state + (0,)
        short_state = base_state + (1,)

        long_q = self.q_table[self._register_state(long_state)]
        short_q = self.q_table[self._register_state(short_state)]

        combined_q = np.vstack((long_q, short_q))
        state_choice, action_idx = np.unravel_index(np.argmax(combined_q), combined_q.shape)
        chosen_is_short = int(state_choice)

        return self.ACTIONS[int(action_idx)], chosen_is_short
        
    def update(self, state: Tuple, action: str, reward: float, next_state: Tuple):
        """Double Q-learning update with bounds checking"""
        try:
            action_idx = self.ACTIONS.index(action)
            state_idx = self._register_state(state)
            next_idx = self._register_state(next_state)

            if np.random.random() < 0.5:
                next_action = int(np.argmax(self.q_table_a[next_idx]))
                target = reward + self.discount_factor * self.q_table_b[next_idx, next_action]
                current_q = self.q_table_a[state_idx, action_idx]
                self.q_table_a[state_idx, action_idx] = current_q + self.learning_rate * (target - current_q)
            else:
                next_action = int(np.argmax(self.q_table_b[next_idx]))
                target = reward + self.discount_factor * self.q_table_a[next_idx, next_action]
                current_q = self.q_table_b[state_idx, action_idx]
                self.q_table_b[state_idx, action_idx] = current_q + self.learning_rate * (target - current_q)

            self.q_table[state_idx] = self.q_table_a[state_idx] + self.q_table_b[state_idx]
            
        except ValueError as e:
            logging.error(f"{RED}Invalid action '{action}' in update: {e}{END}")
        except Exception as e:
            logging.error(f"{RED}Update failed for state {state}: {e}{END}")

    def save_model_with_alpha(self, alpha_dir: Optional[Path] = None):
        """Enhanced model saving with proper type handling"""
        try:
            alpha_dir = alpha_dir or Path("optimized_models")
            alpha_dir.mkdir(parents=True, exist_ok=True)
            
            # Save with versioning
            version = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            model_dir = alpha_dir / f"model_{version}"
            model_dir.mkdir()
            
            # Create pkls directory in the model directory
            pkls_dir = model_dir / 'pkls'
            pkls_dir.mkdir(exist_ok=True)
            
            # Save components
            self._refresh_policy_table()
            np.save(model_dir/'q_table.npy', self.q_table)
            np.save(model_dir/'q_table_a.npy', self.q_table_a)
            np.save(model_dir/'q_table_b.npy', self.q_table_b)
            
            # Save state mappings to pkls directory
            with open(pkls_dir/'state_to_index.pkl', 'wb') as f:
                pickle.dump(self.state_to_index, f)
            
            # Also save in the model directory for backward compatibility
            with open(model_dir/'state_to_index.pkl', 'wb') as f:
                pickle.dump(self.state_to_index, f)
                
            # Save metadata with native types
            metadata = {
                "q_table_shape": [int(dim) for dim in self.q_table.shape],
                "algorithm": "double_q_learning",
                "num_states": int(len(self.state_to_index)),
                "learning_rate": float(self.learning_rate),
                "discount_factor": float(self.discount_factor),
                "epsilon": float(self.epsilon),
                "num_transitions": int(len(self.episode_transitions))
            }
            
            with open(model_dir/'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logging.info(f"{GREEN}Saved alpha model to {model_dir}{END}")
            
        except Exception as e:
            logging.error(f"{RED}Alpha save failed: {e}{END}")
            raise
            
    def save_model_with_metrics(self, alpha_dir=None):
        """Save model when metrics condition is met"""
        global_save_error = None
        try:
            # Save to regular location
            self.save_model()
        except Exception as e:
            global_save_error = e
            logging.error(f"{RED}Global model save failed; continuing artifact export: {e}{END}")

        try:
            # If alpha_dir is provided, also save there
            if alpha_dir:
                if isinstance(alpha_dir, tuple) and len(alpha_dir) >= 1:
                    alpha_dir = alpha_dir[0]

                alpha_path = Path(alpha_dir)
                source_json_path = alpha_path / 'optimization_results.json'
                self.export_model_artifacts(
                    alpha_path,
                    metrics_json_path=source_json_path if source_json_path.exists() else None
                )

                if source_json_path.exists():
                    model_pkls_dir = self.model_dir / 'pkls'
                    try:
                        self._copy_json_artifacts(alpha_path, model_pkls_dir)
                        logging.info(f"{CYAN}Copied JSON metadata to model's pkls directory: {model_pkls_dir}{END}")
                    except Exception as json_copy_error:
                        logging.error(f"{RED}Failed to copy JSON metadata: {json_copy_error}{END}")

        except Exception as e:
            logging.error(f"{RED}Alpha model save failed: {e}{END}")
            return

        if global_save_error is None:
            return

    def add_transition(self, state: Tuple, action: str, reward: float, next_state: Tuple):
        """Store a new transition"""
        transition = {
            'state': self._canonicalize_state(state),
            'action': action,
            'reward': reward,
            'next_state': self._canonicalize_state(next_state),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        self.episode_transitions.append(transition)
        
        # Automatically limit the number of stored transitions to prevent memory issues
        self._limit_transitions()
        
    def _limit_transitions(self, max_transitions: int = 10000):
        """Limit the number of stored transitions to prevent memory issues"""
        if len(self.episode_transitions) > max_transitions:
            # Keep only the most recent transitions
            self.episode_transitions = self.episode_transitions[-max_transitions:]
            logging.info(f"{YELLOW}Limited transitions to {max_transitions} most recent entries{END}")
            
    def clear_transitions(self):
        """Clear all stored transitions"""
        transitions_count = len(self.episode_transitions)
        self.episode_transitions = []
        logging.info(f"{CYAN}Cleared {transitions_count} stored transitions{END}")

    def replay(self, batch_size: int = 32, replay_updates: int = 1) -> int:
        """Sample transitions and perform off-policy updates."""
        num_transitions = len(self.episode_transitions)
        if num_transitions == 0:
            return 0

        sample_size = min(batch_size, num_transitions)
        updates_applied = 0
        for _ in range(max(1, replay_updates)):
            sample_indices = np.random.choice(num_transitions, size=sample_size, replace=False)
            for idx in sample_indices:
                transition = self.episode_transitions[int(idx)]
                self.update(
                    transition['state'],
                    transition['action'],
                    transition['reward'],
                    transition['next_state']
                )
                updates_applied += 1
        return updates_applied

    def find_optimal_model(self, 
                          include_candidates: bool = True, 
                          print_progress: bool = True,
                          alpha_weight: float = 0.3,
                          entropy_weight: float = 0.4,
                          burke_weight: float = 0.3) -> Optional[Path]:
        """
        Locate best-performing model based on multiple criteria:
        - Lower alpha is better
        - Lower entropy is better
        - Higher burke is better
        
        Parameters:
        -----------
        include_candidates: bool
            Whether to include candidate models in the search
        print_progress: bool
            Whether to print progress during the search
        alpha_weight: float
            Weight for alpha in the composite score (0-1)
        entropy_weight: float
            Weight for entropy in the composite score (0-1)
        burke_weight: float
            Weight for burke in the composite score (0-1)
            
        Returns:
        --------
        Optional[Path]: Path to the best model directory, or None if no models found
        """
        try:
            # Search in both optimized_results and optimization_results directories
            metrics_dirs = [
                Path('user_data/optimized_results'),
                Path('user_data/optimization_results')
            ]
            
            # Add candidate directories if requested
            if include_candidates:
                # Find all candidate directories
                candidate_pattern = Path('user_data/optimization_results').glob('candidate_*')
                intermediate_pattern = Path('user_data/optimization_results').glob('intermediate_*')
                metrics_dirs.extend(candidate_pattern)
                metrics_dirs.extend(intermediate_pattern)
            
            best_model = None
            best_score = float('inf')  # Lower score is better
            
            # Track all models for reporting
            all_models = []
            
            if print_progress:
                print(f"\n{BLUE}{'═' * 60}{END}")
                print(f"{BOLD}{CYAN}🔍 Searching for optimal model...{END}")
                print(f"{BLUE}{'═' * 60}{END}")
                print(f"{BOLD}Alpha weight:{END} {YELLOW}{alpha_weight:.2f}{END}")
                print(f"{BOLD}Entropy weight:{END} {YELLOW}{entropy_weight:.2f}{END}")
                print(f"{BOLD}Burke weight:{END} {YELLOW}{burke_weight:.2f}{END}")
                print(f"{BOLD}Including candidates:{END} {YELLOW}{include_candidates}{END}")
                print(f"{BLUE}{'─' * 60}{END}")
            
            # Process each metrics directory
            for metrics_dir in metrics_dirs:
                if not metrics_dir.exists():
                    if print_progress:
                        print(f"{YELLOW}Directory not found: {metrics_dir}{END}")
                    continue
                    
                # Process all subdirectories in this metrics directory
                for result_dir in metrics_dir.glob('*'):
                    metrics_file = result_dir / 'optimization_results.json'
                    
                    if not metrics_file.exists():
                        continue
                        
                    try:
                        with open(metrics_file) as f:
                            metrics_data = json.load(f)
                            
                            # Extract values with fallbacks
                            current_alpha = metrics_data.get('alpha', float('inf'))  # Lower is better
                            
                            # Handle different metrics structures
                            metrics = metrics_data.get('metrics', {})
                            if not metrics:
                                continue
                                
                            # Get burke value - could be a single value or a list
                            burke_value = metrics.get('burke', -float('inf'))
                            if isinstance(burke_value, list):
                                burke_max = max(burke_value) if burke_value else -float('inf')
                            else:
                                burke_max = burke_value
                                
                            current_entropy = metrics.get('entropy', float('inf'))  # Lower is better
                            
                            # Calculate composite score
                            current_score = (
                                (current_alpha * alpha_weight) +  # Lower alpha is better
                                (current_entropy * entropy_weight) -  # Lower entropy is better
                                (burke_max * burke_weight)  # Higher burke is better (subtract to minimize score)
                            )
                            
                            # Store model info for reporting
                            model_info = {
                                'dir': result_dir,
                                'alpha': current_alpha,
                                'burke': burke_max,
                                'entropy': current_entropy,
                                'score': current_score,
                                'timestamp': metrics_data.get('timestamp', 'unknown')
                            }
                            all_models.append(model_info)

                            # Update best model if this one is better
                            if current_score < best_score:
                                best_score = current_score
                                best_model = result_dir
                                
                                # Store values for logging
                                best_values = {
                                    'alpha': current_alpha,
                                    'burke_max': burke_max,
                                    'entropy': current_entropy,
                                    'score': current_score
                                }
                                
                                if print_progress:
                                    alpha_color = GREEN if current_alpha < 0 else RED
                                    burke_color = GREEN if burke_max > 0.5 else (YELLOW if burke_max > 0 else RED)
                                    entropy_color = GREEN if current_entropy < 0.3 else (YELLOW if current_entropy < 0.7 else RED)
                                    
                                    print(f"\n{GREEN}✨ New best model:{END} {CYAN}{result_dir.name}{END}")
                                    print(f"  {BOLD}Score:{END} {GREEN}{current_score:.6f}{END}")
                                    print(f"  {BOLD}Alpha:{END} {alpha_color}{current_alpha:.6f}{END}")
                                    print(f"  {BOLD}Burke:{END} {burke_color}{burke_max:.6f}{END}")
                                    print(f"  {BOLD}Entropy:{END} {entropy_color}{current_entropy:.6f}{END}")
                    except Exception as e:
                        logging.warning(f"{YELLOW}Error processing {metrics_file}: {e}{END}")
                        continue
            
            # Sort and print top models with colors
            if print_progress and all_models:
                # Sort by score (lower is better)
                sorted_models = sorted(all_models, key=lambda x: x['score'])
                
                print(f"\n{BLUE}{'═' * 60}{END}")
                print(f"{BOLD}{CYAN}🏆 Top 5 Models (out of {len(all_models)} total):{END}")
                print(f"{BLUE}{'═' * 60}{END}")
                
                for i, model in enumerate(sorted_models[:5]):
                    # Determine colors based on values
                    alpha_color = GREEN if model['alpha'] < 0 else RED
                    burke_color = GREEN if model['burke'] > 0.5 else (YELLOW if model['burke'] > 0 else RED)
                    entropy_color = GREEN if model['entropy'] < 0.3 else (YELLOW if model['entropy'] < 0.7 else RED)
                    score_color = GREEN if model['score'] < 0 else (YELLOW if model['score'] < 0.5 else RED)
                    
                    # Add medal emoji for top 3
                    medal = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else ""))
                    
                    print(f"{BOLD}{CYAN}{i+1}. {medal} {model['dir'].name}{END}")
                    print(f"   {BOLD}Score:{END} {score_color}{model['score']:.6f}{END}")
                    print(f"   {BOLD}Alpha:{END} {alpha_color}{model['alpha']:.6f}{END}")
                    print(f"   {BOLD}Burke:{END} {burke_color}{model['burke']:.6f}{END}")
                    print(f"   {BOLD}Entropy:{END} {entropy_color}{model['entropy']:.6f}{END}")
                    print(f"   {BOLD}Timestamp:{END} {MAGENTA}{model['timestamp']}{END}")
                    print(f"{BLUE}{'─' * 60}{END}")

            if best_model:
                logging.info(
                    f"{GREEN}Optimal model: {best_model}\n"
                    f"Alpha: {best_values['alpha']:.6f} | "
                    f"Burke Max: {best_values['burke_max']:.6f} | "
                    f"Entropy: {best_values['entropy']:.6f} | "
                    f"Score: {best_values['score']:.6f}{END}"
                )
                return best_model
                
            if print_progress:
                print(f"{RED}No suitable models found.{END}")
            return None
            
        except Exception as e:
            logging.error(f"{RED}Model search failed: {e}{END}")
            return None
