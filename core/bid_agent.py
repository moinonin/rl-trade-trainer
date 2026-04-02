import os
import sys
import logging
import time
import numpy as np
import pandas as pd
import warnings
import json
from typing import List, Tuple, Optional
from pathlib import Path

# Configure logging to completely ignore WARNING level for specific loggers
logging.getLogger('core.rl.agent').setLevel(logging.ERROR)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Only silence external library warnings
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)

# Silence all warnings from numpy
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
#from nlp.imit_main import imit_signal as imit
#from nlp.nlp_main import nlp_signal as nlp # nlp-project==0.0.9
if __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.metrics.nmatrix_hyperopt import calculate_nmatrix, save_optimization_result
from core.pretrained.rlbase import getBidsig, getNlpsig

#from agents.exagent.train_param_opt import exec_optimization

# Rest of imports
#from defirl.rl import RLmodel_bids as bid
from core.rl.agent import BidsAgent
from core.rl.trainer import BidsTrainer
from core.data.fetcher import Fetcher

class BidAgentTrainer:
    def __init__(self):
        # ANSI color codes
        self.BLUE = '\033[94m'
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.BOLD = '\033[1m'
        self.END = '\033[0m'
        
        print(f"\n{self.BOLD}{self.BLUE}=== Initializing Bid Agent Trainer ==={self.END}")
        
        # Create the agent with default model directory
        self.agent = BidsAgent(model_dir=str(Path(__file__).resolve().parent / "rl"))
        self.config_path = Path(__file__).resolve().parent / "config" / "agent_config.json"
        self.pretrained_signal_config = self._load_pretrained_signal_config()
        print(f"{self.YELLOW}Using default model with {len(self.agent.state_to_index)} states{self.END}")
        self.trainer = BidsTrainer(self.agent)
        self.position = None  # None, 'long', or 'short'
        self.entry_price = 0
        self.long_ml_candle = 17
        self.short_ml_candle = 10 #10

        # Best model tracking
        self.best_metrics = {"entropy": 0.9, "burke": 0.00, "alpha": 1.0}
        self.best_model = None

    def _state_rebalance_config(self) -> dict:
        config = self.agent.state_rebalance if isinstance(self.agent.state_rebalance, dict) else {}
        return {
            "enabled": bool(config.get("enabled", False)),
            "mode": config.get("mode", "reward_weighting"),
            "min_weight": float(config.get("min_weight", 1.0)),
            "max_weight": float(config.get("max_weight", 3.0)),
            "warn_ratio_gap": float(config.get("warn_ratio_gap", 0.2)),
            "log_path": config.get("log_path", "user_data/reports/state_distribution.csv"),
        }

    def _convergence_monitor_config(self) -> dict:
        config = self.agent.config.get("convergence_monitor", {}) if isinstance(self.agent.config, dict) else {}
        return {
            "enabled": bool(config.get("enabled", False)),
            "window": int(config.get("window", 100)),
            "dominance_threshold": float(config.get("dominance_threshold", 0.85)),
            "log_path": config.get("log_path", "user_data/reports/convergence_monitor.csv"),
        }

    def _build_reward_weights(self, positions: List[int]) -> Tuple[List[float], dict]:
        rebalance = self._state_rebalance_config()
        long_count = sum(1 for pos in positions if int(pos) == 0)
        short_count = sum(1 for pos in positions if int(pos) == 1)
        total_count = len(positions)

        stats = {
            "long_count": int(long_count),
            "short_count": int(short_count),
            "total_count": int(total_count),
            "long_ratio": float(long_count / total_count) if total_count else 0.0,
            "short_ratio": float(short_count / total_count) if total_count else 0.0,
            "rebalance_enabled": rebalance["enabled"],
            "rebalance_mode": rebalance["mode"],
        }

        if not rebalance["enabled"] or rebalance["mode"] != "reward_weighting" or long_count == 0 or short_count == 0:
            return [1.0] * total_count, stats

        minority_count = min(long_count, short_count)
        majority_count = max(long_count, short_count)
        raw_minority_weight = majority_count / minority_count if minority_count > 0 else rebalance["max_weight"]
        minority_weight = min(rebalance["max_weight"], max(rebalance["min_weight"], raw_minority_weight))

        long_weight = minority_weight if long_count < short_count else rebalance["min_weight"]
        short_weight = minority_weight if short_count < long_count else rebalance["min_weight"]

        stats.update({
            "long_weight": float(long_weight),
            "short_weight": float(short_weight),
        })
        weights = [float(short_weight if int(pos) == 1 else long_weight) for pos in positions]
        return weights, stats

    def _build_reward_scale_stats(self, prices: List[float]) -> dict:
        if len(prices) < 2:
            return {
                "avg_abs_price_change": 0.0,
                "median_abs_price_change": 0.0,
                "max_abs_price_change": 0.0,
                "switching_cost_ratio": 0.0,
                "hold_reward_ratio": 0.0,
            }

        price_changes = np.diff(np.asarray(prices, dtype=float))
        abs_changes = np.abs(price_changes)
        avg_abs_price_change = float(np.mean(abs_changes)) if len(abs_changes) else 0.0
        median_abs_price_change = float(np.median(abs_changes)) if len(abs_changes) else 0.0
        max_abs_price_change = float(np.max(abs_changes)) if len(abs_changes) else 0.0
        baseline = avg_abs_price_change if avg_abs_price_change > 1e-12 else 0.0

        return {
            "avg_abs_price_change": avg_abs_price_change,
            "median_abs_price_change": median_abs_price_change,
            "max_abs_price_change": max_abs_price_change,
            "switching_cost_ratio": float(self.agent.switching_cost / baseline) if baseline else 0.0,
            "hold_reward_ratio": float(abs(self.agent.hold_reward) / baseline) if baseline else 0.0,
        }

    def save_state_distribution_stats(self, stats: dict, run_label: str):
        """Append per-run state distribution stats for debugging rebalancing."""
        rebalance = self._state_rebalance_config()
        output_path = rebalance["log_path"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        row = dict(stats)
        row["run_label"] = run_label
        row["logged_at"] = pd.Timestamp.now().isoformat()
        row.setdefault("long_weight", 1.0)
        row.setdefault("short_weight", 1.0)

        write_header = not os.path.exists(output_path)
        pd.DataFrame([row]).to_csv(output_path, mode='a', header=write_header, index=False)
        logging.info(
            "State distribution %s | long=%s short=%s | weights long=%.3f short=%.3f | avg_abs_change=%.6f",
            run_label,
            row["long_count"],
            row["short_count"],
            row["long_weight"],
            row["short_weight"],
            row.get("avg_abs_price_change", 0.0),
        )

    def report_state_distribution(self, stats: dict, run_label: str):
        """Emit a concise runtime summary for long/short state balance."""
        rebalance = self._state_rebalance_config()
        long_ratio = float(stats.get("long_ratio", 0.0))
        short_ratio = float(stats.get("short_ratio", 0.0))
        ratio_gap = abs(long_ratio - short_ratio)
        imbalance_status = "balanced" if ratio_gap <= rebalance["warn_ratio_gap"] else "imbalanced"
        message = (
            f"State balance {run_label}: "
            f"long={stats.get('long_count', 0)} ({long_ratio:.1%}), "
            f"short={stats.get('short_count', 0)} ({short_ratio:.1%}), "
            f"weights L/S={stats.get('long_weight', 1.0):.2f}/{stats.get('short_weight', 1.0):.2f}, "
            f"avg|dP|={stats.get('avg_abs_price_change', 0.0):.6f}, "
            f"switch_ratio={stats.get('switching_cost_ratio', 0.0):.2f}, "
            f"hold_ratio={stats.get('hold_reward_ratio', 0.0):.2f}, "
            f"status={imbalance_status}"
        )

        if imbalance_status == "imbalanced":
            logging.warning(message)
            print(f"{self.YELLOW}{message}{self.END}")
        else:
            logging.info(message)
            print(f"{self.GREEN}{message}{self.END}")

    def summarize_action_convergence(self, history: List[dict], run_label: str) -> Optional[dict]:
        """Measure recent action dominance without affecting training behavior."""
        monitor = self._convergence_monitor_config()
        if not monitor["enabled"] or not history:
            return None

        window = max(1, monitor["window"])
        recent_history = history[-window:]
        recent_actions = [step.get("action", "unknown") for step in recent_history]
        counts = {action: recent_actions.count(action) for action in self.agent.ACTIONS}
        total = len(recent_actions)
        dominant_action = max(counts, key=counts.get) if counts else "unknown"
        dominant_share = float(counts[dominant_action] / total) if total else 0.0
        avg_rewards = {}
        for action in self.agent.ACTIONS:
            rewards = [float(step.get("reward", 0.0)) for step in recent_history if step.get("action") == action]
            avg_rewards[f"{action}_avg_reward"] = float(np.mean(rewards)) if rewards else 0.0

        summary = {
            "run_label": run_label,
            "window_size": total,
            "dominant_action": dominant_action,
            "dominant_share": dominant_share,
            "dominance_threshold": monitor["dominance_threshold"],
            "status": "warning" if dominant_share >= monitor["dominance_threshold"] else "ok",
            "go_long_share": float(counts.get("go_long", 0) / total) if total else 0.0,
            "go_short_share": float(counts.get("go_short", 0) / total) if total else 0.0,
            "do_nothing_share": float(counts.get("do_nothing", 0) / total) if total else 0.0,
        }
        summary.update(avg_rewards)
        return summary

    def save_convergence_summary(self, summary: Optional[dict]):
        if not summary:
            return

        monitor = self._convergence_monitor_config()
        output_path = monitor["log_path"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        row = dict(summary)
        row["logged_at"] = pd.Timestamp.now().isoformat()
        write_header = not os.path.exists(output_path)
        pd.DataFrame([row]).to_csv(output_path, mode='a', header=write_header, index=False)

        message = (
            f"Convergence {row['run_label']}: dominant={row['dominant_action']} "
            f"({row['dominant_share']:.1%}), shares L/S/N="
            f"{row['go_long_share']:.1%}/{row['go_short_share']:.1%}/{row['do_nothing_share']:.1%}, "
            f"status={row['status']}"
        )
        if row["status"] == "warning":
            logging.warning(message)
            print(f"{self.YELLOW}{message}{self.END}")
        else:
            logging.info(message)
            print(f"{self.GREEN}{message}{self.END}")

    def _load_pretrained_signal_config(self) -> dict:
        with open(self.config_path) as f:
            config = json.load(f)

        mapping = config.get("pretrained_signal_map", {})
        return {
            "long_bias_signal": mapping.get("long_bias_signal", "do_nothing"),
            "short_bias_signal": mapping.get("short_bias_signal", "go_long"),
            "ambiguous_trade_position": mapping.get("ambiguous_trade_position", "do_nothing"),
            "default_is_short": int(mapping.get("default_is_short", 1)),
        }

    def resolve_pretrained_direction(self,
                                     long_signal: Optional[str],
                                     short_signal: Optional[str],
                                     previous_is_short: Optional[int] = None) -> Tuple[int, Optional[str]]:
        """
        Translate raw pretrained outputs into one canonical direction.
        """
        mapping = self.pretrained_signal_config
        fallback_is_short = (
            mapping["default_is_short"] if previous_is_short is None else previous_is_short
        )
        is_long_signal = long_signal == mapping["long_bias_signal"]
        is_short_signal = short_signal == mapping["short_bias_signal"]

        if is_long_signal and not is_short_signal:
            return 0, 'long'
        if is_short_signal and not is_long_signal:
            return 1, 'short'
        return fallback_is_short, mapping["ambiguous_trade_position"]

    def resolve_position_signal(self, prediction: Optional[str], dir: Optional[str]) -> Optional[str]:
        """
        Translate a raw model signal into a canonical position action.
        """
        if prediction == 'go_long':
            return 'long'
        if prediction == 'go_short':
            return 'short'
        if prediction == 'exit':
            return 'exit'
        return None

    def get_pretrained_action(self,
                              is_short: int,
                              dataframe: pd.DataFrame) -> Optional[str]:
        """
        Fetch the raw action from the side-specific pretrained model.

        Keep this as the single call site for pretrained bids inference so
        model-specific candle windows and output shape changes stay isolated.
        """
        ml_candle = self.short_ml_candle if is_short else self.long_ml_candle
        result = getBidsig(
            is_short=is_short,
            ml_candle=ml_candle,
            dataframe=dataframe,
        ).predict_action()
        return result.get('action') if isinstance(result, dict) else None
        
    def run_hyperopt(self, 
                    episode_df: pd.DataFrame, 
                    print_frequency: int = 20,
                    save_intermediate: bool = True,
                    intermediate_save_frequency: int = 50) -> Optional[dict]:
        """
        Run hyperopt optimization and return results
        
        Parameters:
        -----------
        episode_df: pd.DataFrame
            DataFrame containing episode data
        print_frequency: int
            How often to print optimization progress (iterations)
        save_intermediate: bool
            Whether to save intermediate results
        intermediate_save_frequency: int
            How often to save intermediate results (iterations)
            
        Returns:
        --------
        Optional[dict]: Optimization results or None if failed
        """
        try:
            # Add debug logging
            print(f"\n{'='*50}")
            print(f"Starting hyperopt with DataFrame shape: {episode_df.shape}")
            print(f"{'='*50}")
            
            # Ensure we have the required columns
            required_columns = ['state', 'action', 'reward', 'trade_position']
            missing_columns = [col for col in required_columns if col not in episode_df.columns]
            if missing_columns:
                logging.error(f"Missing required columns: {missing_columns}")
                print(f"Error: Missing required columns: {missing_columns}")
                return None

            # Convert episode DataFrame to format needed by nmatrix
            trades_df = pd.DataFrame({
                'profit_abs': episode_df['reward'],
                'is_short': [1 if pos == 'short' else 0 for pos in episode_df['trade_position']]
            })

            # Print summary of trades
            positive_trades = (trades_df['profit_abs'] > 0).sum()
            negative_trades = (trades_df['profit_abs'] <= 0).sum()
            total_profit = trades_df['profit_abs'].sum()
            
            print(f"Trades Summary:")
            print(f"Total Trades: {len(trades_df)}")
            print(f"Positive Trades: {positive_trades} ({positive_trades/len(trades_df)*100:.1f}%)")
            print(f"Negative Trades: {negative_trades} ({negative_trades/len(trades_df)*100:.1f}%)")
            print(f"Total Profit: {total_profit:.2f}")
            print(f"Average Profit per Trade: {total_profit/len(trades_df):.4f}")
            
            # Calculate min and max dates
            min_date = pd.Timestamp.now() - pd.Timedelta(days=1)  # Default to 1 day if no dates
            max_date = pd.Timestamp.now()
            starting_balance = 10000.0  # Default starting balance

            # Call calculate_nmatrix with error handling
            try:
                print(f"\nRunning optimization with:")
                print(f"Print Frequency: {print_frequency}")
                print(f"Save Intermediate: {save_intermediate}")
                print(f"Intermediate Save Frequency: {intermediate_save_frequency}")
                print(f"{'-'*50}")
                
                result = calculate_nmatrix(
                    trades=trades_df,
                    min_date=min_date,
                    max_date=max_date,
                    starting_balance=starting_balance,
                    agent=self.agent,  # Pass the agent to calculate_nmatrix
                    print_frequency=print_frequency,
                    save_intermediate=save_intermediate,
                    intermediate_save_frequency=intermediate_save_frequency
                )
                
                if result is None:
                    logging.error("calculate_nmatrix returned None")
                    print("Error: calculate_nmatrix returned None")
                    return None
                
                # Get results
                metrics = result.get('metrics', {})
                signed_alpha = result.get('signed_alpha', 0)
                save_dir = result.get('save_dir')
                intermediate_saves = result.get('intermediate_saves', [])
                special_saves = result.get('special_saves', [])
                
                print(f"\n{'='*50}")
                print(f"Hyperopt Results:")
                print(f"{'='*50}")
                print(f"Alpha: {signed_alpha:.6f}")
                print(f"Score: {result.get('nmatrix_score', 0):.6f}")
                print(f"Trades Analyzed: {len(trades_df)}")
                
                # Print key metrics
                if metrics:
                    print(f"\nKey Metrics:")
                    print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
                    print(f"Burke: {metrics.get('burke', 0):.6f}")
                    print(f"Entropy: {metrics.get('entropy', 0):.6f}")
                    print(f"Kelly Risk: {metrics.get('kelly_risk', 0):.6f}")
                
                # Persist config for a new best raw alpha regardless of whether the
                # run qualified as a best/candidate save.
                self.agent.persist_params_if_best_raw_alpha(signed_alpha)

                # If save_dir is returned, it means the metrics condition was met
                # Save the model to the same directory.
                if save_dir:
                    print(f"\n🏆 Best model! Saving to: {save_dir}")
                    self.agent.save_model_with_metrics(save_dir)
                
                # Also save to intermediate directories if any
                if intermediate_saves:
                    print(f"\nIntermediate saves: {len(intermediate_saves)}")
                    for i, save_path in enumerate(intermediate_saves):
                        print(f"{i+1}. {save_path}")
                        self.agent.save_model_with_metrics(save_path)
                        # Check if any intermediate results are the best raw alpha
                        # (intermediate_saves entries are (dir_path, json_path))
                        # We extract the alpha from the json later if needed, but for now 
                        # persist_params_if_best_raw_alpha handles its own best-check.
                        # Note: calculate_nmatrix might have returned better alphas in intermediate runs.
                        pass # The global best above should handle the best one from the current run.

                if special_saves:
                    print(f"\nSpecial low-alpha saves: {len(special_saves)}")
                    for i, save_path in enumerate(special_saves):
                        print(f"special {i+1}. {save_path}")
                        self.agent.save_model_with_metrics(save_path)
                
                return {
                    'nmatrix_score': result.get('nmatrix_score', 0),
                    'signed_alpha': signed_alpha,
                    'trades_analyzed': len(trades_df),
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'metrics': metrics,
                    'save_dir': save_dir,
                    'json_path': result.get('json_path'),
                    'intermediate_saves': intermediate_saves,
                    'special_saves': special_saves
                }
                
            except Exception as e:
                logging.error(f"Error in calculate_nmatrix: {str(e)}")
                print(f"Error in calculate_nmatrix: {str(e)}")
                return None
            
        except Exception as e:
            logging.error(f"Error in hyperopt: {str(e)}")
            print(f"Error in hyperopt: {str(e)}")
            if 'episode_df' in locals():
                logging.error(f"Episode DataFrame columns: {episode_df.columns}")
            if 'trades_df' in locals():
                logging.error(f"Trades DataFrame columns: {trades_df.columns}")
            return None

    def generate_episode_report(self, episode_df: pd.DataFrame, starting_balance: float, episode_num: int) -> dict:
        """
        Generate a simplified performance report for a single training episode
        """
        total_trades = len(episode_df)
        winning_trades = len(episode_df[episode_df['reward'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_reward = episode_df['reward'].mean()
        max_reward = episode_df['reward'].max()
        min_reward = episode_df['reward'].min()
        
        action_distribution = episode_df['action'].value_counts(normalize=True).to_dict()
        
        return {
            'episode_number': episode_num,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'action_distribution': action_distribution,
            'final_balance': starting_balance * (1 + episode_df['reward'].sum()),
            'timestamp': pd.Timestamp.now().isoformat(),
            'state': ','.join([str(x) for x in episode_df['state'].tolist()])
            #'ask': [state[0] for state in episode_df['state']],
            #'bid': [state[1] for state in episode_df['state']],
            #'sma_compare': [state[2] for state in episode_df['state']],
            #'is_short': [state[3] for state in episode_df['state']]
        }

    def save_episode_reports(self, reports: List[dict], output_path: str):
        """
        Save episode reports to a CSV file
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert reports to DataFrame
        df = pd.DataFrame(reports)
        
        # Convert action distribution to separate columns
        if 'action_distribution' in df.columns:
            action_dist = pd.json_normalize(df['action_distribution'])
            df = df.drop('action_distribution', axis=1)
            df = pd.concat([df, action_dist], axis=1)
        
        df.to_csv(output_path, index=False)
        logging.info(f"Episode reports saved to {output_path}")

    def save_trade_history(self, episode_df: pd.DataFrame, output_path: str, episode_num: int):
        """Append per-trade model outputs for downstream analysis."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        trade_history_df = episode_df.copy()
        trade_history_df['episode_number'] = episode_num
        trade_history_df['logged_at'] = pd.Timestamp.now().isoformat()

        write_header = not os.path.exists(output_path)
        trade_history_df.to_csv(output_path, mode='a', header=write_header, index=False)
        logging.info(f"Trade history appended to {output_path}")

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[List[Tuple], List[float], List[str], List[float], dict]:
        """
        Prepare training data from DataFrame using transition predictions from old model
        Returns states, prices, positions, reward weights, and state distribution stats
        """
        # Initial validation
        logging.info(f"Initial DataFrame shape: {df.shape}")

        if df.empty:
            raise ValueError("Empty DataFrame provided to prepare_training_data")

        # Verify required columns exist
        required_cols = ['ask', 'bid', 'sma-compare', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        states = []
        positions = []
        trade_positions = []
        # This section uses old models to get initial positions
        # ------------------------------------------------------------------------------------------------------------
        # Collect predictions from old models
        # ------------------------------------------------------------------------------------------------------------
        # Get predictions for each row
        min_required_rows = 27  # max(self.long_ml_candle, self.short_ml_candle)

        for i in range(len(df)):
            current_row = df.iloc[i]  # Get the current row

            if i < min_required_rows - 1:
                # Use default position (1 for short) for initial states
                initial_is_short = self.pretrained_signal_config["default_is_short"]
                states.append((current_row['ask'], current_row['bid'], current_row['sma-compare'], initial_is_short))
                positions.append(initial_is_short)
                trade_positions.append(None)
                continue

            current_df = df.iloc[:i + 1].copy()

            # Get predictions
            long_next_action = self.get_pretrained_action(
                is_short=0,
                dataframe=current_df,
            )
            short_next_action = self.get_pretrained_action(
                is_short=1,
                dataframe=current_df,
            )
            #short_next_action = getNlpsig(ml_candle=self.short_ml_candle, dataframe=current_df)

            is_short, trade_position = self.resolve_pretrained_direction(
                long_signal=long_next_action,
                short_signal=short_next_action,
                previous_is_short=positions[-1] if positions else self.pretrained_signal_config["default_is_short"],
            )

            # Update base_state with the determined is_short
            full_state = (current_row['ask'], current_row['bid'], current_row['sma-compare'], is_short)
            states.append(full_state)
            positions.append(is_short)

            trade_positions.append(trade_position)

        # Update position tracking
        self.position = 'short' if positions[-1] == 1 else 'long'

        # Normalize prices
        prices = df['close'].tolist()
        prices = np.array(prices) / np.mean(prices)
        reward_weights, state_stats = self._build_reward_weights(positions)
        state_stats.update(self._build_reward_scale_stats(prices.tolist()))

        logging.info(f"Prepared {len(states)} states, {len(prices)} prices, {len(trade_positions)} positions")
        return states, prices.tolist(), trade_positions, reward_weights, state_stats

    def update_position(self, prediction: str, current_price: float, dir: str):
        """Update position based on model prediction"""
        target_position = self.resolve_position_signal(prediction, dir=dir)

        if target_position == 'long' and self.position != 'long':
            self.position = 'long'
            self.entry_price = current_price
            logging.info(f"Position changed to LONG at price {current_price}")
        elif target_position == 'short' and self.position != 'short':
            self.position = 'short'
            self.entry_price = current_price
            logging.info(f"Position changed to SHORT at price {current_price}")
        elif target_position == 'exit' and self.position is not None:
            logging.info(f"Exiting {self.position} position from {self.entry_price}")
            self.position = None
            self.entry_price = 0

    def continuous_training(self,
                          pair: str,
                          exchange_name: str,
                          interval: str = '1m',
                          training_window: int = 100,
                          update_interval: int = 20,
                          max_iterations: Optional[int] = None):
        """Continuously train the agent on live market data"""
        fetcher = Fetcher(pair=pair, 
                         exchange_name=exchange_name, 
                         interval=interval,
                         limit=training_window)
        
        iteration = 0
        best_reward = float('-inf')
        episode_reports = []
        
        while True:
            if max_iterations and iteration >= max_iterations:
                break
                
            try:
                # Fetch and prepare latest data
                df = fetcher.computed_indicators_df()
                if df.empty:
                    logging.warning("Received empty DataFrame, skipping iteration")
                    time.sleep(update_interval)
                    continue
                
                # Prepare training data with current position state
                states, prices, trade_positions, reward_weights, state_stats = self.prepare_training_data(df)
                state_run_label = f"continuous_{iteration}"
                self.save_state_distribution_stats(state_stats, state_run_label)
                self.report_state_distribution(state_stats, state_run_label)
                
                # Train on the latest data with enhanced parameters
                history = self.trainer.train_episode(
                    states=states,
                    prices=prices,
                    position_status=1 if self.position == 'short' else 0,
                    reward_weights=reward_weights,
                    save_interval=20,  # Sample lightweight debug summaries every 20 steps
                    print_metrics_interval=50,  # Print metrics every 50 steps
                    save_model_interval=100,  # Save model every 100 steps if improved
                    verbose=True  # Print detailed progress
                )
                convergence_summary = self.summarize_action_convergence(history, state_run_label)
                self.save_convergence_summary(convergence_summary)
                
                # Debug logging
                logging.info(f"Length of states: {len(states)}")
                logging.info(f"Length of prices: {len(prices)}")
                logging.info(f"Length of trade_positions: {len(trade_positions)}")
                logging.info(f"Length of history: {len(history)}")
                logging.info(f"Sample of history: {history[0] if history else 'empty'}")
                
                # Create lists from history with length checks
                history_states = [h['state'] for h in history]
                history_actions = [h['action'] for h in history]
                history_rewards = [h['reward'] for h in history]
                
                logging.info(f"Length of history_states: {len(history_states)}")
                logging.info(f"Length of history_actions: {len(history_actions)}")
                logging.info(f"Length of history_rewards: {len(history_rewards)}")
                logging.info(f"Length of trade_positions: {len(trade_positions)}")
                
                # Ensure all arrays are the same length
                min_length = min(len(history_states), len(history_actions), 
                                len(history_rewards), len(trade_positions))
                
                # Create episode DataFrame with length-matched arrays
                episode_df = pd.DataFrame({
                    'state': history_states[:min_length],
                    'action': history_actions[:min_length],
                    'reward': history_rewards[:min_length],
                    'trade_position': trade_positions[:min_length]
                })
                self.save_trade_history(
                    episode_df,
                    'user_data/reports/trade_history.csv',
                    iteration
                )
                
                logging.info(f"Final DataFrame shape: {episode_df.shape}")
                
                # Run hyperopt with enhanced parameters
                hyperopt_results = self.run_hyperopt(
                    episode_df,
                    print_frequency=20,  # Print progress every 20 iterations
                    save_intermediate=True,  # Save intermediate results
                    intermediate_save_frequency=2  # Save every 2 runs (out of 5)
                )
                
                if hyperopt_results:
                    # No need to call print_fancy_results as run_hyperopt now prints detailed results
                    # Just print a summary of saved models
                    save_dir = hyperopt_results.get('save_dir')
                    json_path = hyperopt_results.get('json_path')
                    intermediate_saves = hyperopt_results.get('intermediate_saves', [])
                    
                    if save_dir or intermediate_saves:
                        print(f"\n{'='*50}")
                        print(f"Model Saving Summary:")
                        print(f"{'='*50}")
                        if save_dir:
                            print(f"Best model saved to: {save_dir}")
                            # Copy the JSON file to the model's pkls directory if it exists
                            if json_path and os.path.exists(json_path):
                                model_pkls_dir = self.agent.model_dir / 'pkls'
                                model_pkls_dir.mkdir(exist_ok=True)
                                target_json_path = model_pkls_dir / 'optimization_results.json'
                                try:
                                    import shutil
                                    shutil.copy2(json_path, target_json_path)
                                    print(f"Copied metrics JSON to model's pkls directory: {target_json_path}")
                                except Exception as json_copy_error:
                                    print(f"Failed to copy metrics JSON: {json_copy_error}")
                        print(f"Intermediate models saved: {len(intermediate_saves)}")
                
                # Generate and store episode report
                episode_report = self.generate_episode_report(episode_df, starting_balance=10000.0, episode_num=iteration)
                episode_reports.append(episode_report)
                
                # Save reports periodically
                if iteration % 10 == 0:
                    self.save_episode_reports(episode_reports, 'user_data/reports/episode_reports.csv')
                
                # Log training progress
                # Calculate total reward from history entries
                total_reward = sum([h['reward'] for h in history])
                avg_reward = total_reward / len(history) if history else 0
                
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    logging.info(f"New best reward achieved: {best_reward:.4f}")
                
                logging.info(f"Iteration {iteration}: Avg Reward = {avg_reward:.4f}")
                
                # Make predictions and update position
                latest_state = states[-1]
                base_state = latest_state[:3]
                current_price = df['close'].iloc[-1]
                prediction, is_short = self.agent.strategic_action_selection(base_state)
                dir = 'short' if is_short else 'long' 
                self.update_position(prediction, current_price, dir)
                
                logging.info(f"Latest prediction: {prediction}, Current position: {self.position}")
                if self.position:
                    pnl = (current_price - self.entry_price) * (1 if self.position == 'long' else -1)
                    logging.info(f"Current PnL: {pnl:.2f}")
                
                iteration += 1
                time.sleep(update_interval)
                
            except Exception as e:
                logging.error(f"Error during continuous training: {e}")
                time.sleep(update_interval)

    def should_save_model(self, hyperopt_results):
        """
        Determines whether the new model should be saved based on metrics
        """
        alpha = hyperopt_results.get('signed_alpha', 0)
        metrics = hyperopt_results.get('metrics', {})
        
        # Add safety checks
        if not metrics:
            logging.warning("No metrics available for model evaluation")
            return False
        
        burke = metrics.get('burke', 0)
        entropy = metrics.get('entropy', 1)
        
        return (
            alpha < self.best_metrics["alpha"] and
            burke > self.best_metrics["burke"] and
            entropy < self.best_metrics["entropy"]
        )

    def update_best_model(self, hyperopt_results):
        """
        Updates the best model if the current one outperforms the previous best.
        """
        alpha = hyperopt_results.get('signed_alpha', 0)
        metrics = hyperopt_results.get('metrics', {})

        if self.should_save_model(hyperopt_results):
            self.best_metrics = {"entropy": metrics.get('entropy', 1), "burke": metrics.get('burke', 0), "alpha": alpha}
            self.best_model = self.agent  # Save reference to the best model
            return True  # Indicates the model is updated
        return False  # No update needed

    def print_fancy_results(self, hyperopt_results):
        """Print fancy formatted hyperopt results"""
        # Early return if hyperopt_results is None
        if hyperopt_results is None:
            print("\nNo hyperopt results available to display")
            return
        
        # ANSI color codes
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        RED = '\033[91m'
        YELLOW = '\033[93m'
        CYAN = '\033[96m'
        MAGENTA = '\033[95m'
        BOLD = '\033[1m'
        BG_BLUE = '\033[44m'
        BG_GREEN = '\033[42m'
        BG_RED = '\033[41m'
        BG_CYAN = '\033[46m'
        END = '\033[0m'
        
        # ASCII art border
        border = f"{BLUE}╔{'═' * 60}╗{END}"
        separator = f"{BLUE}╟{'─' * 60}╢{END}"
        bottom = f"{BLUE}╚{'═' * 60}╝{END}"
        
        # Safely get metrics with default empty dict
        metrics = hyperopt_results.get('metrics', {})
        alpha = hyperopt_results.get('signed_alpha', 0)
        alpha_color = GREEN if alpha < 0 else RED
        
        try:
            print("\n" + border)
            print(f"{BLUE}║{END} {BG_CYAN}{BOLD}  🤖 HYPEROPT OPTIMIZATION RESULTS  {END}{' ' * 30}{BLUE}║{END}")
            print(separator)
            
            # Main metrics with sparklines
            alpha_sparkline = "▼" if alpha < 0 else "▲"
            score_sparkline = "▼" if hyperopt_results.get('nmatrix_score', 0) < 0 else "▲"
            
            print(f"{BLUE}║{END} {BOLD}Alpha:{END} {alpha_color}{alpha:.6f} {alpha_sparkline}{END}{' ' * (49-len(str(alpha)))}{BLUE}║{END}")
            print(f"{BLUE}║{END} {BOLD}NMatrix Score:{END} {YELLOW}{hyperopt_results.get('nmatrix_score', 0):.6f} {score_sparkline}{END}{' ' * (41-len(str(hyperopt_results.get('nmatrix_score', 0))))}{BLUE}║{END}")
            print(separator)
            
            # Performance metrics with colorful indicators
            burke = metrics.get('burke', 0)
            entropy = metrics.get('entropy', 0)
            burke_color = GREEN if burke > 0.5 else (YELLOW if burke > 0 else RED)
            entropy_color = GREEN if entropy < 0.3 else (YELLOW if entropy < 0.7 else RED)
            
            print(f"{BLUE}║{END} {BG_CYAN}{BOLD}  📊 PERFORMANCE METRICS  {END}{' ' * 38}{BLUE}║{END}")
            print(f"{BLUE}║{END} Win Rate: {GREEN}{metrics.get('win_rate', 0):.2%}{END}{' ' * (49-len(str(metrics.get('win_rate', 0))))}{BLUE}║{END}")
            print(f"{BLUE}║{END} Kelly Risk: {YELLOW}{metrics.get('kelly_risk', 0):.4f}{END}{' ' * (47-len(str(metrics.get('kelly_risk', 0))))}{BLUE}║{END}")
            print(f"{BLUE}║{END} Burke Ratio: {burke_color}{burke:.4f}{END}{' ' * (47-len(str(burke)))}{BLUE}║{END}")
            print(f"{BLUE}║{END} Entropy: {entropy_color}{entropy:.4f}{END}{' ' * (50-len(str(entropy)))}{BLUE}║{END}")
            print(f"{BLUE}║{END} Var Coefficient: {MAGENTA}{metrics.get('var_coeff', 0):.4f}{END}{' ' * (42-len(str(metrics.get('var_coeff', 0))))}{BLUE}║{END}")
            print(separator)
            
            # Trade statistics with improved formatting
            total_trades = metrics.get('total_trades', 0)
            profitable_trades = metrics.get('profitable_trades', 0)
            losing_trades = metrics.get('losing_trades', 0)
            
            # Calculate win percentage for visual bar
            win_pct = profitable_trades / total_trades if total_trades > 0 else 0
            bar_length = 20
            win_bar = int(win_pct * bar_length)
            win_bar_str = f"{GREEN}{'█' * win_bar}{RED}{'█' * (bar_length - win_bar)}{END}"
            
            print(f"{BLUE}║{END} {BG_CYAN}{BOLD}  📈 TRADE STATISTICS  {END}{' ' * 40}{BLUE}║{END}")
            print(f"{BLUE}║{END} Total Trades: {YELLOW}{total_trades}{END}{' ' * (46-len(str(total_trades)))}{BLUE}║{END}")
            print(f"{BLUE}║{END} Win/Loss: {win_bar_str} {GREEN}{win_pct:.1%}{END}{' ' * (29-len(f'{win_pct:.1%}'))}{BLUE}║{END}")
            print(f"{BLUE}║{END} Long Trades: {CYAN}{metrics.get('long_trades', 0)}{END}{' ' * (47-len(str(metrics.get('long_trades', 0))))}{BLUE}║{END}")
            print(f"{BLUE}║{END} Short Trades: {CYAN}{metrics.get('short_trades', 0)}{END}{' ' * (46-len(str(metrics.get('short_trades', 0))))}{BLUE}║{END}")
            
            # Format profitable trades with split
            profitable_split = f"{profitable_trades} ({metrics.get('profitable_long_trades', 0)}/{metrics.get('profitable_short_trades', 0)} L/S)"
            print(f"{BLUE}║{END} Profitable: {GREEN}{profitable_split}{END}{' ' * (48-len(profitable_split))}{BLUE}║{END}")
            print(f"{BLUE}║{END} Losing: {RED}{losing_trades}{END}{' ' * (51-len(str(losing_trades)))}{BLUE}║{END}")
            print(separator)
            
            # Timestamp with icon
            timestamp = hyperopt_results.get('timestamp', '')
            print(f"{BLUE}║{END} 🕒 {MAGENTA}{timestamp}{END}{' ' * (54-len(timestamp))}{BLUE}║{END}")
            print(bottom)
            
            # Print metrics tracker
            self._print_metrics_tracker(alpha, burke, entropy, hyperopt_results.get('nmatrix_score', 0))
        
        except Exception as e:
            print(f"\n{RED}Error printing hyperopt results: {str(e)}{END}")
            print(f"{YELLOW}Raw hyperopt_results:{END}", hyperopt_results)
            
    def _print_metrics_tracker(self, alpha, burke, entropy, score, prev_metrics=None):
        """Print a colorful metrics tracker"""
        # ANSI color codes
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        RED = '\033[91m'
        YELLOW = '\033[93m'
        CYAN = '\033[96m'
        MAGENTA = '\033[95m'
        BOLD = '\033[1m'
        END = '\033[0m'
        
        # Get previous metrics from best_metrics if available
        prev_alpha = self.best_metrics.get("alpha", 1.0)
        prev_burke = self.best_metrics.get("burke", 0.0)
        prev_entropy = self.best_metrics.get("entropy", 0.9)
        prev_score = 0  # We don't track previous score
        
        # Calculate improvements
        alpha_improved = prev_alpha > alpha
        burke_improved = prev_burke < burke
        entropy_improved = prev_entropy > entropy
        
        # Calculate improvement percentages
        alpha_pct = abs((alpha - prev_alpha) / prev_alpha * 100) if prev_alpha != 0 else 0
        burke_pct = abs((burke - prev_burke) / max(0.001, prev_burke) * 100)
        entropy_pct = abs((entropy - prev_entropy) / max(0.001, prev_entropy) * 100)
        
        # Determine colors and indicators
        alpha_color = GREEN if alpha_improved else RED
        burke_color = GREEN if burke_improved else RED
        entropy_color = GREEN if entropy_improved else RED
        
        alpha_indicator = "▼" if alpha_improved else "▲"
        burke_indicator = "▲" if burke_improved else "▼"
        entropy_indicator = "▼" if entropy_improved else "▲"
        
        # Print metrics tracker
        print("\n" + f"{BLUE}╔{'═' * 60}╗{END}")
        print(f"{BLUE}║{END} {BOLD}{MAGENTA}🔍 METRICS TRACKER - MODEL SAVED{END}{' ' * 30}{BLUE}║{END}")
        print(f"{BLUE}╠{'═' * 60}╣{END}")
        
        # Print metrics with improvements
        print(f"{BLUE}║{END} {BOLD}ALPHA:{END}    {alpha_color}{alpha:.6f} {alpha_indicator}{END} " + 
              (f"{GREEN}(Improved by {alpha_pct:.1f}%){END}" if alpha_improved else f"{RED}(Worsened by {alpha_pct:.1f}%){END}") + 
              f"{' ' * (20-len(f'{alpha_pct:.1f}'))}{BLUE}║{END}")
              
        print(f"{BLUE}║{END} {BOLD}BURKE:{END}    {burke_color}{burke:.6f} {burke_indicator}{END} " + 
              (f"{GREEN}(Improved by {burke_pct:.1f}%){END}" if burke_improved else f"{RED}(Worsened by {burke_pct:.1f}%){END}") + 
              f"{' ' * (20-len(f'{burke_pct:.1f}'))}{BLUE}║{END}")
              
        print(f"{BLUE}║{END} {BOLD}ENTROPY:{END}  {entropy_color}{entropy:.6f} {entropy_indicator}{END} " + 
              (f"{GREEN}(Improved by {entropy_pct:.1f}%){END}" if entropy_improved else f"{RED}(Worsened by {entropy_pct:.1f}%){END}") + 
              f"{' ' * (20-len(f'{entropy_pct:.1f}'))}{BLUE}║{END}")
              
        print(f"{BLUE}║{END} {BOLD}SCORE:{END}    {YELLOW}{score:.6f}{END}{' ' * 45}{BLUE}║{END}")
        
        # Overall assessment
        improvements = sum([alpha_improved, burke_improved, entropy_improved])
        if improvements >= 2:
            print(f"{BLUE}║{END} {GREEN}{BOLD} ✅ OVERALL: SIGNIFICANT IMPROVEMENT {END}{' ' * 29}{BLUE}║{END}")
        elif improvements == 1:
            print(f"{BLUE}║{END} {YELLOW}{BOLD} ⚠️ OVERALL: MIXED RESULTS {END}{' ' * 37}{BLUE}║{END}")
        else:
            print(f"{BLUE}║{END} {RED}{BOLD} ❌ OVERALL: NO IMPROVEMENT {END}{' ' * 37}{BLUE}║{END}")
            
        print(f"{BLUE}╚{'═' * 60}╝{END}")

    def evaluate_model(self, df: pd.DataFrame) -> dict:
        """Evaluate model on dataset"""
        predictions = []
        actual = []
        
        for _, row in df.iterrows():
            pred = self.predict(
                ask=row.ask,
                bid=row.bid,
                sma_compare=row['sma-compare'],
                is_short=row.is_short if 'is_short' in df.columns else 0
            )
            predictions.append(pred)
            if 'refined-action' in df.columns:
                actual.append(row['refined-action'])
        
        # Calculate metrics
        accuracy = None
        if actual:
            accuracy = sum(np.array(predictions) == np.array(actual)) / len(predictions)
            
        # Calculate action distribution
        action_dist = pd.Series(predictions).value_counts(normalize=True)
        
        return {
            'predictions': predictions,
            'accuracy': accuracy,
            'action_distribution': action_dist.to_dict()
        }
    
    def predict(self, ask: float, bid: float, sma_compare: int) -> str:
        """Make a prediction for a single state"""
        is_short = 1 if self.position == 'short' else 0
        state = (ask, bid, sma_compare, is_short)
        return self.agent.select_action(state, epsilon=0)  # epsilon=0 for pure exploitation

    def train_on_historical_data(self,
                               historical_df: pd.DataFrame,
                               batch_size: int = 100,
                               historical_interval: int = 0,
                               save_reports: bool = True):
        """Train on historical data in batches with optional time interval"""
        print(f"{self.BLUE}Training on historical data with {len(historical_df)} samples{self.END}")
        
        episode_reports = []
        best_reward = float('-inf')
        
        # Process data in batches
        for i in range(0, len(historical_df), batch_size):
            batch_df = historical_df.iloc[i:i + batch_size].copy()
            
            # Prepare training data with positions
            states, prices, trade_positions, reward_weights, state_stats = self.prepare_training_data(batch_df)
            state_run_label = f"historical_{i // batch_size}"
            self.save_state_distribution_stats(state_stats, state_run_label)
            self.report_state_distribution(state_stats, state_run_label)
            
            # Train on the batch with enhanced parameters
            history = self.trainer.train_episode(
                states=states,
                prices=prices,
                position_status=1 if self.position == 'short' else 0,
                reward_weights=reward_weights,
                save_interval=20,  # Sample lightweight debug summaries every 20 steps
                print_metrics_interval=50,  # Print metrics every 50 steps
                save_model_interval=100,  # Save model every 100 steps if improved
                verbose=True  # Print detailed progress
            )
            convergence_summary = self.summarize_action_convergence(history, state_run_label)
            self.save_convergence_summary(convergence_summary)
            
            # Debug logging
            print(f"Length of states: {len(states)}")
            print(f"Length of prices: {len(prices)}")
            print(f"Length of trade_positions: {len(trade_positions)}")
            print(f"Length of history: {len(history)}")
            print(f"Sample of history: {history[0] if history else 'empty'}")
            
            # Create lists from history with length checks
            history_states = [h['state'] for h in history]
            history_actions = [h['action'] for h in history]
            history_rewards = [h['reward'] for h in history]
            
            print(f"Length of history_states: {len(history_states)}")
            print(f"Length of history_actions: {len(history_actions)}")
            print(f"Length of history_rewards: {len(history_rewards)}")
            print(f"Length of trade_positions: {len(trade_positions)}")
            
            # Ensure all arrays are the same length
            min_length = min(len(history_states), len(history_actions), 
                            len(history_rewards), len(trade_positions))
            
            # Create episode DataFrame with length-matched arrays
            episode_df = pd.DataFrame({
                'state': history_states[:min_length],
                'action': history_actions[:min_length],
                'reward': history_rewards[:min_length],
                'trade_position': trade_positions[:min_length]
            })
            self.save_trade_history(
                episode_df,
                'user_data/reports/trade_history.csv',
                i // batch_size
            )
            
            print(f"Final DataFrame shape: {episode_df.shape}")
            
            # Run hyperopt with enhanced parameters
            hyperopt_results = self.run_hyperopt(
                episode_df,
                print_frequency=20,  # Print progress every 20 iterations
                save_intermediate=True,  # Save intermediate results
                intermediate_save_frequency=2  # Save every 2 runs (out of 5)
            )
            
            if hyperopt_results:
                # No need to call print_fancy_results as run_hyperopt now prints detailed results
                # Just print a summary of saved models
                save_dir = hyperopt_results.get('save_dir')
                json_path = hyperopt_results.get('json_path')
                intermediate_saves = hyperopt_results.get('intermediate_saves', [])
                
                if save_dir or intermediate_saves:
                    print(f"\n{'='*50}")
                    print(f"Model Saving Summary:")
                    print(f"{'='*50}")
                    if save_dir:
                        print(f"Best model saved to: {save_dir}")
                        # Copy the JSON file to the model's pkls directory if it exists
                        if json_path and os.path.exists(json_path):
                            model_pkls_dir = self.agent.model_dir / 'pkls'
                            model_pkls_dir.mkdir(exist_ok=True)
                            target_json_path = model_pkls_dir / 'optimization_results.json'
                            try:
                                import shutil
                                shutil.copy2(json_path, target_json_path)
                                print(f"Copied metrics JSON to model's pkls directory: {target_json_path}")
                            except Exception as json_copy_error:
                                print(f"Failed to copy metrics JSON: {json_copy_error}")
                    print(f"Intermediate models saved: {len(intermediate_saves)}")
            
            # Update position and continue processing...
            current_price = batch_df['close'].iloc[-1]
            latest_state = states[-1]
            base_state = latest_state[:3]
            prediction, is_short = self.agent.strategic_action_selection(base_state)
            dir = 'short' if is_short else 'long' 
            self.update_position(prediction, current_price, dir)
            
            # Optional delay for historical processing
            if historical_interval > 0:
                time.sleep(historical_interval)
            
            # Generate and store episode report
            episode_report = self.generate_episode_report(
                episode_df, 
                starting_balance=10000.0, 
                episode_num=i//batch_size
            )
            episode_reports.append(episode_report)
            
            # Save reports periodically
            if save_reports and i % (batch_size * 10) == 0:
                self.save_episode_reports(
                    episode_reports, 
                    'user_data/reports/historical_episode_reports.csv'
                )
            
            # Log training progress
            # Calculate total reward from history entries
            total_reward = sum([h['reward'] for h in history])
            avg_reward = total_reward / len(history) if history else 0
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                print(f"{self.GREEN}New best reward achieved: {best_reward:.4f}{self.END}")
            
            print(f"Batch {i//batch_size}: Avg Reward = {avg_reward:.4f}")
            
            if self.position:
                pnl = (current_price - self.entry_price) * (1 if self.position == 'long' else -1)
                print(f"Current PnL: {pnl:.2f}")

    def train_mixed_mode(self,
                        historical_df: pd.DataFrame,
                        pair: str,
                        exchange_name: str,
                        interval: str = '1m',
                        training_window: int = 100,
                        historical_interval: int = 0,  # For historical data
                        live_interval: int = 20,      # For live data
                        max_iterations: Optional[int] = None):
        """Train first on historical data, then switch to live mode with different intervals"""
        
        # First train on historical data with minimal or no delay
        self.train_on_historical_data(
            historical_df,
            historical_interval=historical_interval
        )
        
        print(f"{self.YELLOW}Switching to live mode with {live_interval}s interval...{self.END}")
        
        # Then switch to live mode with exchange rate limiting
        self.continuous_training(
            pair=pair,
            exchange_name=exchange_name,
            interval=interval,
            training_window=training_window,
            update_interval=live_interval,
            max_iterations=max_iterations
        )

def main():
    import glob
    import os

    agent = BidsAgent()
    trainer = BidAgentTrainer()

    # Use project root as base path
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Find all files ending with '_refined.csv'
    search_phrase = '*_large_83rl_mod_balanced.csv'
    refined_files = glob.glob(os.path.join(base_path, f'**/{search_phrase}'), recursive=True)
    
    if not refined_files:
        print(f"\n⚠️ No {search_phrase} files found. Skipping directly to live mode...")
    else:
        print(f"\n{'-'*50}")
        print(f"Found {len(refined_files)} refined dataset(s):")
        for file in refined_files:
            print(f"📊 {os.path.basename(file)}")
        print(f"{'-'*50}\n")
    
    # Track successful and failed files
    successful_files = []
    failed_files = []
    
    # Train on each dataset sequentially if any found
    if refined_files:
        for file_path in refined_files:
            try:
                print(f"\n{'='*50}")
                print(f"🔄 Training on: {os.path.basename(file_path)}")
                print(f"{'='*50}\n")
                
                # Load historical data
                historical_df = pd.read_csv(file_path)
                #exec_optimization(file_path)
                
                if historical_df.empty:
                    print(f"⚠️ Skipping empty file: {os.path.basename(file_path)}")
                    failed_files.append((file_path, "Empty DataFrame"))
                    continue
                    
                # Check for required columns
                required_columns = ['open', 'high', 'low', 'close', 'volume']  # Add any other required columns
                missing_columns = [col for col in required_columns if col not in historical_df.columns]
                
                if missing_columns:
                    print(f"⚠️ Skipping file due to missing columns {missing_columns}: {os.path.basename(file_path)}")
                    failed_files.append((file_path, f"Missing columns: {missing_columns}"))
                    continue
                
                # Option 1: Fast historical training with no delay
                trainer.train_on_historical_data(
                    historical_df.iloc[-300:],
                    batch_size=100,
                    historical_interval=0  # No delay between batches
                )
                agent.decay_epsilon()
                successful_files.append(file_path)
                print(f"✅ Successfully processed: {os.path.basename(file_path)}")
                
            except Exception as e:
                print(f"❌ Error processing {os.path.basename(file_path)}: {str(e)}")
                failed_files.append((file_path, str(e)))
                continue
        
        # Print summary report
        print(f"\n{'='*50}")
        print("📊 Training Summary Report")
        print(f"{'='*50}")
        print(f"Total files found: {len(refined_files)}")
        print(f"Successfully processed: {len(successful_files)}")
        print(f"Failed to process: {len(failed_files)}")
        
        if failed_files:
            print("\n⚠️ Files that failed:")
            for file_path, error in failed_files:
                print(f"- {os.path.basename(file_path)}: {error}")

    # Always attempt live mode if no files were found OR if we've finished processing them
    print("\n🚀 Starting live mode...")
    print(f"{'='*50}\n")
    
    try:
        trainer.continuous_training(
            pair="BTC/USDT:USDT",
            exchange_name="bybit",
            interval="1m",
            training_window=100,
            update_interval=20,
            max_iterations=None
        )
    except Exception as e:
        print(f"❌ Critical error in live mode: {e}")

if __name__ == "__main__":
    main()
