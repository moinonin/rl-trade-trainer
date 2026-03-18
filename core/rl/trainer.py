from typing import List, Tuple, Dict, Optional
import numpy as np
import logging
import os
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from .agent import BidsAgent

# ANSI color codes for colorful terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
BLUE = '\033[94m'
BOLD = '\033[1m'
END = '\033[0m'

logger = logging.getLogger(__name__)

class BidsTrainer:
    def __init__(self, agent: BidsAgent):
        self.agent = agent
        self.episode_count = 0
        
    def calculate_reward(self, 
                        action: str,
                        next_price: float,
                        current_price: float,
                        position_status: int) -> float:
        """Calculate reward with explicit position handling"""
        try:
            price_change = next_price - current_price
            
            # Use agent's action definitions for safety
            if action == self.agent.ACTIONS[1]:  # go_long
                return price_change if position_status == 0 else -abs(price_change)
            elif action == self.agent.ACTIONS[2]:  # go_short
                return -price_change if position_status == 1 else -abs(price_change)
            else:  # do_nothing
                return 0.0
        except Exception as e:
            logger.error("Reward calculation error: %s", e, exc_info=True)
            return 0.0
            
    def train_step(self, 
                   current_state: Tuple,
                   next_state: Tuple,
                   next_price: float,
                   current_price: float,
                   position_status: int) -> Dict:
        """Enhanced training step with validation"""
        result = {
            "action": self.agent.ACTIONS[0],  # do_nothing
            "reward": 0.0,
            "state": current_state,
            "next_state": next_state
        }
        
        try:
            action = self.agent.select_action(current_state)
            reward = self.calculate_reward(action, next_price, current_price, position_status)

            # Validate state transition before updating
            if not self._valid_state_transition(current_state, next_state):
                logger.warning("Invalid state transition: %s -> %s", current_state, next_state)
                return result
                
            self.agent.update(current_state, action, reward, next_state)
            #np.array(current_state, action, reward, next_state, dtype=object) = episode_transitions
            
            # Update result with actual values
            result.update({
                "action": action,
                "reward": reward,
                "q_value": self.agent.q_table[self.agent.get_state_index(current_state)].max()
            })
            
        except Exception as e:
            logger.error("Training step failed: %s", e, exc_info=True)
            
        return result

    def _valid_state_transition(self, current: Tuple, next: Tuple) -> bool:
        """Validate state transition logic"""
        # Implement your domain-specific transition rules here
        return True  # Add actual validation logic

    def train_episode(self, 
                     states: List[Tuple],
                     prices: List[float],
                     position_status: int,
                     save_interval: int = 10,
                     print_metrics_interval: int = 50,
                     save_model_interval: int = 100,
                     verbose: bool = True) -> List[Dict]:
        """
        Enhanced episode training with epsilon decay and interactive feedback
        
        Parameters:
        -----------
        states: List[Tuple]
            List of state tuples
        prices: List[float]
            List of prices corresponding to states
        position_status: int
            Current position status (0 for long, 1 for short)
        save_interval: int
            How often to save progress to history (steps)
        print_metrics_interval: int
            How often to print metrics (steps)
        save_model_interval: int
            How often to save the model (steps)
        verbose: bool
            Whether to print detailed progress information
            
        Returns:
        --------
        List[Dict]: History of training steps
        """
        history = []
        total_reward = 0.0
        best_reward = float('-inf')
        last_save_step = 0
        metrics_history = []
        
        try:
            if len(states) != len(prices):
                raise ValueError("States and prices must have equal length")
                
            desc = f"Training Episode {self.episode_count + 1}"
            with tqdm(total=len(states)-1, desc=desc, unit="step") as pbar:
                for i in range(len(states) - 1):
                    step_result = self.train_step(
                        current_state=states[i],
                        next_state=states[i+1],
                        next_price=prices[i+1],
                        current_price=prices[i],
                        position_status=position_status
                    )
                    
                    total_reward += step_result["reward"]
                    step_result["total_reward"] = total_reward
                    step_result["epsilon"] = self.agent.epsilon
                    
                    # Only append to history at save_interval to reduce memory usage
                    if i % save_interval == 0:
                        history.append(step_result)
                    
                    # Update progress metrics
                    pbar.update(1)
                    pbar.set_postfix({
                        'total_reward': f"{total_reward:,.2f}",
                        'epsilon': f"{self.agent.epsilon:.3f}",
                        'action': step_result["action"]
                    })
                    
                    # Print metrics at specified intervals
                    if verbose and i > 0 and i % print_metrics_interval == 0:
                        # Calculate metrics for this interval
                        interval_metrics = {
                            'step': i,
                            'total_reward': total_reward,
                            'avg_reward': total_reward / i,
                            'epsilon': self.agent.epsilon,
                            'q_value': step_result.get("q_value", 0),
                            'action_counts': {}
                        }
                        
                        # Count actions in recent history
                        recent_actions = [h.get("action") for h in history[-min(len(history), 100):]]
                        for action in self.agent.ACTIONS:
                            interval_metrics['action_counts'][action] = recent_actions.count(action)
                        
                        metrics_history.append(interval_metrics)
                        
                        # Print metrics with colors
                        reward_color = GREEN if total_reward > 0 else RED
                        avg_reward_color = GREEN if total_reward/i > 0 else RED
                        
                        print(f"\n{BLUE}{'─' * 50}{END}")
                        print(f"{BOLD}{CYAN}📊 Step {i}/{len(states)-1} Metrics:{END}")
                        print(f"{BLUE}{'─' * 50}{END}")
                        print(f"{BOLD}Total Reward:{END} {reward_color}{total_reward:.2f}{END}")
                        print(f"{BOLD}Avg Reward:{END} {avg_reward_color}{total_reward/i:.4f}{END}")
                        print(f"{BOLD}Epsilon:{END} {YELLOW}{self.agent.epsilon:.4f}{END}")
                        
                        # Print action distribution with colors
                        print(f"{BOLD}Recent Actions:{END}")
                        for action, count in interval_metrics['action_counts'].items():
                            action_color = CYAN
                            if action == self.agent.ACTIONS[1]:  # go_long
                                action_color = GREEN
                            elif action == self.agent.ACTIONS[2]:  # go_short
                                action_color = RED
                            
                            percentage = count / sum(interval_metrics['action_counts'].values()) * 100 if sum(interval_metrics['action_counts'].values()) > 0 else 0
                            bar_length = int(percentage / 5)  # 20 chars = 100%
                            bar = f"{action_color}{'█' * bar_length}{END}"
                            
                            print(f"  {action_color}{action}{END}: {bar} {count} ({percentage:.1f}%)")
                            
                        print(f"{BOLD}Q-Value:{END} {MAGENTA}{step_result.get('q_value', 0):.4f}{END}")
                    
                    # Save model at specified intervals
                    if i > 0 and i % save_model_interval == 0:
                        # Check if this is the best reward so far
                        if total_reward > best_reward:
                            best_reward = total_reward
                            if verbose:
                                print(f"\n🏆 New best reward at step {i}: {total_reward:.2f}")
                                print(f"Saving model...")
                            
                            # Save model with timestamp
                            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S") if 'pd' in globals() else time.strftime("%Y%m%d_%H%M%S")
                            save_dir = f"user_data/optimization_results/intermediate_step_{i}_{timestamp}"
                            
                            # Create directory if it doesn't exist
                            os.makedirs(save_dir, exist_ok=True)
                            
                            # Save model
                            self.agent.save_model_with_metrics(save_dir)
                            
                            if verbose:
                                print(f"\n{GREEN}🏆 Model saved to:{END} {CYAN}{save_dir}{END}")
                                
                                # Print a visual progress bar
                                progress_pct = i / (len(states) - 1) * 100
                                bar_length = 40
                                filled_length = int(bar_length * i // (len(states) - 1))
                                bar = f"{GREEN}{'█' * filled_length}{YELLOW}{'░' * (bar_length - filled_length)}{END}"
                                print(f"{BLUE}[{END}{bar}{BLUE}]{END} {BOLD}{progress_pct:.1f}%{END}")
                        
                        last_save_step = i
            
            # Post-episode processing
            self.episode_count += 1
            self.agent.decay_epsilon()
            
            # Always save at the end of the episode if we haven't saved recently
            if len(states) - 1 - last_save_step > save_model_interval // 2:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S") if 'pd' in globals() else time.strftime("%Y%m%d_%H%M%S")
                save_dir = f"user_data/optimization_results/episode_{self.episode_count}_{timestamp}"
                os.makedirs(save_dir, exist_ok=True)
                self.agent.save_model_with_metrics(save_dir)
                if verbose:
                    print(f"\nEpisode completed - Saved final model to: {save_dir}")
            
            # Clear transitions to prevent memory issues
            self.agent.clear_transitions()
            
            # Print episode summary with colors
            if verbose:
                # Determine colors based on performance
                final_reward_color = GREEN if total_reward > 0 else RED
                best_reward_color = GREEN if best_reward > 0 else RED
                
                print(f"\n{BLUE}{'═' * 60}{END}")
                print(f"{BOLD}{CYAN}🎮 Episode {self.episode_count} Summary:{END}")
                print(f"{BLUE}{'═' * 60}{END}")
                print(f"{BOLD}Total Steps:{END} {YELLOW}{len(states)-1}{END}")
                print(f"{BOLD}Final Reward:{END} {final_reward_color}{total_reward:.2f}{END}")
                print(f"{BOLD}Best Reward:{END} {best_reward_color}{best_reward:.2f}{END}")
                print(f"{BOLD}Final Epsilon:{END} {YELLOW}{self.agent.epsilon:.4f}{END}")
                print(f"{BOLD}States Known:{END} {MAGENTA}{len(self.agent.state_to_index)}{END}")
                
                # Print action distribution from the last 100 steps with visual bars
                if len(history) > 0:
                    recent_actions = [h.get("action") for h in history[-min(len(history), 100):]]
                    action_counts = {}
                    for action in self.agent.ACTIONS:
                        action_counts[action] = recent_actions.count(action)
                    
                    print(f"\n{BOLD}{CYAN}📊 Action Distribution (last 100 steps):{END}")
                    
                    # Calculate max count for scaling
                    max_count = max(action_counts.values()) if action_counts else 1
                    
                    for action, count in action_counts.items():
                        # Choose color based on action
                        action_color = CYAN
                        if action == self.agent.ACTIONS[1]:  # go_long
                            action_color = GREEN
                        elif action == self.agent.ACTIONS[2]:  # go_short
                            action_color = RED
                        
                        # Create visual bar
                        percentage = count / len(recent_actions) * 100
                        bar_length = int(percentage / 2.5)  # 40 chars = 100%
                        bar = f"{action_color}{'█' * bar_length}{END}"
                        
                        # Print with padding for alignment
                        action_padded = f"{action:<10}"
                        print(f"  {action_color}{action_padded}{END} {bar} {count} ({percentage:.1f}%)")
                
                # Print a decorative footer
                print(f"{BLUE}{'═' * 60}{END}")
            
            logger.info("Episode %d completed - Total reward: %.2f", 
                        self.episode_count, total_reward)
                        
        except Exception as e:
            logger.error("Episode training failed: %s", e, exc_info=True)
            
        return history
