import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from metrics.nmatrix_hyperopt import calculate_nmatrix
import warnings
from tqdm import tqdm
import time

# Suppress specific numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy.core._methods')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy.core.fromnumeric')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in scalar divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in scalar power')


def calculate_max_drawdown(rewards):
    cumulative = np.cumsum(rewards)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    return np.max(drawdown) if len(drawdown) > 0 else 0

def process_trading_history(csv_path: str, starting_balance: float) -> dict:
    """
    Process trading history CSV with columns: action, reward, state, next_state
    """
    print("\n🤖 Initializing trading history analysis...\n")
    time.sleep(0.5)  # Add dramatic effect
    
    # Read trading history with progress bar
    print("📊 Loading trading data...")
    df = pd.read_csv(csv_path)
    print(f"📈 Loaded {len(df)} trading entries\n")
    print(df.columns)
    print(df.head())
    time.sleep(0.3)
    
    with tqdm(total=6, desc="🔄 Processing metrics", ncols=100) as pbar:
        # Get rewards directly from the 'reward' column
        rewards = df['avg_reward'].values
        pbar.update(1)
        time.sleep(0.2)
        
        # Create timestamps for nmatrix calculation
        print("\n⏰ Generating temporal analysis framework...")
        df['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(days=len(df)),
            periods=len(df),
            freq='h'
        )
        pbar.update(1)
        time.sleep(0.2)
        
        # Extract is_short from state tuples
        print("🎯 Extracting position states...")
        state_components = df['state'].str.extract(r'\(([^,)]+),\s*([^,)]+),\s*([^,)]+)(?:,\s*(\d+))?\)')
        
        # Create trades DataFrame with all state components
        trades_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'profit_abs': df['avg_reward'] * starting_balance,
            'ask': state_components[0].astype(float),
            'bid': state_components[1].astype(float),
            'sma_compare': state_components[2].astype(float),
            'is_short': pd.to_numeric(state_components[3], errors='coerce').fillna(0).astype(int)
        })
        trades_df['action'] = np.where(
            trades_df['is_short'] == 1,
            'go_short',
            np.where(
                trades_df['is_short'] == 0,
                'go_long',
                'do_nothing'
            )
        )

        pbar.update(1)
        time.sleep(0.2)
        
        print("\n🧮 Calculating nmatrix metrics...")
        min_date = trades_df['timestamp'].min()
        max_date = trades_df['timestamp'].max()
        # Skip optimization for reporting
        nmatrix_results = calculate_nmatrix(trades_df, min_date, max_date, starting_balance, optimize=False)
        pbar.update(1)
<<<<<<< Updated upstream
        
        action_distribution = {
            'do_nothing': df['do_nothing'].mean() if 'do_nothing' in df.columns else 0,
            'go_long': df['go_long'].mean() if 'go_long' in df.columns else 0,
            'go_short': df['go_short'].mean() if 'go_short' in df.columns else 0
        }
        
=======

        # --- SECOND UPDATED SECTION: action distribution from action or one-hot columns ---
        if 'action' in df.columns:
            action_distribution = (
                df['action']
                .value_counts(normalize=True)
                .reindex(['do_nothing', 'go_long', 'go_short'], fill_value=0.0)
                .to_dict()
            )
        else:
            action_distribution = {
                'do_nothing': df['do_nothing'].mean() if 'do_nothing' in df.columns else 0,
                'go_long': df['go_long'].mean() if 'go_long' in df.columns else 0,
                'go_short': df['go_short'].mean() if 'go_short' in df.columns else 0
            }
        # ------------------------------------------------------------------------

>>>>>>> Stashed changes
        print("\n📊 Compiling final report...")
        report = {
            'rewards': rewards,
            'total_trades': len(df),
            'action_distribution': action_distribution,
            'avg_reward': np.mean(rewards),
            'max_drawdown': calculate_max_drawdown(rewards),
            'cumulative_reward': np.sum(rewards),
            'alpha': nmatrix_results.get('signed_alpha'),
            'burke': nmatrix_results.get('metrics', {}).get('burke'),
            'entropy': nmatrix_results.get('metrics', {}).get('entropy'),
            'kelly_risk': nmatrix_results.get('metrics', {}).get('kelly_risk')
        }
        pbar.update(1)
    
    print("\n✨ Analysis complete! Generating report...\n")
    return report

def generate_performance_report(csv_path: str, starting_balance: float) -> dict:
    """
    Generate comprehensive performance report from trading history
    """
    try:
        report = process_trading_history(csv_path, starting_balance)
        
        # Print report with fancy formatting
        print("\n" + "="*50)
        print("🚀 Trading Performance Report 🚀")
        print("="*50)
        
        metrics_emoji = {
            'rewards': '💰',
            'total_trades': '🔄',
            'action_distribution': '📊',
            'avg_reward': '📈',
            'max_drawdown': '📉',
            'cumulative_reward': '💎',
            'alpha': '🎯',
            'burke': '🛡️',
            'entropy': '🧩',
            'kelly_risk': '⚖️',
            'avg_long_duration': '⏳',
            'avg_short_duration': '⏳',
            'short_trades': '📉',
            'long_trades': '📈',
            'short_profit': '💹',
            'long_profit': '💹',
            'win_rate': '🎯',
            'short_win_rate': '🎯',
            'long_win_rate': '🎯',
            'avg_profit_per_trade': '💵',
            'avg_loss_per_trade': '💸',
            'profit_factor': '⚖️',
            'risk_reward_ratio': '⚡'
        }

        df = pd.read_csv(csv_path)
        # Calculate additional metrics
        state_components_mask = df['state'].str.extract(r'\(([^,)]+),\s*([^,)]+),\s*([^,)]+)(?:,\s*(\d+))?\)')
        short_mask = pd.to_numeric(state_components_mask[3], errors='coerce').fillna(0).astype(int) == 1
        long_mask = ~short_mask
        
        # Trade counts
        report['short_trades'] = short_mask.sum()
        report['long_trades'] = long_mask.sum()
        
        # Profit calculations
        short_rewards = df[short_mask]['avg_reward'].values
        long_rewards = df[long_mask]['avg_reward'].values
        
        report['short_profit'] = np.sum(short_rewards)
        report['long_profit'] = np.sum(long_rewards)
        
        # Win rates
        report['win_rate'] = (df['avg_reward'] > 0).mean()
        report['short_win_rate'] = (short_rewards > 0).mean() if len(short_rewards) > 0 else 0
        report['long_win_rate'] = (long_rewards > 0).mean() if len(long_rewards) > 0 else 0
        
        # Profit/Loss metrics
        profitable_trades = df[df['avg_reward'] > 0]['avg_reward']
        losing_trades = df[df['avg_reward'] < 0]['avg_reward']
        
        report['avg_profit_per_trade'] = profitable_trades.mean() if len(profitable_trades) > 0 else 0
        report['avg_loss_per_trade'] = losing_trades.mean() if len(losing_trades) > 0 else 0
        
        # Risk metrics
        report['profit_factor'] = abs(profitable_trades.sum() / losing_trades.sum()) if len(losing_trades) > 0 else float('inf')
        report['risk_reward_ratio'] = abs(report['avg_profit_per_trade'] / report['avg_loss_per_trade']) if report['avg_loss_per_trade'] != 0 else float('inf')
        
        # Calculate duration metrics
        def calculate_durations_from_states(state_str):
            if pd.isna(state_str) or not isinstance(state_str, str):
                return 0, 0
            # Extract is_short (the 4th element in each tuple)
            import re
            matches = re.findall(r'\([^,)]+,\s*[^,)]+,\s*[^,)]+,\s*(\d+)\)', state_str)
            if not matches:
                return 0, 0
            
            is_shorts = [int(m) for m in matches]
            from itertools import groupby
            groups = [(key, sum(1 for _ in group)) for key, group in groupby(is_shorts)]
            longs = [g[1] for g in groups if g[0] == 0]
            shorts = [g[1] for g in groups if g[0] == 1]
            return (sum(longs)/len(longs) if longs else 0), (sum(shorts)/len(shorts) if shorts else 0)

        # Apply calculation to each episode
        durations = df['state'].apply(calculate_durations_from_states)
        report['avg_long_duration'] = durations.apply(lambda x: x[0]).mean()
        report['avg_short_duration'] = durations.apply(lambda x: x[1]).mean()

        # Print metrics with appropriate formatting for each type
        for metric, value in report.items():
            emoji = metrics_emoji.get(metric, '📌')
            if metric == 'action_distribution':
                print(f"\n{emoji} {metric}:")
                for action, pct in value.items():
                    print(f"   {action}: {pct:.2%}")
            elif metric == 'rewards':
                print(f"{emoji} {metric}: [array of {len(value)} values]")
            elif metric in ['total_trades', 'short_trades', 'long_trades']:
                print(f"{emoji} {metric}: {value}")
            elif 'duration' in metric:
                print(f"{emoji} {metric.replace('_', ' ').title()}: {value:.2f} steps")
            elif isinstance(value, (float, np.float64, np.float32)):
                print(f"{emoji} {metric}: {value:.4f}")
            else:
                print(f"{emoji} {metric}: {value}")
        
        print("="*50 + "\n")
        return report
        
    except FileNotFoundError:
        print(f"❌ Error: Trading history file not found at {csv_path}")
        raise
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")
        raise

def generate_episode_report(episode_data: pd.DataFrame, starting_balance: float, episode_num: int) -> dict:
    """
    Generate performance report for a single training episode
    """
    try:
        report = process_trading_history(episode_data, starting_balance)
        report['episode_number'] = episode_num
        
        # Print episode-specific header
        print(f"\n{'='*50}")
        print(f"🎮 Episode {episode_num} Performance Report 🎮")
        print(f"{'='*50}")
        
        # Rest of the reporting logic remains the same...
        return report
    except Exception as e:
        print(f"❌ Error generating episode report: {str(e)}")
        raise

def save_episode_reports(reports: List[dict], output_path: str):
    """
    Save all episode reports to a CSV file
    """
    try:
        df = pd.DataFrame(reports)
        df.to_csv(output_path, index=False)
        print(f"✅ Episode reports saved to {output_path}")
    except Exception as e:
        print(f"❌ Error saving episode reports: {str(e)}")
        raise

def action_reward(csv_path: str, starting_balance: float, action: str, is_short: int):
    train_data = pd.read_csv(csv_path)
    m = train_data[(train_data['predicted_action'] == f'{action}') & (train_data['is_short'] == is_short)]
    counts = m['is_short'].value_counts()
    total_reward = m['reward'].cumsum()[-1:].values[0]
    wins = len(m[m['reward'] > 0])
    losses = len(m[m['reward'] <= 0])
    return {
        'counts': counts.get(is_short),
        'total reward': total_reward,
        'winrate': f'{wins * 100 / (losses + wins):.2f}%',
        'per trade profit': m[m['reward'] > 0]['reward'].sum() / wins,
        'per trade loss': m[m['reward'] <= 0]['reward'].sum() / losses
    }

def show_action_reward(action: str, is_short: int):
    action_mapping = {"go_long": 0, "go_short": 1, "do_nothing": 2}
    dirs = [0,1]
    for action in action_mapping.keys():
        for is_short in dirs:
            try:
                print(f'{action} {is_short}: {action_reward(action, is_short)}')
            except IndexError as e:
                print(e)

if __name__ == "__main__":
    csv_path = "user_data/reports/episode_reports.csv"
    starting_balance = 10000.0
    #process_trading_history(csv_path, starting_balance)
    generate_performance_report(csv_path, starting_balance)
