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

def extract_last_is_short(state_str):
    """Extract is_short value from the LAST tuple in a state sequence string"""
    try:
        if not isinstance(state_str, str): return 0
        import re
        # Match the last element of the last tuple: (..., value)
        matches = re.findall(r',\s*(\d+)\)', state_str)
        if matches:
            return int(matches[-1])
        return 0
    except:
        return 0

def infer_action_distribution(df: pd.DataFrame) -> Dict[str, float]:
    """Return action distribution from actual trade actions when available."""
    if 'action' in df.columns:
        counts = df['action'].fillna('unknown').value_counts(normalize=True).to_dict()
        return {action: float(counts.get(action, 0.0)) for action in ['do_nothing', 'go_long', 'go_short']}

    return {
        'do_nothing': float(df['do_nothing'].fillna(0).mean()) if 'do_nothing' in df.columns else 0.0,
        'go_long': float(df['go_long'].fillna(0).mean()) if 'go_long' in df.columns else 0.0,
        'go_short': float(df['go_short'].fillna(0).mean()) if 'go_short' in df.columns else 0.0
    }

def determine_actual_action(df: pd.DataFrame) -> pd.Series:
    """Resolve actual actions from a detailed trade log or episode summary fallback."""
    if 'action' in df.columns:
        return df['action'].fillna('unknown')

    action_cols = [c for c in ['do_nothing', 'go_long', 'go_short'] if c in df.columns]
    if action_cols:
        valid_action_cols = [c for c in action_cols if not df[c].isna().all()]
        if valid_action_cols:
            return df[valid_action_cols].fillna(0).idxmax(axis=1)

    return pd.Series(['unknown'] * len(df), index=df.index)

def process_trading_history(csv_path: str, starting_balance: float) -> dict:
    """
    Process trading history CSV with columns: action, reward, state, next_state
    (Minimal update: only the two requested sections and timestamp frequency fix)
    """
    print("\n🤖 Initializing trading history analysis...\n")
    time.sleep(0.5)

    # Read trading history with progress bar
    print("📊 Loading trading data...")
    df = pd.read_csv(csv_path)
    print(f"📈 Loaded {len(df)} trading entries\n")
    
    with tqdm(total=6, desc="🔄 Processing metrics", ncols=100) as pbar:
        # Get rewards directly from the 'reward' column
        reward_col = 'avg_reward' if 'avg_reward' in df.columns else 'reward'
        rewards = df[reward_col].values
        pbar.update(1)
        time.sleep(0.2)

        # Create timestamps for nmatrix calculation (fixed freq='h')
        print("\n⏰ Generating temporal analysis framework...")
        df['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(days=len(df)),
            periods=len(df),
            freq='h'
        )
        pbar.update(1)
        time.sleep(0.2)

        # Extract is_short from state tuples (last tuple in the sequence)
        print("🎯 Extracting position states...")
        df['inferred_is_short'] = df['state'].apply(extract_last_is_short)

        # Create trades DataFrame with all state components
        trades_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'profit_abs': df[reward_col] * starting_balance,
            'is_short': df['inferred_is_short']
        })

        trades_df['action'] = determine_actual_action(df)

        pbar.update(1)
        time.sleep(0.2)

        print("\n🧮 Calculating nmatrix score...")
        min_date = trades_df['timestamp'].min()
        max_date = trades_df['timestamp'].max()
        nmatrix_score = calculate_nmatrix(trades_df, min_date, max_date, starting_balance)
        pbar.update(1)

        action_distribution = infer_action_distribution(df)

        print("\n📊 Compiling final report...")
        report = {
            'rewards': rewards,
            'total_trades': len(df),
            'action_distribution': action_distribution,
            'avg_reward': np.mean(rewards),
            'max_drawdown': calculate_max_drawdown(rewards),
            'cumulative_reward': np.sum(rewards),
            'nmatrix_score': nmatrix_score.get('signed_alpha') if nmatrix_score else None,
            'history_source': 'trade_history' if 'action' in df.columns else 'episode_report_summary'
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
            'history_source': '🧭',
            'action_distribution': '📊',
            'avg_reward': '📈',
            'max_drawdown': '📉',
            'cumulative_reward': '💎',
            'nmatrix_score': '🎯',
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
        df['is_short_val'] = df['state'].apply(extract_last_is_short)
        df['actual_action'] = determine_actual_action(df)
        short_mask = df['is_short_val'] == 1
        long_mask = df['is_short_val'] == 0
        
        # Trade counts
        report['short_trades'] = short_mask.sum()
        report['long_trades'] = long_mask.sum()
        
        # Profit calculations
        reward_col = 'avg_reward' if 'avg_reward' in df.columns else 'reward'
        short_rewards = df[short_mask][reward_col].values
        long_rewards = df[long_mask][reward_col].values
        
        report['short_profit'] = np.sum(short_rewards)
        report['long_profit'] = np.sum(long_rewards)
        
        # Win rates
        report['win_rate'] = (df[reward_col] > 0).mean()
        report['short_win_rate'] = (short_rewards > 0).mean() if len(short_rewards) > 0 else 0
        report['long_win_rate'] = (long_rewards > 0).mean() if len(long_rewards) > 0 else 0
        
        # Profit/Loss metrics
        profitable_trades = df[df[reward_col] > 0][reward_col]
        losing_trades = df[df[reward_col] < 0][reward_col]
        
        report['avg_profit_per_trade'] = profitable_trades.mean() if len(profitable_trades) > 0 else 0
        report['avg_loss_per_trade'] = losing_trades.mean() if len(losing_trades) > 0 else 0
        
        # Risk metrics
        report['profit_factor'] = abs(profitable_trades.sum() / losing_trades.sum()) if len(losing_trades) > 0 else float('inf')
        report['risk_reward_ratio'] = abs(report['avg_profit_per_trade'] / report['avg_loss_per_trade']) if report['avg_loss_per_trade'] != 0 else float('inf')
        
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
            elif isinstance(value, (float, np.float64, np.float32)):
                print(f"{emoji} {metric}: {value:.4f}")
            else:
                print(f"{emoji} {metric}: {value}")
        
        # --- NEW SECTION: Dominant Actions for Winning Trades ---
        print("\n🏆 Dominant Actions for Winning Trades:")
        
        # Determine action taken
        winning_df = df[df[reward_col] > 0]
        if not winning_df.empty:
            for pos_type, mask in [('Long', long_mask), ('Short', short_mask)]:
                pos_winners = winning_df[mask[winning_df.index]]
                if not pos_winners.empty:
                    print(f"   {pos_type} winners dominant actions:")
                    counts = pos_winners['actual_action'].value_counts()
                    total_pos_winners = len(pos_winners)
                    for action, count in counts.items():
                        print(f"     - {action}: {count} trades ({count/total_pos_winners:.1%})")
                else:
                    print(f"   {pos_type} winners: No profitable trades found")
        else:
            print("   No winning trades found in the history.")
        # ---------------------------------------------------------

        print("="*50 + "\n")
        return report
        
    except FileNotFoundError:
        print(f"❌ Error: Trading history file not found at {csv_path}")
        raise
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")
        raise

def action_reward(csv_path: str, action: str, is_short: int):
    """Calculate reward metrics for a specific action and position type"""
    train_data = pd.read_csv(csv_path)
    
    # Handle reward column
    reward_col = 'reward' if 'reward' in train_data.columns else 'avg_reward'
    
    # Consistent is_short extraction
    train_data['is_short_val'] = train_data['state'].apply(extract_last_is_short)

    # Determine action taken
    if 'action' in train_data.columns:
        train_data['actual_action'] = train_data['action']
    else:
        action_cols = [c for c in ['do_nothing', 'go_long', 'go_short'] if c in train_data.columns]
        if action_cols:
            valid_action_cols = [c for c in action_cols if not train_data[c].isna().all()]
            train_data['actual_action'] = train_data[valid_action_cols].fillna(0).idxmax(axis=1) if valid_action_cols else 'unknown'
        else:
            train_data['actual_action'] = 'unknown'

    m = train_data[(train_data['actual_action'] == f'{action}') & (train_data['is_short_val'] == is_short)]
    if len(m) == 0:
        return "No data"

    total_reward = m[reward_col].sum()
    wins = len(m[m[reward_col] > 0])
    losses = len(m[m[reward_col] <= 0])
    
    return {
        'count': len(m),
        'total_reward': f'{total_reward:.4f}',
        'winrate': f'{wins * 100 / (losses + wins):.2f}%' if (losses + wins) > 0 else '0.00%',
        'avg_profit': f'{m[m[reward_col] > 0][reward_col].mean():.4f}' if wins > 0 else '0',
        'avg_loss': f'{m[m[reward_col] <= 0][reward_col].mean():.4f}' if losses > 0 else '0'
    }

def show_action_reward(csv_path: str):
    """Show reward metrics for all possible actions and positions"""
    action_mapping = ["go_long", "go_short", "do_nothing"]
    dirs = [0, 1]
    print("\n📊 Detailed Action/Position Reward Analysis:")
    print("-" * 45)
    for action in action_mapping:
        for is_short in dirs:
            pos_name = "Short" if is_short == 1 else "Long"
            result = action_reward(csv_path, action, is_short)
            print(f"{action} ({pos_name}): {result}")

if __name__ == "__main__":
    trade_history_path = "user_data/reports/trade_history.csv"
    csv_path = trade_history_path if os.path.exists(trade_history_path) else "user_data/reports/episode_reports.csv"
    starting_balance = 10000.0
    generate_performance_report(csv_path, starting_balance)
    try:
        show_action_reward(csv_path)
    except Exception as e:
        print(f"\n💡 Note: Detailed action analysis skip or failed: {e}")
