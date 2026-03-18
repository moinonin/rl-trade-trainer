Reinforcement Learning Trading Workflow Documentation
=====================================================

Quick Start
-----------

To run the main trading and training orchestration:

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. Run the bid agent
python core/bid_agent.py
```

The system will automatically search for historical training data (matching `*_large_83rl_mod_balanced.csv`). If no files are found, it will **automatically skip to live mode** using the Binance exchange.

Note
----

The codebase has been reorganized to focus on the pretrained-signal and RL pipeline. Core components now live under `core/`:

- `core/pretrained/` for pretrained signal generation
- `core/rl/` for RL agent + trainer (and PKL artifacts)
- `core/data/` for data fetching/indicator prep
- `core/metrics/` and `core/optimize/` for evaluation/optimization helpers
- `core/bid_agent.py` as the main training/orchestration entry point

Overview
--------

This workflow implements a complete reinforcement learning (RL) trading system that generates trading signals, backtests strategies, balances training data, optimizes RL parameters, and selects optimal trading artifacts for production deployment.

Architecture Flow
-----------------

```text
   Strategy Generation → Backtesting → Data Sampling → RL Optimization → Analysis
      (Strategy01)    (backtester)   (sampler)    (train_param_optimizer) (analyze_rl_stats)
```

Component Details
-----------------

### 1. Strategy01.py - Signal Generation Core

**Purpose**: Generates fundamental trading signals from market data by identifying entry and exit points for long and short positions.

**Key Features**:
* Vectorized trade generation using pandas DataFrames
* Dual signal system for long and short positions
* Configurable lookback windows (long_ml_candle, short_ml_candle)
* Exit logic with profit-taking and stop-loss conditions
* Trade duration and PnL calculation

### 2. backtester.py - Multi-Pair Backtesting Engine

**Purpose**: Extends strategy to actual exchange data, runs continuous backtesting on multiple pairs, and collects training data for RL.

**Key Features**:
* Fetches real-time OHLCV data from exchanges (Bybit supported)
* Computes technical indicators
* Generates signals using pre-trained RL models
* Continuous backtesting loop with configurable intervals

### 3. sampler.py - Training Data Balancer

**Purpose**: Addresses class imbalance in trading data by upsampling minority position types.

### 4. train_param_optimizer.py - RL Hyperparameter Optimization

**Purpose**: Optimizes Q-learning parameters using Bayesian optimization and generates RL artifacts.

**Key Features**:
* Next-state augmented Q-learning with discrete state spaces
* Bayesian optimization using scikit-optimize (skopt)
* Quantile-based state discretization

### 5. analyze_rl_stats.py - Performance Analyzer

**Purpose**: Analyzes RL training results and selects optimal artifacts for production.

File Structure
--------------

```text
   core/rl/
  ├── rl_training_stats.csv          # Raw trade data from backtesting
  ├── balanced_rl_training_stats.csv # Balanced training data
  ├── state_to_index.pkl             # State mapping
  ├── episode_transitions.pkl        # Training transitions
  ├── q_table.pkl                    # Optimized Q-table
  ├── final_trades.csv               # Evaluation results
  └── agent_config.json              # Optimal hyperparameters
```

Dependencies
------------

```text
   pandas
   numpy
   scikit-learn
   scikit-optimize
   arrow
   tqdm
   fire
   pickle
   json
   logging
   python-dotenv
   nlp-project
   defirl
```

Usage Workflow
--------------

### Step 1: Strategy Development
* Modify Strategy01.py to adjust entry/exit logic
* Test with historical data

### Step 2: Backtesting
```bash
python backtester.py
```
* Runs continuous backtesting on configured pairs
* Generates rl_training_stats.csv

### Step 3: Data Preparation
```bash
python sampler.py
```
* Balances the training dataset
* Outputs balanced_rl_training_stats.csv

### Step 4: RL Optimization
```bash
python train_param_optimizer.py
```
* Runs Bayesian optimization
* Generates Q-tables and configuration

### Step 5: Analysis
```bash
python analyze_rl_stats.py
```
* Evaluates performance metrics
* Identifies best strategies

Performance Metrics
-------------------

The system evaluates strategies based on:
1. **Total PnL**: Sum of all profits/losses
2. **Win Rate**: Percentage of profitable trades
3. **Profit Factor**: Gross profit / gross loss
4. **Max Drawdown**: Largest peak-to-trough decline
5. **Alpha Score**: Risk-adjusted performance metric (via N-Matrix)
6. **Trade Duration**: Average holding period

Production Deployment
---------------------

For production use:
1. Extract optimal Q-table from core/rl/q_table.pkl
2. Load configuration from agent_config.json
3. Integrate with live trading infrastructure

Security Considerations
-----------------------
* Secure API keys for exchange access (use `.env` file)
* Encrypt sensitive configuration files
* Implement rate limiting for exchange API calls

Support
-------
For issues or questions:
1. Check log files for error messages
2. Validate input data formats
3. Ensure all dependencies are correctly installed

_This documentation covers version 1.1 of the RL trading workflow._
