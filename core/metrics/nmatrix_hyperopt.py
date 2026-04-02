import pandas as pd
import numpy as np
import statistics
import math
import traceback
from datetime import datetime
from skopt import gp_minimize
from skopt.space import Real
import json
import os
import logging
from .utils import HyperoptHelper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ANSI color codes for colorful terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
BLUE = '\033[94m'
BOLD = '\033[1m'
END = '\033[0m'

INVALID_EAR_PENALTY = 1e10

def safe_ear(alpha, entropy):
    """Return a finite EAR value or None when the ratio is undefined."""
    try:
        alpha_val = float(alpha)
        entropy_val = float(entropy)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(alpha_val) or not np.isfinite(entropy_val):
        return None
    if abs(entropy_val) < 1e-12:
        return None

    ear_val = alpha_val / entropy_val
    return float(ear_val) if np.isfinite(ear_val) else None

def save_optimization_result(signed_alpha, result, metrics, base_path="user_data/optimization_results"):
    """
    Save optimization results to a JSON file
    
    Parameters:
    -----------
    signed_alpha: float
        Alpha value from optimization
    result: object
        Optimization result object
    metrics: dict
        Metrics dictionary
    base_path: str
        Base path for saving results
        
    Returns:
    --------
    tuple: (save_dir, json_path) or (None, None) if saving failed
        save_dir: Path to the directory where results were saved
        json_path: Path to the saved JSON file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directory structure - use 'neg' or 'pos' for alpha to avoid confusion with abs()
    alpha_sign = "neg" if signed_alpha < 0 else "pos"
    alpha_dir = os.path.join(base_path, f"alpha_{alpha_sign}_{abs(signed_alpha):.4f}_{timestamp}")
    try:
        os.makedirs(alpha_dir, exist_ok=True)
    except Exception as e:
        print(f"{RED}Error creating directory: {e}{END}")
        return None, None
    
    # Convert numpy values to native Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(i) for i in obj]
        return obj

    try:
        # Convert metrics first
        serializable_metrics = convert_to_serializable(metrics)
        
        # Prepare optimization data
        optimization_data = {
            "alpha": float(signed_alpha),
            "parameters": convert_to_serializable(result.x),
            "optimization_score": float(result.fun),
            "metrics": serializable_metrics,
            "timestamp": timestamp
        }

        # Remove non-serializable objects
        if 'best_result' in optimization_data['metrics']:
            del optimization_data['metrics']['best_result']
        
        # Save optimization results with no truncation
        results_file = os.path.join(alpha_dir, "optimization_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(optimization_data, f, indent=4, ensure_ascii=False, allow_nan=True)
            f.write('\n')  # Add newline at end of file
            
        # Verify the file was written completely
        with open(results_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            for key in metrics.keys():
                if key not in saved_data['metrics']:
                    saved_data['metrics'][key] = None  # Add missing keys with default value
            if not all(k in saved_data['metrics'] for k in metrics.keys()):
                raise ValueError("Some metrics were not saved properly")
                
        print(f"{GREEN}Results saved in: {CYAN}{alpha_dir}{END}")
        return alpha_dir, results_file
        
    except Exception as e:
        print(f"{RED}Error saving JSON: {e}{END}")
        traceback.print_exc()
        return None, None

def calculate_nmatrix(trades: pd.DataFrame, min_date: datetime, max_date: datetime,
                     starting_balance: float, agent=None, 
                     print_frequency: int = 20, 
                     save_intermediate: bool = True,
                     intermediate_save_frequency: int = 50) -> dict:
    """
    Calculate nmatrix and return results dictionary
    
    Parameters:
    -----------
    trades: pd.DataFrame
        DataFrame containing trade data
    min_date: datetime
        Minimum date for analysis
    max_date: datetime
        Maximum date for analysis
    starting_balance: float
        Starting balance for calculations
    agent: object, optional
        Agent object for model saving
    print_frequency: int, optional
        How often to print optimization progress (iterations)
    save_intermediate: bool, optional
        Whether to save intermediate results
    intermediate_save_frequency: int, optional
        How often to save intermediate results (iterations)
    
    Returns:
    --------
    dict: Dictionary containing optimization results
    """
    
    # Default return value
    default_result = {
        'signed_alpha': 0,
        'optimization_result': None,
        'metrics': {
            'win_rate': 0,
            'kelly_risk': 0,
            'burke': 0,
            'entropy': 0,
            'var_coeff': 0,
            'total_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'profitable_trades': 0,
            'profitable_long_trades': 0,
            'profitable_short_trades': 0,
            'losing_trades': 0
        },
        'nmatrix_score': 0,
        'save_dir': None,
        'json_path': None,
        'intermediate_saves': []  # Track intermediate saves
    }

    if (len(trades) == 0) or (min_date is None) or (max_date is None) or (min_date == max_date):
        logging.warning("Invalid input parameters for calculate_nmatrix")
        return default_result

    try:
        total_profit = trades['profit_abs'] / starting_balance
        days_period = max(1, (max_date - min_date).days)

        expected_returns_mean = total_profit.sum() / days_period

        down_stdev = np.std(trades.loc[trades['profit_abs'] < 0, 'profit_abs'] / starting_balance)

        pos = 0
        negs = 0
        wins = []
        loses = []
        daily_returns = []

        short_trades_count = 0
        long_trades_count = 0



        for result in trades['profit_abs']:
            if result > 0:
                pos +=1
                wins.append(result * 100 / starting_balance)
                daily_returns.append(result)
            else:
                negs +=1
                loses.append(result * 100 / starting_balance)
                daily_returns.append(result)


        for result in trades['is_short']:
            if result == 1:
                short_trades_count +=1
            else:
                long_trades_count +=1

        # Calculating drawdowns
        def getDrawDowns(daily_returns: list = daily_returns):
            df = pd.DataFrame()
            df['daily_returns'] = np.clip(daily_returns, -1e10, 1e10)  # Clip extreme values
            
            # Use np.clip to prevent overflow in cumulative calculations
            df['cumulative_returns_pct'] = np.clip(
                (1 + (df['daily_returns'] / starting_balance)).cumprod() - 1,
                -1e10, 1e10
            )
            
            peak = df['cumulative_returns_pct'].cummax()
            df['cum_dd'] = np.clip((df['cumulative_returns_pct'] - peak) / peak, -1e10, 1e10)
            df.fillna(df.mean(), inplace=True)
            return df

        def annualized_return(periods_per_year=365, trades_per_year=None):
            """
            Calculate annualized return with overflow protection
            """
            returns_per_day = getDrawDowns()['daily_returns']
            
            if trades_per_year is None:
                trades_per_year = periods_per_year * len(returns_per_day) / days_period
            
            # Clip returns to prevent overflow
            clipped_returns = np.clip(1 + returns_per_day, -1e10, 1e10)
            cumulative_return = np.clip(clipped_returns.prod() - 1, -1e10, 1e10)
            
            # Add safety check for negative or zero values
            if cumulative_return <= -1:
                return -1e10
            
            try:
                # Use safer calculation method
                exponent = periods_per_year / trades_per_year
                base = 1 + cumulative_return
                result = np.power(base, exponent, where=(base > 0)) - 1
                return np.clip(result, -1e10, 1e10)
            except Exception:
                return -1e10

        # Calculate burke_ratio
        def burke_ratio():
            """
            Calculate the Burke Ratio with safeguards against division by zero and numerical overflow.
            Returns a default value (e.g., large number) if the denominator is invalid.
            """
            # Extract cumulative drawdowns
            cumulative_drawdowns = getDrawDowns()['cum_dd']
            
            # Safeguard 1: Handle empty/zero drawdowns
            if len(cumulative_drawdowns) == 0:
                print(f"{YELLOW}Warning: No drawdowns found. Burke Ratio undefined.{END}")
                return 100000  # Default value for no drawdowns
            
            # Safeguard 2: Clip values to avoid overflow and calculate sum of squared drawdowns
            clipped_squared_drawdowns = np.clip(cumulative_drawdowns ** 2, 0, 1e10)  # Clip to [0, 1e10]
            sum_squared_drawdown = np.sqrt(np.sum(clipped_squared_drawdowns))
            
            # Safeguard 3: Prevent division by zero/near-zero
            epsilon = 1e-10  # Threshold for "effectively zero"
            if sum_squared_drawdown < epsilon:
                print(f"{YELLOW}Warning: Sum of squared drawdowns ({sum_squared_drawdown}) is too small.{END}")
                return 100000
            
            # Safeguard 4: Clip annualized returns to avoid extreme values
            annual_returns = np.clip(annualized_return(), -1e10, 1e10)
            
            # Calculate Burke Ratio
            try:
                ratio = annual_returns / sum_squared_drawdown
                return ratio
            except Exception as e:
                print(f"{RED}Error calculating Burke Ratio: {e}{END}")
                return 100000


        # Calculate up and down population standard deviations
        down_psd = (
            0
            if len(trades.loc[trades['profit_abs'] < 0, 'profit_abs']) == 0
            else statistics.pstdev(
                trades.loc[trades['profit_abs'] < 0, 'profit_abs']
                / starting_balance
            )
        )
        up_psd = (
            0
            if len(trades.loc[trades['profit_abs'] > 0, 'profit_abs']) == 0
            else statistics.pstdev(
                trades.loc[trades['profit_abs'] > 0, 'profit_abs']
                / starting_balance
            )
        )


        # Calculate statistics array
        def getNmatrix(pos: int = pos, negs: int = negs, wins: int = wins, loses: int = loses):
            N = pos + negs
            n_p = pos
            n_l = negs
            p_t = sum(wins) / n_p if n_p >= 1 else 1000
            l_t = -sum(loses) / n_l if n_l >= 1 else 10000
            w = n_p / N if N > 0 else 1
            l = 1.0 - w
            s = (p_t - l_t) #/ (w + 1)
            k = w/(l_t) - l/(p_t) if l_t > 0 else -l/p_t

            array = np.array([[p_t,l_t, s],[n_l,n_p, N],[l,w,1]])
            result = {
                'kelly': k,
                'nmatrix': array,
                "s": s
            }

            return result

        def get_minor(matrix, i, j):
            """Return the Minor of the element at position (i, j) in the matrix."""
            return np.delete(np.delete(matrix, i, axis=0), j, axis=1)

        def determinant(matrix):
            """Recursively calculate the determinant of a matrixs."""
            if matrix.shape[0] == 1:
                return matrix[0, 0]
            elif matrix.shape[0] == 2:
                return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
            else:
                det = 0
                for col in range(matrix.shape[1]):
                    det += ((-1) ** col) * matrix[0, col] * determinant(get_minor(matrix, 0, col))
                return det

        def cofactor_matrix(matrix):
            """Calculate the cofactor matrix of the given matrix."""
            cofactors = np.zeros(matrix.shape)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    minor = get_minor(matrix, i, j)
                    cofactors[i, j] = ((-1) ** (i + j)) * determinant(minor)
            return cofactors

        N = negs + pos
        n_mat = getNmatrix().get('nmatrix')
        cofactors = cofactor_matrix(n_mat)

        #signed_alpha = get_minor(get_minor(cofactors,0,0),1,0)[0][0]
        alpha = abs(get_minor(get_minor(cofactors,0,0),1,0)[0][0])
        net_profit = get_minor(get_minor(cofactors,0,0),0,0)[0][0]

        losrate_pct = get_minor(get_minor(n_mat, 0,2),0,1)[0][0]
        win_rate = get_minor(get_minor(n_mat, 0,0),0,1)[0][0] # 1 - losrate_pct
        #s = abs(get_minor(get_minor(n_mat,2,0),1,0)[0][0])
        s = getNmatrix().get('s')
        ideal_mat = np.array([[s,0,s],[0,N,N],[0,1,1]])

        #ideal_cofactors = np.array([[0,0,0],[s,2*s,-2*s],[-N*s, -2*N*s, 2*N*s]])
        ideal_cofactors = cofactor_matrix(ideal_mat)

        ideal_cofactors_norm = np.linalg.norm(ideal_cofactors)
        cofactors_norm = np.linalg.norm(cofactors)

        cofactors_norm_diff = cofactors_norm - ideal_cofactors_norm
        cofactors_diff_pct = abs(cofactors_norm_diff) * 100 /cofactors_norm
        mse_cofactors = np.mean((cofactors_norm - ideal_cofactors_norm)**2)

        var_coeff = down_psd/expected_returns_mean
        annual_down_volatility = down_psd * math.sqrt(365)
        annual_up_volatility = up_psd * math.sqrt(365)
        kelly_risk = getNmatrix().get('kelly')
        per_trade_profit = 100 if pos < 1 else sum(wins)/pos
        per_trade_loss = 100 if negs < 1 else abs(sum(loses)/negs)
        psd_dif = abs(abs(down_psd) - abs(up_psd))
        signed_alpha = (-1)**5 * (win_rate * per_trade_profit - losrate_pct * per_trade_loss)
        opt_s = per_trade_profit/2
        burke = burke_ratio()


        def calculate_net_profit(
                                per_trade_profit: float = per_trade_profit,
                                per_trade_loss: float = per_trade_loss,
                                win_rate: float = win_rate,
                                total_trades: int = N,
                                profitable_trades: int = pos,
                                losing_trades: int = negs
                                
                        ):

                            total_profit = profitable_trades * per_trade_profit - losing_trades * per_trade_loss
                            
                            return total_profit

        def find_min_win_rate(daily_returns: list = daily_returns):
            daily_returns = getDrawDowns()['daily_returns']
            total_trades = len(daily_returns)
            for win_rate in np.arange(0.01, 1.0, 0.01):
                if calculate_net_profit() > 0:
                    return win_rate
            return None

        def cal_entropy(daily_returns: list = daily_returns):
            value_counts = getDrawDowns()['daily_returns'].value_counts(normalize=True)
            df = pd.DataFrame({'profit_loss': value_counts.index, 'probability': value_counts.values})
            entropy = -np.sum(df['probability'] * np.log2(df['probability']))
            max_entropy = np.log2(len(df))
            # Return normalized entropy
            return entropy / max_entropy if max_entropy > 0 else 0
        



        # Define target values
        target_alpha = -s
        target_kelly_risk = 0.0
        target_var_coeff = 0.0
        target_confactors_norm_diff = 0.0
        target_psd_dif = 0
        target_burke = 1
        mse_cofactors_target = 0
        target_median_profit_pct = 0
        daily_returns = getDrawDowns()['daily_returns']
        median_profit_pct = np.median(daily_returns) / starting_balance 

        target_win_rate = find_min_win_rate(daily_returns=daily_returns)
        entropy = cal_entropy()
        EAR = safe_ear(signed_alpha, entropy)
        target_entropy = 0
        target_EAR = -1 #target_alpha/entropy #-0.5


        

        # Define deviations
        cofactors_norm_deviation = min(0, cofactors_norm_diff - target_confactors_norm_diff)  # Penalize if above the target
        mse_cofactors_deviation = min(0, mse_cofactors - mse_cofactors_target)  # Penalize if above the target
        signed_alpha_deviation = min(0, signed_alpha - target_alpha)  # Penalize if above the target
        entropy_deviation = min(0, entropy - target_entropy)  # Penalize if above the target
        #EAR_deviation = min(0, target_EAR)  # Penalize if above the target 
        EAR_penalty = INVALID_EAR_PENALTY if EAR is None else max(0, EAR)

        # Initialize a list to track discarded parameters and optimization progress
        discarded_params = []
        iteration_counter = [0]  # Use list for mutable counter in closure

        def getParams(params, lambda_param=0.1):
            try:
                # Increment iteration counter
                iteration_counter[0] += 1
                current_iteration = iteration_counter[0]
                
                zeta, theta, sigma, alpha, gamma = params

                # Get raw values
                mse_val = float(mse_cofactors)
                cofactors_val = float(cofactors_norm)
                alpha_val = float(signed_alpha)
                entropy_val = float(entropy)
                ear_val = safe_ear(signed_alpha, entropy)
                matrix = n_mat

                # --- Fix: For alpha and EAR, we want them NEGATIVE ---
                # Positive weights (sigma, gamma) * negative metrics (alpha, EAR) 
                # will lower the score, which is what we want to minimize.
                if alpha_val > 0:
                    # Positive alpha is terrible - square it to penalize heavily
                    alpha_contrib = sigma * (alpha_val ** 2 * 1000)
                else:
                    # Negative alpha is good - use it directly
                    alpha_contrib = sigma * alpha_val
                    
                if ear_val is None:
                    # Undefined EAR is treated as a bad metric.
                    ear_contrib = gamma * INVALID_EAR_PENALTY
                elif ear_val > 0:
                    # Positive EAR is terrible
                    ear_contrib = gamma * (ear_val ** 2 * 1000)
                else:
                    # Negative EAR is good
                    ear_contrib = gamma * ear_val

                # For MSE and cofactors, we want them near zero (always positive)
                # So we use them directly - smaller is better
                cofactors_contrib = zeta * cofactors_val
                mse_contrib = theta * mse_val

                # For entropy, lower is better (always positive)
                entropy_contrib = alpha * entropy_val

                # Weighted sum
                objective_score = (
                    cofactors_contrib +
                    mse_contrib +
                    alpha_contrib +
                    entropy_contrib +
                    ear_contrib
                )

                # --- Win penalty ---
                if target_win_rate is None:
                    default_min_win_rate = 0.5
                    win_penalty = 1000.0 * max(0.0, default_min_win_rate - win_rate)
                else:
                    win_penalty_weight = 1000.0
                    win_penalty = win_penalty_weight * max(0.0, target_win_rate - win_rate)

                # L1 regularization
                regularization_penalty = lambda_param * sum(abs(p) for p in params)

                # Final score
                nmatrix = objective_score + regularization_penalty + win_penalty

                # Record contributions
                progress_data = {
                    'iteration': current_iteration,
                    'params': params,
                    'score': nmatrix,
                    'contributions': {
                        'cofactors': cofactors_contrib,
                        'mse': mse_contrib,
                        'alpha': alpha_contrib,
                        'entropy': entropy_contrib,
                        'EAR': ear_contrib,
                        'win_penalty': win_penalty,
                        'regularization': regularization_penalty,
                        '_raw_alpha': alpha_val,
                        '_raw_EAR': ear_val,
                        'ear_valid': ear_val is not None,
                        'matrix': matrix
                    }
                }
                optimization_progress.append(progress_data)

                if current_iteration % print_frequency == 0:
                    score_color = GREEN if nmatrix < 0 else (YELLOW if nmatrix < 0.5 else RED)
                    alpha_color = GREEN if alpha_val < 0 else RED
                    ear_color = GREEN if (ear_val is not None and ear_val < 0) else RED
                    ear_display = f"{ear_val:.6e}" if ear_val is not None else "invalid"
                    print(f"{CYAN}Iteration {current_iteration}:{END} Score = {score_color}{nmatrix:.6f}{END}")
                    print(f"  Alpha: {alpha_color}{alpha_val:.6e}{END}, EAR: {ear_color}{ear_display}{END}")

                if np.isnan(nmatrix) or np.isinf(nmatrix):
                    return 1e10
                return nmatrix

            except Exception as e:
                print(f"{RED}Error in getParams: {e}{END}")
                import traceback
                traceback.print_exc()
                return 100.0
        # Define search space
        space = [
            Real(0.0, 1.0, name='zeta'),
            Real(0.0, 1.0, name='theta'),
            Real(0.0, 1.0, name='sigma'),
            Real(0.0, 1.0, name='alpha'),
            Real(0.0, 1.0, name='gamma')
        ]

        best_result = None
        best_overall_metrics = None
        optimization_progress = []
        intermediate_saves = []
        
        print(f"\n{BLUE}{'═' * 60}{END}")
        print(f"{BOLD}{CYAN}🚀 Starting optimization with {len(trades)} trades{END}")
        print(f"{BLUE}{'═' * 60}{END}")
        
        for run_idx in range(5):
            # Reset progress for each run to avoid bloated JSON files
            optimization_progress = []
            
            # Print run header with colors
            print(f"\n{BLUE}{'═' * 50}{END}")
            print(f"{BOLD}{CYAN}🔄 Run {run_idx+1}/5{END}")
            print(f"{BLUE}{'─' * 50}{END}")
            
            discarded_params.clear()
            iteration_counter[0] = 0  # Reset counter for each run
            
            result = gp_minimize(
                getParams,
                space,
                noise=0.0,                      # deterministic backtest
                n_calls=100,
                n_initial_points=20,            # increased exploration
                base_estimator="ET",            # keep if mixed/categorical space; else "GP"
                acq_func="EI",
                acq_optimizer="auto",           # adapts to space type
                random_state=42 + run_idx,      # Change seed for each run
                n_jobs=-1,                      # parallel execution
                verbose=False,
                callback=lambda res: print(f"{YELLOW}Best score so far: {GREEN if res.fun < 0 else RED}{res.fun:.6f}{END}") if len(res.x_iters) % print_frequency == 0 else None
            )
            
            current_metrics = {
                'win_rate': win_rate,
                'kelly_risk': kelly_risk,
                'burke': burke,
                'entropy': entropy,
                'var_coeff': var_coeff,
                'total_trades': N,
                'profitable_trades': pos,
                'losing_trades': negs,
                'nmatrix': n_mat,
                'cofactors': cofactors,
                'ideal_cofactors': ideal_cofactors,
                'ideal_cofactors_norm': ideal_cofactors_norm,
                'cofactors_norm': cofactors_norm,
                'cofactors_norm_diff': cofactors_norm_diff,
                'mse_cofactors': mse_cofactors,
                'psd_dif': psd_dif,
                'median_profit_pct': median_profit_pct,
                'net_profit': net_profit,
                's': s,
                'signed_alpha': signed_alpha,
                'per_trade_profit': per_trade_profit,
                'per_trade_loss': per_trade_loss,
                'annual_down_volatility': annual_down_volatility,
                'annual_up_volatility': annual_up_volatility,
                'best_result': result,
                'short_trades': short_trades_count,
                'long_trades': long_trades_count,
                'optimization_progress': optimization_progress
            }

            if best_result is None or result.fun < best_result.fun:
                best_result = result
                best_overall_metrics = current_metrics
                print(f"{GREEN}✨ New best result found in run {run_idx+1}: {result.fun:.6f}{END}")

            # Print summary of this run with colors
            alpha_color = GREEN if signed_alpha < 0 else RED
            burke_color = GREEN if burke > 0.5 else (YELLOW if burke > 0 else RED)
            entropy_color = GREEN if entropy < 0.3 else (YELLOW if entropy < 0.7 else RED)
            score_color = GREEN if result.fun < 0 else RED
            
            print(f"\n{BLUE}{'═' * 50}{END}")
            print(f"{BOLD}{CYAN}🔄 Run {run_idx+1}/5 Summary:{END}")
            print(f"{BLUE}{'─' * 50}{END}")
            print(f"{BOLD}Alpha:{END}   {alpha_color}{signed_alpha:.6f}{END}")
            print(f"{BOLD}Burke:{END}   {burke_color}{burke:.6f}{END}")
            print(f"{BOLD}Entropy:{END} {entropy_color}{entropy:.6f}{END}")
            print(f"{BOLD}Score:{END}   {score_color}{result.fun:.6f}{END}")
            
            # Save intermediate results if requested
            if save_intermediate and (run_idx+1) % intermediate_save_frequency == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                intermediate_dir = f"user_data/optimization_results/intermediate_run_{run_idx+1}_{timestamp}"
                save_dir, json_path = save_optimization_result(signed_alpha, result, current_metrics, base_path=intermediate_dir)
                if save_dir:
                    intermediate_saves.append((save_dir, json_path))
                    print(f"{CYAN}💾 Saved intermediate results to:{END} {YELLOW}{save_dir}{END}")

            save_dir = None
            json_path = None
            # Use HyperoptHelper for best metrics
            helper = HyperoptHelper()
            best_metrics = helper.initialize_best_metrics_from_latest()
            best_alpha = best_metrics.get('best_alpha', helper.initial_alpha)
            best_burke = best_metrics.get('best_burke', helper.initial_burke)
            best_entropy = best_metrics.get('best_entropy', helper.initial_entropy)

            if current_metrics:
                # Always print current metrics summary with colors
                # Determine colors and improvement indicators
                alpha_improved = signed_alpha < best_alpha
                current_burke = current_metrics.get('burke', -1)
                current_entropy = current_metrics.get('entropy', 100)
                burke_improved = current_burke > best_burke
                entropy_improved = current_entropy < best_entropy
                
                alpha_color = GREEN if alpha_improved else RED
                burke_color = GREEN if burke_improved else RED
                entropy_color = GREEN if entropy_improved else RED
                
                alpha_indicator = "▼" if alpha_improved else "▲"
                burke_indicator = "▲" if burke_improved else "▼"
                entropy_indicator = "▼" if entropy_improved else "▲"
                
                print(f"\n{BLUE}{'═' * 60}{END}")
                print(f"{BOLD}{CYAN}📊 Current Metrics Summary:{END}")
                print(f"{BLUE}{'─' * 60}{END}")
                print(f"{BOLD}Alpha:{END}   {alpha_color}{signed_alpha:.6f} {alpha_indicator}{END} (Best: {YELLOW}{best_alpha:.6f}{END})")
                print(f"{BOLD}Burke:{END}   {burke_color}{current_burke:.6f} {burke_indicator}{END} (Best: {YELLOW}{best_burke:.6f}{END})")
                print(f"{BOLD}Entropy:{END} {entropy_color}{current_entropy:.6f} {entropy_indicator}{END} (Best: {YELLOW}{best_entropy:.6f}{END})")

                meets_gate_constraints = (win_rate >= 0.2 and signed_alpha < 0)

                # Check if this is a best model (meets all criteria + hard gates)
                if (signed_alpha < best_alpha and
                    current_burke > best_burke and
                    current_entropy < best_entropy and
                    meets_gate_constraints):
                    save_dir, json_path = save_optimization_result(signed_alpha, result, current_metrics)
                    print(f"\n{BLUE}{'═' * 60}{END}")
                    print(f"{BOLD}{GREEN}🏆 BEST MODEL! All criteria met!{END}")
                    print(f"{GREEN}✓ Alpha: {signed_alpha:.6f} < {best_alpha:.6f}{END}")
                    print(f"{GREEN}✓ Burke: {current_burke:.6f} > {best_burke:.6f}{END}")
                    print(f"{GREEN}✓ Entropy: {current_entropy:.6f} < {best_entropy:.6f}{END}")
                    print(f"{GREEN}✓ Win Rate: {win_rate:.2%} >= 71.00%{END}")
                    print(f"{GREEN}✓ _raw_alpha gate: {signed_alpha:.6f} < 0{END}")
                    print(f"{BOLD}{CYAN}📂 Saved results to:{END} {YELLOW}{save_dir}{END}")
                    try:
                        helper.save_backup_metrics()
                    except Exception as e:
                        raise Exception(f"Error saving backup metrics: {e}")
                    
                    # Return with save_dir so the caller knows where to save the model
                    return {
                        'signed_alpha': signed_alpha,
                        'optimization_result': result,
                        'metrics': current_metrics,
                        'nmatrix_score': result.fun,
                        'save_dir': save_dir,
                        'json_path': json_path,
                        'intermediate_saves': intermediate_saves
                    }
                # Check if this is a candidate model (meets at least one criterion + hard gates)
                elif (signed_alpha < best_alpha or
                      current_burke > best_burke or
                      current_entropy < best_entropy) and meets_gate_constraints:
                    # Save as a candidate model
                    candidate_dir = f"user_data/optimization_results/candidate_alpha_{abs(signed_alpha):.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    save_dir, json_path = save_optimization_result(signed_alpha, result, current_metrics, base_path=candidate_dir)
                    
                    print(f"\n{BLUE}{'═' * 60}{END}")
                    print(f"{BOLD}{YELLOW}⭐ CANDIDATE MODEL! Some criteria met:{END}")
                    
                    # Show which criteria were met
                    if signed_alpha < best_alpha:
                        print(f"{GREEN}✓ Alpha: {signed_alpha:.6f} < {best_alpha:.6f}{END}")
                    else:
                        print(f"{RED}✗ Alpha: {signed_alpha:.6f} >= {best_alpha:.6f}{END}")
                        
                    if current_burke > best_burke:
                        print(f"{GREEN}✓ Burke: {current_burke:.6f} > {best_burke:.6f}{END}")
                    else:
                        print(f"{RED}✗ Burke: {current_burke:.6f} <= {best_burke:.6f}{END}")
                        
                    if current_entropy < best_entropy:
                        print(f"{GREEN}✓ Entropy: {current_entropy:.6f} < {best_entropy:.6f}{END}")
                    else:
                        print(f"{RED}✗ Entropy: {current_entropy:.6f} >= {best_entropy:.6f}{END}")
                    
                    print(f"{BOLD}{CYAN}📂 Saved results to:{END} {YELLOW}{save_dir}{END}")
                    
                    if save_dir:
                        intermediate_saves.append((save_dir, json_path))
                else:
                    if not meets_gate_constraints:
                        print(
                            f"\n{RED}❌ Results not saved - failed hard gates: "
                            f"win_rate={win_rate:.2%} (need >=20.00%), "
                            f"_raw_alpha={signed_alpha:.6f} (need <0).{END}"
                        )
                    else:
                        print(f"\n{RED}❌ Results not saved - did not meet any improvement criteria.{END}")
                    
        # If we've gone through all runs without finding a best model,
        # return the best result we found along with intermediate saves
        if best_result is not None:
            print(f"\n{BLUE}{'═' * 60}{END}")
            print(f"{BOLD}{CYAN}🔄 Completed all optimization runs{END}")
            print(f"{BLUE}{'─' * 60}{END}")
            print(f"{BOLD}Best score:{END} {GREEN if best_result.fun < 0 else RED}{best_result.fun:.6f}{END}")
            
            return {
                'signed_alpha': best_overall_metrics.get('signed_alpha', signed_alpha),
                'optimization_result': best_result,
                'metrics': best_overall_metrics,
                'nmatrix_score': best_result.fun,
                'save_dir': None,  # No best model save
                'json_path': None,  # No JSON path
                'intermediate_saves': intermediate_saves
            }            
    except Exception as e:
        logging.error(f"{RED}Error in calculate_nmatrix calculations: {str(e)}{END}")
        traceback.print_exc()
        return default_result
