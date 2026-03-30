import os, sys
import pickle
from nlp.imit_main import imit_signal as imit
from nlp.nlp_main import nlp_signal as nlp # nlp-project==0.0.9
#from defirl.rlhf_rdql import RDQLmodel_bids as bid
from defirl.rlhf_rdql import RDQLmodel_bids as bid
# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Only go up one level since we're already in 'bid'
sys.path.append(project_root)
import pandas as pd
import talib.abstract as ta
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass
from scipy.spatial import KDTree
#from datahandler.fetcher import Fetcher
#from datahandler.fetcher import main as sig_gen


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).flatten()
    if x.size == 0:
        return x
    max_x = np.max(x)
    ex = np.exp(x - max_x)
    denom = np.sum(ex)
    return ex / denom if denom != 0 else np.full_like(ex, 1.0 / len(ex))


@dataclass
class RDQLmodel_bids:
    """
    Double Q-learning inference model for bid signals.
    Loads q_table_a/q_table_b from local core/rl/pkls and combines them at inference time.
    """
    ask: float
    bid: float
    sma_compare: int
    is_short: int

    model_dir: str = f'{Path(__file__).resolve().parents[1]}/rl/pkls'

    q_table_a: np.ndarray = None
    q_table_b: np.ndarray = None
    q_table: np.ndarray = None
    state_to_index_dict: dict = None
    action_mapping: dict = None
    kdtree: KDTree = None
    kdtree_index_map: dict = None

    def __post_init__(self):
        self._load_data()
        self._build_kdtree()

    def _load_pickle(self, path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_data(self):
        model_path = Path(self.model_dir)
        q_table_a_path = model_path / "q_table_a.pkl"
        q_table_b_path = model_path / "q_table_b.pkl"
        q_table_path = model_path / "q_table.pkl"
        state_index_path = model_path / "state_to_index.pkl"

        self.action_mapping = {"go_long": 0, "go_short": 1, "do_nothing": 2}
        self.state_to_index_dict = {}
        self.q_table = np.array([])

        if q_table_a_path.exists() and q_table_b_path.exists():
            self.q_table_a = np.asarray(self._load_pickle(q_table_a_path))
            self.q_table_b = np.asarray(self._load_pickle(q_table_b_path))
            self.q_table = self.q_table_a + self.q_table_b
        elif q_table_path.exists():
            # Backward compatibility: single-table model.
            self.q_table = np.asarray(self._load_pickle(q_table_path))
            self.q_table_a = self.q_table / 2.0
            self.q_table_b = self.q_table / 2.0
        else:
            logging.error("No Q-table files found in %s", model_path)

        if state_index_path.exists():
            loaded_state_index = self._load_pickle(state_index_path)
            if isinstance(loaded_state_index, dict):
                self.state_to_index_dict = loaded_state_index
            elif isinstance(loaded_state_index, list):
                self.state_to_index_dict = dict(loaded_state_index)
        else:
            logging.error("State index file not found at %s", state_index_path)

    def _to_numeric(self, val):
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            v = val.lower()
            if v == "neutral":
                return 0.0
            if "up" in v:
                return 1.0
            if "down" in v:
                return -1.0
            try:
                return float(val)
            except Exception:
                return float(hash(val) % 1000)
        try:
            return float(val)
        except Exception:
            return 0.0

    def _canonicalize_state(self, state_tuple):
        normalized = []
        for value in tuple(state_tuple):
            if isinstance(value, (np.floating, float)):
                normalized.append(round(float(value), 4))
            elif isinstance(value, (np.integer, int, np.bool_, bool)):
                normalized.append(int(value))
            else:
                normalized.append(value)
        return tuple(normalized)

    def _build_kdtree(self):
        self.kdtree = None
        self.kdtree_index_map = {}

        if not self.state_to_index_dict:
            return

        raw_states = []
        for state_tuple, state_index in self.state_to_index_dict.items():
            try:
                numeric = [self._to_numeric(x) for x in tuple(state_tuple)]
                if np.all(np.isfinite(numeric)):
                    raw_states.append((numeric, state_index))
            except Exception:
                continue

        if not raw_states:
            return

        max_dim = max(len(state) for state, _ in raw_states)
        normalized_states = []
        for numeric_state, state_index in raw_states:
            if len(numeric_state) < max_dim:
                numeric_state = numeric_state + [0.0] * (max_dim - len(numeric_state))
            kd_idx = len(normalized_states)
            normalized_states.append(numeric_state)
            self.kdtree_index_map[kd_idx] = state_index

        try:
            self.kdtree = KDTree(np.array(normalized_states))
        except Exception as e:
            logging.error("Error building KDTree: %s", e)
            self.kdtree = None

    def prep_state(self):
        state = np.array([[self.ask, self.bid, self.sma_compare, self.is_short]])
        if not np.all(np.isfinite(state)):
            state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        return state

    def _nearest_state_index(self, state_tuple):
        if self.kdtree is None:
            return -1
        query = np.array([self._to_numeric(x) for x in state_tuple], dtype=float)
        if query.shape[0] != self.kdtree.m:
            if query.shape[0] > self.kdtree.m:
                query = query[:self.kdtree.m]
            else:
                query = np.pad(query, (0, self.kdtree.m - query.shape[0]))
        if not np.all(np.isfinite(query)):
            return -1
        _, kd_idx = self.kdtree.query(query)
        return self.kdtree_index_map.get(int(kd_idx), -1)

    def predict_action(self):
        if self.q_table.size == 0 or not self.state_to_index_dict:
            return {
                "raw_state": None,
                "state_tuple": None,
                "best_action_index": -1,
                "action": "error_model_not_loaded",
                "confidence": {},
            }

        state = self.prep_state()
        state_tuple = self._canonicalize_state(tuple(state.flatten()))
        state_index = self.state_to_index_dict.get(state_tuple, -1)
        if state_index == -1:
            state_index = self._nearest_state_index(state_tuple)

        if state_index == -1:
            return {
                "raw_state": state,
                "state_tuple": state_tuple,
                "best_action_index": -1,
                "action": "error_no_state_match",
                "confidence": {},
            }

        try:
            q_values = np.asarray(self.q_table[state_index]).flatten()
        except Exception:
            return {
                "raw_state": state,
                "state_tuple": state_tuple,
                "best_action_index": -1,
                "action": "error_qtable_index",
                "confidence": {},
            }

        if len(q_values) != len(self.action_mapping):
            if len(q_values) > len(self.action_mapping):
                q_values = q_values[:len(self.action_mapping)]
            else:
                q_values = np.pad(q_values, (0, len(self.action_mapping) - len(q_values)), constant_values=-np.inf)

        confidence = _softmax_1d(q_values)
        best_action_index = int(np.argmax(q_values))
        action = next((name for name, idx in self.action_mapping.items() if idx == best_action_index), "do_nothing")

        return {
            "raw_state": state,
            "state_tuple": state_tuple,
            "best_action_index": best_action_index,
            "action": action,
            "confidence": {name: float(confidence[idx]) for name, idx in self.action_mapping.items()},
        }

def compute_nlp_indicators(dataframe: pd.DataFrame = None):
    long_k_period = 15
    long_d_period = 3


    short_k_period = 14
    short_d_period = 3

    try:
        df = dataframe
        if df.empty:
            raise ValueError("Fetched data is empty.")

        # Compute indicators
        df['sma-05'] = ta.SMA(df['close'], timeperiod=5)
        df['sma-07'] = ta.SMA(df['close'], timeperiod=7)
        df['sma-25'] = ta.SMA(df['close'], timeperiod=25)
        df['sma-compare'] = ((df['sma-07'] > df['sma-05']) & (df['sma-25'] > df['sma-07'])).astype(int)
        df['ask'] = df['close'] * df['volume'] / (df['close'] + df['open'])
        df['bid'] = df['close'] * df['volume'] / (df['close'] + df['open'])
        df['ema-26'] = ta.EMA(df['close'], timeperiod=12)
        df['ema-12'] = ta.EMA(df['close'], timeperiod=26)
        df['macd'] = df['ema-12'] - df['ema-26']
        df['macdsignal'] = ta.EMA(df['macd'], timeperiod=9)
        df['macd-histogram'] = df['macd'] - df['macdsignal']
        df['grad-histogram'] = np.gradient(df['macd-histogram'].rolling(center=False, window=2).mean())
        df['mean-grad-hist'] = (df['grad-histogram'] > df['grad-histogram'].mean()).astype(int)
        df['long_k'] = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=long_k_period, slowk_period=long_d_period, slowd_period=long_d_period)[0]
        df['long_d'] = ta.SMA(df['long_k'], timeperiod=long_d_period)
        df['long_j'] = 3 * df['long_d'] - 2 * df['long_k']
        df['long_jcrosk'] = ((df['long_j'].shift(1) < df['long_k'].shift(1)) & (df['long_j'] > df['long_k'])).astype(int)
        df['short_k'] = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=short_k_period, slowk_period=short_d_period, slowd_period=short_d_period)[0]
        df['short_d'] = ta.SMA(df['short_k'], timeperiod=short_d_period)
        df['short_j'] = 3 * df['short_d'] - 2 * df['short_k']
        df['short_kdj_ratio'] = df['short_d'] / df['short_j']
        df['short_kdj'] = ((df['short_d'] < df['short_k']) & (df['short_k'] < df['short_j'])).astype(int)

        return df

    except Exception as e:
        logging.error(f"Error computing indicators: {e}")
        return pd.DataFrame(columns=['ask', 'bid', 'sma-compare', 'close', 'high', 'low', 'volume', 'open', 'is_short'])


#------------------------------------------------------------------------------------------------------------
""""
def getBidsig(is_short: int, ml_candle: int, dataframe: pd.DataFrame = df):
    if len(dataframe) <= ml_candle:
        logging.warning(f"Not enough data (len={len(dataframe)}) for ml_candle={ml_candle}")
        return 'do_nothing'
    
    try:
        row_data = dataframe.iloc[-ml_candle]
        model_output = bid(
            row_data['ask'],
            row_data['bid'],
            row_data['sma-compare'],
            is_short
        ).predict_action().get("action") # get('trans_action')
        return model_output
    except Exception as e:
        logging.error(f"Error in getBidsig: {str(e)}")
        return 'do_nothing'
#------------------------------------------------------------------------------------------------------------
"""
def getBidsig(is_short: int, ml_candle: int, dataframe: pd.DataFrame = None):
    if len(dataframe) <= ml_candle:
        logging.warning(f"Not enough data (len={len(dataframe)}) for ml_candle={ml_candle}")
        return None
    
    try:
        row_data = dataframe.iloc[-ml_candle]
        model_output = bid(
            row_data['ask'],
            row_data['bid'],
            row_data['sma-compare'],
            is_short
        )
        return model_output
    except Exception as e:
        logging.error(f"Error in getBidsig: {str(e)}")
        return None

def getNlpsig(ml_candle: int, dataframe: pd.DataFrame = None):
    if len(dataframe) <= ml_candle:
        logging.warning(f"Not enough data (len={len(dataframe)}) for ml_candle={ml_candle}")
        return None
    
    try:
        df = compute_nlp_indicators(dataframe)
        row_data = df.iloc[-ml_candle]
        elems = [
            row_data['open'],
            row_data['high'],
            row_data['ema-26'],
            row_data['ema-12'],
            row_data['low'],
            row_data['mean-grad-hist'],
            row_data['close'], 
            row_data['volume'], 
            row_data['sma-25'],
            row_data['long_jcrosk'],
            row_data['short_kdj']
        ]

        df[f'buy_imit_short'] = imit(*elems)
        model_output = nlp(df['buy_imit_short'].iloc[-3:].values)
        return model_output

    except Exception as e:
        logging.error(f"Error in getBidsig: {str(e)}")
        return None


"""
Example usage
print(Fetcher('ADA/USDT:USDT', 'bybit').compute_nlp_action())
print(sig_gen('ADA/USDT:USDT', 'bybit', method='nlp'))
print(sig_gen('ADA/USDT:USDT', 'bybit', method='bidrl', is_short=0).predict_action())
print(sig_gen('ADA/USDT:USDT', 'bybit', method='largerl', is_short=0).predict_action().get("action"))"
print(sig_gen('ADA/USDT:USDT', 'bybit', method='smallrl', is_short=0, ml_candle=10).predict_action().get("action"))"
"""

if __name__ == '__main__':
    compute_nlp_indicators()
