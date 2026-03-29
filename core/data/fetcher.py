import ccxt
import pandas as pd
from pandas import Series
import talib.abstract as ta
import numpy as np
import logging, fire
from defirl.rlhf_long import RLmodel_bids as bid
#from defirl.rl import RLmodel_large as large
import time
#from defirl.rl import RLmodel_small as small # uses defirl==0.5.0

from nlp.imit_main import imit_signal as imit
from nlp.nlp_main import nlp_signal as nlp # nlp-project==0.1.1
from typing import Optional

from core.exchange.exchange import init_exchange, API_KEY, API_SECRET
from contextlib import suppress
from core.optimize.parameters import CategoricalParameter
with suppress(ImportError):
    from skopt.space import Categorical

class CustomCategoricalParameter(CategoricalParameter):
    def get_space(self):
        return self.space

buy_imit_short = CustomCategoricalParameter(['do_nothing','go_long','go_short'], default= 'go_short', space='buy', optimize=False)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Fetcher:
    def __init__(self,pair: str, exchange_name: Optional[str] = 'bybit', interval: str = '1s',limit: int = 30):
        self.pair = pair
        self.exchange_name = exchange_name
        self.interval = interval
        self.limit = limit
        self.client = getattr(ccxt, exchange_name)() # init_exchange(exchange_name=exchange_name, testnet=True, api_key=API_KEY, api_secret=API_SECRET)

    def fetch_pair_df(self):
        try:
            pair_data = self.client.fetch_ohlcv(f'{self.pair.upper()}',f'{self.interval}',limit = self.limit)
            lst = []
            for i in pair_data:
                pair_ohlcv = {
                    "timestamp": i[0],
                    "open": i[1],
                    "high": i[2],
                    "low": i[3],
                    "close": i[4],
                    "volume": i[5]

                }
                lst.append(pair_ohlcv)
            return pd.DataFrame.from_dict(lst)
        except ccxt.BaseError as e:
            logging.error(f"Error fetching pair data: {e}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    def computed_indicators_df(self):
        long_k_period = 15
        long_d_period = 3
        long_j_period = 3

        short_k_period = 14
        short_d_period = 3
        short_j_period = 3

        try:
            df = pd.DataFrame.from_dict(self.fetch_pair_df())
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
            df['short_kdj'] = (df['short_d'] < df['short_k']) & (df['short_k'] < df['short_j']).astype(int)

            ''''
            df['is_short'] = (
                (df['close'] < df['sma-25']) &  
                (df['sma-07'] < df['sma-05']) &  
                (df['macd'] < df['macdsignal'])   
            ).astype(int)
            '''
            return df

        except Exception as e:
            logging.error(f"Error computing indicators: {e}")
            return pd.DataFrame(columns=['ask', 'bid', 'sma-compare', 'close', 'high', 'low', 'volume', 'open', 'is_short'])

    # Compute reinforcement learning methods
    def getBidsig(self, is_short: int, ml_candle: int = 1):
        df = self.computed_indicators_df()

        if df.empty:
            logging.info("Error: DataFrame is empty. Cannot compute bidsig.")
            return None

        if ml_candle > len(df):
            logging.info(f"Error: ml_candle ({ml_candle}) is out of range. DataFrame has only {len(df)} rows.")
            return None

        try:
            row = df.iloc[-ml_candle].squeeze()

            required_columns = ['ask', 'bid', 'sma-compare']
            if not all(col in row.index for col in required_columns):
                logging.info(f"Error: DataFrame is missing required columns: {required_columns}")
                return None

            if row[required_columns].isnull().any():
                logging.error(f"Error: Required columns contain NaN values: {required_columns}")
                return None

            model_output = bid(
                row['ask'],
                row['bid'],
                row['sma-compare'],
                is_short
            )
            return model_output

        except Exception as e:
            logging.error(f"Error computing bidsig: {e}")
            return None
    '''
    def getSmallsig(self, is_short: int, ml_candle: int = 1):
        df = self.computed_indicators_df()

        if df.empty:
            logging.info("Error: DataFrame is empty. Cannot compute smallsig.")
            return None

        if ml_candle > len(df):
            logging.info(f"Error: ml_candle ({ml_candle}) is out of range. DataFrame has only {len(df)} rows.")
            return None

        try:
            row = df.iloc[-ml_candle].squeeze()

            required_columns = ['sma-05', 'sma-07', 'sma-25', 'sma-compare']
            if not all(col in row.index for col in required_columns):
                logging.info(f"Error: DataFrame is missing required columns: {required_columns}")
                return None

            model_output = small(
                row['sma-05'],
                row['sma-07'],
                row['sma-25'],
                row['sma-compare'],
                is_short
            )
            return model_output

        except Exception as e:
            logging.info(f"Error computing smallsig: Details {e}")
            return None

    def getLargesig(self, is_short: int, ml_candle: int = 1):
        # Fetch the computed indicators DataFrame
        df = self.computed_indicators_df()

        # Check if the DataFrame is empty
        if df.empty:
            print("Error: DataFrame is empty. Cannot compute largesig.")
            return None

        # Ensure ml_candle is within the valid range
        if ml_candle > len(df):
            print(f"Error: ml_candle ({ml_candle}) is out of range. DataFrame has only {len(df)} rows.")
            return None

        try:
            # Access the specified row and squeeze it into a Series (if it's a single row)
            row = df.iloc[-ml_candle].squeeze()

            # Ensure required columns are present
            required_columns = [
                'open', 'high', 'ema-26', 'ema-12', 'low', 'mean-grad-hist', 
                'close', 'volume', 'sma-25', 'long_jcrosk', 'short_kdj', 
                'sma-compare', 'ask', 'bid'
            ]
            if not all(col in row.index for col in required_columns):
                print(f"Error: DataFrame is missing required columns: {required_columns}")
                return None

            # Compute the model output
            model_output = large(
                row['open'],
                row['high'],
                row['ema-26'],
                row['ema-12'],
                row['low'],
                row['mean-grad-hist'],
                row['close'], 
                row['volume'], 
                row['sma-25'],
                row['long_jcrosk'],
                row['short_kdj'],
                row['sma-compare'],
                row['ask'],
                row['bid'],
                is_short
            )
            return model_output

        except Exception as e:
            print(f"Error computing largesig: {e}")
            return None
    '''
    # Compute nlp methods
    def compute_ml_input_signal(self):
        try:
            row = self.computed_indicators_df().tail(self.limit)
            cols = ['open','high','ema-26','ema-12','low','mean-grad-hist','close','volume','sma-25','long_jcrosk','short_kdj']
            return row[cols]
        except AttributeError as e:
            return {'error': e}

    def compute_imit_action_df(self, ml_candle: int = 10):
        try:
            df = self.compute_ml_input_signal()
            df['short_kdj'] = df['short_kdj'].astype(int)

            if not df.empty:
                row = df.iloc[-ml_candle].squeeze()
                args_short = (
                                row['open'], row['high'], row['ema-26'], row['ema-12'],
                                row['low'], row['mean-grad-hist'], row['close'], row['volume'], 
                                row['sma-25'], row['long_jcrosk'], row['short_kdj']
                            
                            )
                for i in buy_imit_short.opt_range:
                    df[f'buy_imit_short_{i}'] = imit(*args_short)
                return df
            else:
                return {'error': 'No data found for the given pair'}
        except Exception as e:
            return {'error': e}

    def compute_nlp_action(self):
        df = self.compute_imit_action_df()
        if df.empty:
            logging.error("DataFrame is empty. Cannot compute NLP action.")
            return {'error': 'No data found for the given pair'}

        column_name = f'buy_imit_short_{buy_imit_short.value}'

        if column_name not in df.columns:
            logging.error(f"Column '{column_name}' not found in DataFrame.")
            return {'error': f"Column '{column_name}' not found"}

        try:
            ser = df[column_name]

            if len(ser) < 3:
                logging.error(f"Column '{column_name}' has fewer than 3 values.")
                return {'error': f"Column '{column_name}' has fewer than 3 values"}

            vals = ser.iloc[-3:].values
            nlp_result = nlp(vals)
            return nlp_result

        except Exception as e:
            logging.error(f"Error computing NLP action: {e}", exc_info=True)
            return {'error': str(e)}

def main(pair,exchange_name,method: str, *args,**kwargs):
    try:
        fetcher = Fetcher(pair,exchange_name)
        if method == 'bidrl':
            return fetcher.getBidsig(*args, **kwargs)
        elif method == 'largerl':
            return fetcher.getLargesig(*args, **kwargs)
        elif method == 'smallrl':
            return fetcher.getSmallsig(*args, **kwargs)
        elif method == 'nlp':
            return fetcher.compute_nlp_action()
        else:
            raise ValueError("Invalid method specified. Usage only support 'bidrl', 'largerl' or 'smallrl'.")
    except Exception as e:
        print(e)



#print(Fetcher('ADA/USDT:USDT', 'bybit', '1m', limit=30).compute_nlp_action())

'''
if __name__ == '__main__':
    fire.Fire(main)
'''
