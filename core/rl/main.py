from .agent import BidsAgent
from .trainer import BidsTrainer

agent = BidsAgent()
trainer = BidsTrainer(agent)

def train_model(states, prices, is_short=0):
    """
    Train the model on new data
    
    Args:
        states: List of tuples (ask, bid, sma_compare, is_short)
        prices: List of corresponding prices
        is_short: Integer flag for short position
    """
    return trainer.train_episode(states, prices, is_short=is_short)
