import pickle
import numpy as np

# Load the Q-table
with open('user_data/optimization_results/alpha_0.0000_20260319_003500/pkls/q_table.pkl', 'rb') as f:
    q_table_bids = pickle.load(f)

# Convert to numpy array if it's a list for easier inspection
q_table_bids = np.array(q_table_bids)

# Check the shape and values
print("Bids Q-table shape:", q_table_bids.shape)
print("\nFirst 5 states' Q-values:")
print(q_table_bids[:5])