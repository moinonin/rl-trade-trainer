import numpy as np
import pandas as pd

# Example DataFrame: Rows are metrics, columns are algorithm versions
data = {
    'defirl_0.5.0': [100, -20, -0.5, 0.02, 2.5],  # Example values for each metric
    'Version_B': [120, -25, -0.6, 0.01, 3.0],
    'Version_C': [90, -15, -0.4, 0.03, 2.0]
}

metrics = ['profit', 'drawdown', 'burke', 'entropy', 'sharpe']
df = pd.DataFrame(data, index=metrics)

# Define weights (importance of each metric)
weights = {
    'profit': 0.3,   # Higher is better
    'drawdown': 0.2,  # Lower is better
    'burke': 0.1,     # Lower (negative) is better
    'entropy': 0.2,   # Lower is better (closer to zero)
    'sharpe': 0.2     # Higher is better
}

# Normalize the metrics (Min-Max scaling)
def normalize(series, maximize=True):
    """Normalize values between 0 and 1. If maximize=True, higher values are better."""
    min_val, max_val = series.min(), series.max()
    norm = (series - min_val) / (max_val - min_val) if max_val != min_val else np.ones_like(series)
    return norm if maximize else 1 - norm  # Invert for metrics where lower is better

# Apply normalization
df.loc['profit'] = normalize(df.loc['profit'], maximize=True)
df.loc['drawdown'] = normalize(df.loc['drawdown'], maximize=False)
df.loc['burke'] = normalize(df.loc['burke'], maximize=False)
df.loc['entropy'] = normalize(df.loc['entropy'], maximize=False)
df.loc['sharpe'] = normalize(df.loc['sharpe'], maximize=True)

# Compute weighted scores
weighted_scores = df.mul(pd.Series(weights), axis=0).sum()
best_version = weighted_scores.idxmax()

print("Weighted Scores:\n", weighted_scores)
print(f"Best Version: {best_version}")
