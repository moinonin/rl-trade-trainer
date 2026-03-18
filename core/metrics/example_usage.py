from process_history import generate_performance_report

# Generate report from your trading history
report = generate_performance_report(
    csv_path="/home/defi/Desktop/portfolio/projects/python/indicatorsApi/bid/agents/spreadsheets/training_history.csv",
    starting_balance=10000.0
)

# Access nmatrix score
nmatrix_score = report['nmatrix_score']

# Print detailed metrics
'''
print("\n=== Trading Performance Report ===")
print(f"Total Reward: {sum(report['rewards']):.4f}")
print(f"Action Distribution:")
for action, count in report['action_distribution'].items():
    print(f"  {action}: {count:.2%}")
print(f"Average Reward: {report['avg_reward']:.4f}")
print(f"Max Drawdown: {report['max_drawdown']:.4f}")
'''

