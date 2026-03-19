train:
	python3.11 core/bid_agent.py
proc:
	python core/metrics/process_history.py
purge:
	rm -rf user_data/optimization_results/*
	rm core/rl/pkls/*
