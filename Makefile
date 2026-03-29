.PHONY: train proc purge filter find-min-score-all find-min-details auto-filter best-model all

train:
	python3.11 core/bid_agent.py
proc:
	python core/metrics/process_history.py
purge:
	rm -rf user_data/optimization_results/*
	rm core/rl/pkls/*

thresholds ?= 301
base_dir ?= core/rl/pkls/ # 
best_models_txt_file ?= best_models/optimization_results_20260320_093755.txt
best_models_json_file ?= best_models/optimization_results_20260320_093755.json

filter:
	python scripts/find_low_alpha.py $(base_dir) > $(best_models_txt_file)
	#python scripts/find_low_score.py $(base_dir) $(thresholds) >> $(best_models_txt_file)
	python scripts/clean_models.py $(best_models_txt_file) $(thresholds) > $(best_models_json_file)

# Get global minimum score
find-min-score-all:
	@echo "Minimum score across all optimization results:"
	@grep -Eo '"score":[[:space:]]*[-0-9.e+]+' $(best_models_txt_file) | \
		awk -F: '{gsub(/[[:space:]]/, "", $$2); print $$2}' | \
		sort -n | \
		head -n 1

# Get global minimum score
find-min-score-all:
	@echo "Minimum score across all optimization results:"
	@grep -Eo '"score":[[:space:]]*[-0-9.e+]+' $(best_models_txt_file) | \
		awk -F: '{gsub(/[[:space:]]/, "", $$2); print $$2}' | \
		sort -n | \
		head -n 1

# Find details for the global minimum score
find-min-details:
	@if [ ! -s "$(best_models_txt_file)" ]; then \
		echo "Error: missing or empty $(best_models_txt_file). Run 'make filter' first."; \
		exit 1; \
	fi
	@echo "========================================="; \
	MIN_SCORE=$$(grep -Eo '"score":[[:space:]]*[-0-9.e+]+' "$(best_models_txt_file)" | \
		awk -F: '{gsub(/[[:space:]]/, "", $$2); print $$2}' | sort -n | head -n 1); \
	if [ -z "$$MIN_SCORE" ]; then \
		echo "Error: could not extract MIN_SCORE from $(best_models_txt_file)."; \
		exit 1; \
	fi; \
	RAW_ALPHA=$$(awk -v score="$$MIN_SCORE" 'BEGIN{RS=""; FS="\n"} $$0 ~ score {for(i=1;i<=NF;i++) if($$i ~ /_raw_alpha/) {split($$i,a,":"); gsub(/[ ,}]/, "", a[2]); print a[2]; exit}}' "$(best_models_txt_file)"); \
	FILE_PATH=$$(awk -v score="$$MIN_SCORE" 'BEGIN{RS=""; FS="\n"} $$0 ~ score {for(i=1;i<=NF;i++) if($$i ~ /^File:/) {gsub(/^File: /, "", $$i); print $$i; exit}}' "$(best_models_txt_file)"); \
	echo "Global Minimum Score: $$MIN_SCORE"; \
	echo "Raw Alpha: $$RAW_ALPHA"; \
	echo "File: $$FILE_PATH"; \
	echo "========================================="

# Auto-run filter with the global minimum as threshold
auto-filter:
	@if [ ! -s "$(best_models_txt_file)" ]; then \
		echo "Error: missing or empty $(best_models_txt_file). Run 'make filter' first."; \
		exit 1; \
	fi
	@MIN_SCORE=$$(grep -Eo '"score":[[:space:]]*[-0-9.e+]+' "$(best_models_txt_file)" | \
		awk -F: '{gsub(/[[:space:]]/, "", $$2); print $$2}' | sort -n | head -n 1); \
	echo "Using global minimum as threshold: $$MIN_SCORE"; \
	$(MAKE) filter thresholds="$$MIN_SCORE"

# Combined workflow: find min details without running filter again
best-model:
	@echo "Finding best model (global minimum)..."
	@if [ ! -s "$(best_models_txt_file)" ]; then \
		echo "Error: missing or empty $(best_models_txt_file). Run 'make filter' first."; \
		exit 1; \
	fi
	@MIN_SCORE=$$(grep -Eo '"score":[[:space:]]*[-0-9.e+]+' "$(best_models_txt_file)" | \
		awk -F: '{gsub(/[[:space:]]/, "", $$2); print $$2}' | sort -n | head -n 1); \
	if [ -z "$$MIN_SCORE" ]; then \
		echo "Error: could not extract MIN_SCORE from $(best_models_txt_file)."; \
		exit 1; \
	fi; \
	RAW_ALPHA=$$(awk -v score="$$MIN_SCORE" 'BEGIN{RS=""; FS="\n"} $$0 ~ score {for(i=1;i<=NF;i++) if($$i ~ /_raw_alpha/) {split($$i,a,":"); gsub(/[ ,}]/, "", a[2]); print a[2]; exit}}' "$(best_models_txt_file)"); \
	FILE_PATH=$$(awk -v score="$$MIN_SCORE" 'BEGIN{RS=""; FS="\n"} $$0 ~ score {for(i=1;i<=NF;i++) if($$i ~ /^File:/) {gsub(/^File: /, "", $$i); print $$i; exit}}' "$(best_models_txt_file)"); \
	echo "Best Model Found:"; \
	echo "  Score: $$MIN_SCORE"; \
	echo "  Raw Alpha: $$RAW_ALPHA"; \
	echo "  File: $$FILE_PATH"; \
	echo ""; \
	echo "To use this model, run:"; \
	echo "  python scripts/use_model.py --alpha $$RAW_ALPHA --file \"$$FILE_PATH\""
all:
	$(MAKE) filter && $(MAKE) best-model
