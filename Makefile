train:
	python3.11 core/bid_agent.py
proc:
	python core/metrics/process_history.py
purge:
	rm -rf user_data/optimization_results/*
	rm core/rl/pkls/*

thresholds ?= 0.03
base_dir ?= user_data/optimization_results/ # core/rl/pkls/ # 
best_models_txt_file ?= best_models/optimization_results_20260320_093755.txt
best_models_json_file ?= best_models/optimization_results_20260320_093755.json

filter:
	python scripts/find_low_alpha.py $(base_dir) > $(best_models_txt_file)
	python scripts/clean_models.py $(best_models_txt_file) > $(best_models_json_file)

# Get global minimum score
find-min-score-all:
	@echo "Minimum score across all optimization results:"
	@grep -Eo '"score":[[:space:]]*[-0-9.e+]+' $(best_models_txt_file) | \
		awk -F: '{gsub(/[[:space:]]/, "", $$2); print $$2}' | \
		sort -n | \
		head -n 1

# Find details for the global minimum score
find-min-details:
	$(eval MIN_SCORE := $(shell grep -Eo '"score":[[:space:]]*[-0-9.e+]+' $(best_models_txt_file) | awk -F: '{gsub(/[[:space:]]/, "", $$2); print $$2}' | sort -n | head -n 1))
	$(eval RAW_ALPHA := $(shell awk -v score="$(MIN_SCORE)" 'BEGIN{RS=""; FS="\n"} $$0 ~ score {for(i=1;i<=NF;i++) if($$i ~ /_raw_alpha/) {split($$i,a,":"); gsub(/[ ,}]/, "", a[2]); print a[2]; exit}}' $(best_models_txt_file)))
	$(eval FILE_PATH := $(shell awk -v score="$(MIN_SCORE)" 'BEGIN{RS=""; FS="\n"} $$0 ~ score {for(i=1;i<=NF;i++) if($$i ~ /^File:/) {gsub(/^File: /, "", $$i); print $$i; exit}}' $(best_models_txt_file)))
	
	@echo "========================================="
	@echo "Global Minimum Score: $(MIN_SCORE)"
	@echo "Raw Alpha: $(RAW_ALPHA)"
	@echo "File: $(FILE_PATH)"
	@echo "========================================="

# Auto-run filter with the global minimum as threshold
auto-filter:
	$(eval MIN_SCORE := $(shell grep -Eo '"score":[[:space:]]*[-0-9.e+]+' $(best_models_txt_file) | awk -F: '{gsub(/[[:space:]]/, "", $$2); print $$2}' | sort -n | head -n 1))
	@echo "Using global minimum as threshold: $(MIN_SCORE)"
	$(MAKE) filter thresholds=$(MIN_SCORE)

# Combined workflow: find min details without running filter again
best-model:
	@echo "Finding best model (global minimum)..."
	$(eval MIN_SCORE := $(shell grep -Eo '"score":[[:space:]]*[-0-9.e+]+' $(best_models_txt_file) | awk -F: '{gsub(/[[:space:]]/, "", $$2); print $$2}' | sort -n | head -n 1))
	$(eval RAW_ALPHA := $(shell awk -v score="$(MIN_SCORE)" 'BEGIN{RS=""; FS="\n"} $$0 ~ score {for(i=1;i<=NF;i++) if($$i ~ /_raw_alpha/) {split($$i,a,":"); gsub(/[ ,}]/, "", a[2]); print a[2]; exit}}' $(best_models_txt_file)))
	$(eval FILE_PATH := $(shell awk -v score="$(MIN_SCORE)" 'BEGIN{RS=""; FS="\n"} $$0 ~ score {for(i=1;i<=NF;i++) if($$i ~ /^File:/) {gsub(/^File: /, "", $$i); print $$i; exit}}' $(best_models_txt_file)))
	
	@echo "Best Model Found:"
	@echo "  Score: $(MIN_SCORE)"
	@echo "  Raw Alpha: $(RAW_ALPHA)"
	@echo "  File: $(FILE_PATH)"
	@echo ""
	@echo "To use this model, run:"
	@echo "  python scripts/use_model.py --alpha $(RAW_ALPHA) --file \"$(FILE_PATH)\""
all:
	make filter && make best-model
