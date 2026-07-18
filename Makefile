dataset:      python scripts/build_dataset.py --config configs/default.yaml
train:        python scripts/train.py --config configs/default.yaml
diagnose:     python scripts/diagnose.py --config configs/default.yaml
robustness:   python scripts/run_robustness_suite.py --config configs/default.yaml
benchmark:    python scripts/benchmark_latency.py
determinism:  python scripts/check_determinism.py
smoke:        python scripts/submission_smoke_test.py
test:         pytest tests/
ci:           pytest tests/unit tests/causality tests/determinism
run_all:      python scripts/run_all.py

.PHONY: dataset train diagnose robustness benchmark determinism smoke test ci run_all
