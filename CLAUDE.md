## Rules for Claude Code
- Time series are split chronologically. Never shuffle before splitting.
- The test period is scored once, at the end. Tuning uses validation only.
- Every model result is reported next to the naive baselines.
- Features use only data at time t or earlier; targets use only data after t.
- Training and any live forecaster import the same function from pipeline/features.py.
- Results go to results/tables and results/figures with the step number in the file name.
