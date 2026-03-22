# Fuzzy Group Preference Active-Learning Experiments

This repository contains a lightweight experiment harness for the section 6 strategy comparison described in the discussion.

## What is implemented

- Synthetic additive-utility data generation with monotone piecewise-linear marginal utilities.
- Group decision-maker simulation that returns fuzzy class-interval answers.
- A surrogate collective learner that maintains interval-valued utilities and class-compatibility distributions.
- Thirteen query strategies, including `random` as the 13th baseline.
- Batch execution for 100 repetitions per strategy (1300 runs total by default).
- Three SVG comparison figures based on the requested presentation rule:
  - `Accuracy`
  - `Iterations`
  - `Accuracy / number of questions`

## Run the requested section 6 experiment

```bash
python experiments/run_section6.py --output-dir outputs/section6_run --repetitions 100 --seed 20260322
python experiments/plot_section6.py
```

This produces:

- `outputs/section6_run/section6_strategy_comparison_runs.csv`
- `outputs/section6_run/section6_strategy_comparison_summary.json`
- `outputs/section6_run/plots/accuracy_boxplot.svg`
- `outputs/section6_run/plots/iterations_boxplot.svg`
- `outputs/section6_run/plots/accuracy_per_question_boxplot.svg`

## Plot convention

- Red diamond: mean
- Box body: standard error (SE)
- Longer whisker with end caps: 95% confidence interval

## Notes

The implementation uses only the Python standard library so it can run in a minimal environment without external scientific Python packages.
