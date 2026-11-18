# Changelog

Versions follow `MAJOR.MINOR.PATCH`.

## Unreleased

## v0.1.0

First public release.

- FastAPI job server: start, stop, status and history of PyTorch training runs.
- Next.js dashboard to launch runs and compare results, with an LLM advisor for hyperparameters.
- 86 recorded MNIST runs in `training_jobs/`. Best run: 99.74% test accuracy after 60 epochs.
- Results figure drawn from the run logs (`scripts/plot_results.py`), and tests that check the README's numbers against the recorded runs.
- Fixed the indentation of the training loop in `train.py`, which did not parse.
