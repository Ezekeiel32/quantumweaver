# QuantumWeaver

[![CI](https://github.com/Ezekeiel32/quantumweaver/actions/workflows/ci.yml/badge.svg)](https://github.com/Ezekeiel32/quantumweaver/actions/workflows/ci.yml) [![Release](https://img.shields.io/github/v/release/Ezekeiel32/quantumweaver?label=release)](https://github.com/Ezekeiel32/quantumweaver/releases) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE) ![Python](https://img.shields.io/badge/Python-3-3776AB) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C) ![FastAPI](https://img.shields.io/badge/FastAPI-009688) ![Next.js](https://img.shields.io/badge/Next.js-000000)

A web platform for running and tracking PyTorch training experiments. A FastAPI backend starts training jobs, reports their progress and keeps a record of every run. A Next.js dashboard launches jobs, compares results and shows GPU and system load. An LLM assistant suggests parameters for the next run based on the results so far.

The model under study is a residual CNN with an adaptive layer after each block (called "ZPE flow" in the code). Each of these layers is a gain vector, kept between 0.8 and 1.2, that is updated on every forward pass from the batch statistics, with momentum and injected noise. The experiments test whether this kind of adaptive perturbation improves accuracy.

## Results

- 86 recorded training runs on MNIST, saved in `training_jobs/`
- Best run: **99.74%** accuracy on the MNIST test set after 60 epochs (`training_jobs/zpe_job_169cda4c.json`)

![Validation accuracy on MNIST across recorded training runs](docs/results-light.png#gh-light-mode-only)
![Validation accuracy on MNIST across recorded training runs](docs/results-dark.png#gh-dark-mode-only)

The figure is drawn from the run logs by `scripts/plot_results.py`.

## Features

- Training jobs over a REST API: start, stop, status and history (`/api/train`, `/api/stop/{job_id}`, `/api/status/{job_id}`, `/api/jobs`)
- Dashboard with accuracy and loss per epoch, the adaptive layer values of each run, and side-by-side run comparison
- GPU and system monitoring (`/api/gpu-stats`, `/api/system-stats`)
- Saved model configurations (`/api/configs`)
- LLM advisors built as Genkit flows on Gemini: parameter suggestions and run analysis. The dataset advisor falls back across Groq, OpenRouter, Together AI, Hugging Face, Mistral, Ollama and LM Studio
- Dataset download from Kaggle, Hugging Face and GitHub
- Deployment blueprint for Render (`render.yaml`)

## Stack

- **Backend:** Python, FastAPI, PyTorch, torchvision, scikit-learn, pandas
- **Frontend:** Next.js, React, TypeScript, Tailwind CSS, shadcn/ui, Genkit, p5.js

## Run locally

Backend, on port 8000:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Frontend, on port 9002 (it calls the backend at `http://localhost:8000/api`; set `NEXT_PUBLIC_API_BASE_URL` to change it):

```bash
cd src
npm install
npm run dev
```

MNIST is downloaded automatically on the first training run. The Genkit flows need a Gemini API key in `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).

## Project layout

| Path | Contents |
|---|---|
| `app.py` | FastAPI backend: model, training loop, job manager, monitoring and config endpoints |
| `train.py` | Standalone training script with the same model |
| `src/app/` | Next.js pages: dashboard, train, performance, GPU monitor, configurations, data portal, visualizations |
| `src/ai/flows/` | Genkit flows for parameter advice and run analysis |
| `training_jobs/` | One JSON record per training run (parameters, per-epoch log, final accuracy) |
| `scripts/` | Dataset download and export helpers |

## Development

```bash
pip install pytest
pytest -q
```

The tests read the recorded runs in `training_jobs/` and check that the numbers in this README match them. CI runs them on every pull request. The branch model and the pull request checklist are in [CONTRIBUTING.md](CONTRIBUTING.md), and release notes are in [CHANGELOG.md](CHANGELOG.md).

## License

MIT
