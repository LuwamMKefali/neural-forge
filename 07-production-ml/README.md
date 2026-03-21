# 07 · Production ML

Engineering rigour — code that a senior engineer could review.

## Contents
- `docker/` — Containerised project templates
- `ci-cd/` — GitHub Actions workflows for ML (test, lint, eval on push)
- `experiment-tracking/` — W&B and MLflow setup templates
- `system-design/` — Notes on ML system design patterns

## Standards Applied
- Type hints + docstrings on every function
- pytest coverage for all data pipelines and model outputs
- Conventional commits: `feat:` / `fix:` / `exp:` / `docs:`
- Reproducibility: seeds, `requirements.txt`, Docker for every project
