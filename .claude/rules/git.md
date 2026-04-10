# Git Workflow Rules
**Active when:** running git commands, creating commits, or opening PRs

---

## Branch Strategy

Trunk-based development. Always branch from main:

```bash
git checkout main && git pull
git checkout -b feature/add-wind-direction-feature
# work...
git checkout -b fix/sbr-team-name-mapping
git checkout -b data/backfill-weather-2021-2022
git checkout -b test/leakage-tests-pitcher-features
git checkout -b docs/update-architecture-schema
```

Prefix rules:
- `feature/` — new functionality
- `fix/`     — bug fix
- `data/`    — data pipeline or schema change
- `test/`    — tests only, no logic change
- `docs/`    — documentation only

## Commit Messages — Conventional Commits

Format: `<type>(<scope>): <description>`

```bash
# Good examples
feat(features): add wind direction encoding for dome stadiums
fix(scraper): handle missing probable pitcher in MLB Stats API response
data(schema): add dispersion_alpha column to predictions table
test(features): add cold-start leakage test for bullpen_era_7d
docs(architecture): update model section to reflect Poisson GLM

# Bad examples
git commit -m "updates"
git commit -m "fix bug"
git commit -m "WIP"
```

Types: `feat`, `fix`, `data`, `test`, `docs`, `refactor`, `chore`

## What to Never Commit

The following are in .gitignore and must never be committed:
- `data/mlb.db` — database file
- `data/raw/`   — cached raw API responses
- `data/models/` — serialized model files
- `.env`         — environment variables (API keys)
- `__pycache__/`, `*.pyc`, `.pytest_cache/`
- `notebooks/.ipynb_checkpoints/`

If accidentally staged: `git rm --cached data/mlb.db`

## Pre-Commit Checklist

Before every commit, verify:
```bash
ruff check . && ruff format --check .   # no lint errors
pytest tests/unit/ -v -q                # unit tests pass
```

Do NOT commit if either command fails.

## Pull Request Rules

- PR title must follow Conventional Commits format
- Every PR touching mlb/features.py must include a leakage test
- Every PR touching mlb/betting.py or mlb/calibration.py must include a math test
- Squash merge to keep main history clean
