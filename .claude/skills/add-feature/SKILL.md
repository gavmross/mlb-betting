---
name: add-feature
description: >
  Safely add a new feature to the MLB feature engineering pipeline with
  full leakage protection and test coverage. Use when the user says
  "add feature", "add stat", "engineer new feature", or "add X as a feature".
---

# Add New Feature — Safe Workflow

## Before Writing Any Code

Ask the user these questions if not already answered:
1. What is the feature name? (use snake_case, e.g. `sp_velocity_l3`)
2. What raw stat does it derive from? (e.g. `velocity_avg` from the `pitchers` table)
3. What is the time window? (e.g. last 3 starts, rolling 10 games, season-to-date)
4. Is it a per-team feature or per-pitcher feature?
5. Does it apply to both home and away teams?

## Step 1 — Check for Leakage Risk

Read @docs/DATA_LEAKAGE.md and confirm: can this feature be computed
using only data available before first pitch on game day?

Ask yourself:
- Does it use the current game's outcome? (BANNED)
- Does it use stats from the day-of game? (BANNED unless pre-game announcement)
- Does it use future season stats? (BANNED)

If any answer is yes — refuse to implement and explain why to the user.

## Step 2 — Read Existing Pattern

```bash
# Read features.py to understand the existing pattern before writing anything
```
Read mlb/features.py and identify the most similar existing feature.
Follow the same structure exactly.

## Step 3 — Implement With Temporal Guard

Template for a pitcher rolling feature:
```python
def add_sp_velocity_l3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average fastball velocity over last 3 starts.

    Parameters
    ----------
    df : pd.DataFrame
        Pitcher game log sorted by (pitcher_id, date) ascending.

    Returns
    -------
    pd.DataFrame
        Input df with sp_velocity_l3 column added.
    """
    df = df.sort_values(['pitcher_id', 'date'])
    df['sp_velocity_l3'] = (
        df.groupby('pitcher_id')['velocity_avg']
        .shift(1)                               # exclude current game
        .rolling(3, min_periods=1)
        .mean()
    )
    return df
```

Template for a team rolling feature:
```python
def add_team_ops_10d(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['team', 'date'])
    df['team_ops_10d'] = (
        df.groupby('team')['ops']
        .shift(1)
        .rolling(10, min_periods=3)
        .mean()
    )
    return df
```

## Step 4 — Write Two Tests

Add to tests/unit/test_features.py:

```python
def test_{feature_name}_no_leakage(sample_games):
    """Feature must not use data from the current game."""
    df = build_features(sample_games)
    # For each row, verify value equals rolling mean of prior rows only
    # (spot-check 10 random rows)
    for _ in range(10):
        idx = random.randint(5, len(df) - 1)
        row = df.iloc[idx]
        prior = df[(df['pitcher_id'] == row['pitcher_id']) & (df.index < idx)]
        expected = prior['velocity_avg'].tail(3).mean()
        np.testing.assert_allclose(row['sp_velocity_l3'], expected, rtol=1e-5)

def test_{feature_name}_cold_start(sample_games):
    """Feature must be NaN for the first game of a pitcher's history."""
    df = build_features(sample_games)
    first_idx = df.groupby('pitcher_id').head(1).index
    # With min_periods=1 we allow non-NaN from first game — adjust if needed
    # With min_periods=3 first two games should be NaN
    # Verify whichever applies
```

## Step 5 — Run Tests

```bash
pytest tests/unit/test_features.py::test_{feature_name}_no_leakage -v
pytest tests/unit/test_features.py::test_{feature_name}_cold_start -v
```

Do NOT proceed until both pass.

## Step 6 — Register Feature

Add the new feature name to `configs/model_config.yaml` under the appropriate
feature group. Document in docs/ARCHITECTURE.md under the relevant feature group.

## Step 7 — Lint

```bash
ruff check mlb/features.py && ruff format mlb/features.py
```

## Done

Report to user: "Feature `{name}` added. Leakage test passes. Registered in
model_config.yaml. Run `/backtest` to measure its impact on model performance."
