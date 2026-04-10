# Data Leakage Rules

This document defines what constitutes data leakage in this project,
provides worked examples, and explains how to detect and prevent it.
Claude Code must read this file before implementing any feature in mlb/features.py.

---

## What Is Data Leakage?

Data leakage occurs when information from the future is used to predict
the past. In a sports betting context, leakage means using data that would
not have been available before first pitch to make a pre-game prediction.

**The consequence:** a model trained with leakage will appear to perform
well in backtesting but fail completely in live use — because the
"features" it relied on don't exist yet when you need to make a decision.

---

## The Golden Rule

> A feature value for game N must be computable using ONLY data
> from games 0 through N-1. Never game N itself.

---

## The Mandatory Pattern

Every rolling or expanding window feature must use `.shift(1)` to
exclude the current game before applying any aggregation:

```python
# CORRECT — shift(1) excludes game N before the rolling window
df = df.sort_values(['team', 'date'])
df['team_ops_10d'] = (
    df.groupby('team')['ops']
    .shift(1)               # game N is now NaN; games 0..N-1 shift up
    .rolling(10, min_periods=3)
    .mean()
)

# WRONG — .rolling() sees game N's ops value
df['team_ops_10d'] = df.groupby('team')['ops'].rolling(10).mean()

# ALSO WRONG — shifting the result does not help
df['team_ops_10d'] = (
    df.groupby('team')['ops']
    .rolling(10)
    .mean()
    .shift(1)
)
```

---

## Category 1: Rolling Window Features

**Rule:** `.shift(1)` must come BEFORE `.rolling()` in all cases.

### Pitcher Rolling Stats
```python
# CORRECT
df = df.sort_values(['pitcher_id', 'date'])
df['sp_era_l3'] = (
    df.groupby('pitcher_id')['era_game']
    .shift(1)
    .rolling(3, min_periods=1)
    .mean()
)

# CORRECT — season-to-date using expanding window
df['sp_siera_season'] = (
    df.groupby(['pitcher_id', 'season'])['siera_game']
    .shift(1)
    .expanding(min_periods=1)
    .mean()
)
```

### Team Rolling Stats
```python
# CORRECT
df = df.sort_values(['team', 'date'])
df['bullpen_era_7d'] = (
    df.groupby('team')['bullpen_era_game']
    .shift(1)
    .rolling(7, min_periods=2)
    .mean()
)
```

### Market Features (SBR Total Line)
The opening line is known before the game and is safe as a feature.
The closing line is the outcome of the betting market — use it only
as the benchmark in betting simulation, never as a model feature.

```python
# CORRECT — opening line is pre-game information
df['total_line_feature'] = df['total_open']

# WRONG — closing line incorporates sharp money movement throughout the day
df['total_line_feature'] = df['total_close']   # BANNED as feature

# Line movement as a feature also requires shift to avoid look-ahead
df['line_movement'] = (df['total_close'] - df['total_open']).shift(1)
# This is only valid as a historical market signal for prior games,
# not for the current game being predicted
```

---

## Category 2: Season-to-Date Stats

Season-to-date stats for game N must exclude game N.
Use `.shift(1).expanding()`, never `.expanding()` alone.

```python
# CORRECT
df['sp_k9_season'] = (
    df.groupby(['pitcher_id', 'season'])['k9_game']
    .shift(1)
    .expanding(min_periods=3)
    .mean()
)

# WRONG — includes today's game
df['sp_k9_season'] = (
    df.groupby(['pitcher_id', 'season'])['k9_game']
    .expanding()
    .mean()
)
```

---

## Category 3: Pre-Game vs Post-Game Data

### Pre-game data — ALLOWED as features
- Probable starter name and historical stats (through game N-1)
- Confirmed lineup (announced ~3 hours before game)
- Weather forecast at prediction time (Open-Meteo forecast endpoint)
- Opening betting line — `total_open` from SBR
- Elo rating entering the game (computed from prior games only)
- Park factors (static per season — not game-specific)

### Post-game data — BANNED as features
- Actual runs scored in game N (`home_score`, `away_score`, `total_runs`)
- Starter's actual innings pitched in game N
- Actual bullpen usage in game N
- Statcast data generated during game N
- Closing line for game N (`total_close`) — available only after betting closes
- Any game result fields

### Same-day edge cases
A pitcher's season stats entering game N are safe — but only when
computed through game N-1 using `.shift(1)`. Stats FROM game N are
not available until after game N completes.

---

## Category 4: Train/Test Split Leakage

Walk-forward CV is the only valid strategy.

```python
# CORRECT
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, gap=162)
# gap=162 provides at least one full season buffer between train and test

# WRONG — random split leaks future games into training
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# ALSO WRONG
from sklearn.model_selection import StratifiedKFold, KFold
```

---

## Category 5: NaN Imputation Leakage

Never fit an imputer on combined train+test data.

```python
# CORRECT — fit only on training data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)       # transform only, no fit

# WRONG — fitting on all data leaks test statistics into training
imputer.fit(X_all)
```

For rolling features, NaN at the start of a team/pitcher's history
is expected and correct behaviour. Use `min_periods` to allow partial
windows rather than filling early NaNs with global statistics.

---

## Category 6: Feature Engineering Order of Operations

Always process in this exact order:

```
1. Sort by (entity_id, date) ascending
2. Apply .shift(1)
3. Apply .rolling() or .expanding()
4. Join to game-level frame on (entity_id, date)
5. Sort final frame by date for walk-forward CV
```

Never sort the dataframe after computing rolling features — sorting
changes which rows the rolling window sees and silently corrupts the output.

---

## Leakage Detection Checklist

Before adding any feature to `mlb/features.py`, verify each item:

- [ ] Can this value be known before first pitch on game day?
- [ ] Does the rolling/expanding window use `.shift(1)` before the aggregation?
- [ ] Is the dataframe sorted by `(entity_id, date)` before the rolling op?
- [ ] Are season-to-date stats computed with `.shift(1).expanding()`?
- [ ] Is `total_close` used only in betting simulation, never as a model feature?
- [ ] Is there a leakage test in `tests/unit/test_features.py` for this feature?

If any answer is "no" or "unsure" — do not add the feature yet.

---

## Leakage Test Template

Every feature in `mlb/features.py` must have a corresponding test:

```python
import numpy as np
import pandas as pd
import pytest
from mlb.features import build_features


def test_sp_era_l3_no_leakage(sample_games: pd.DataFrame) -> None:
    """sp_era_l3 at row N must not use any data from game N."""
    df = build_features(sample_games.copy())
    df = df.sort_values(['pitcher_id', 'date']).reset_index(drop=True)

    for pitcher_id, group in df.groupby('pitcher_id'):
        group = group.reset_index(drop=True)
        for i in range(1, min(len(group), 10)):   # spot-check first 10 rows
            prior_mean = group.iloc[:i]['era_game'].tail(3).mean()
            actual = group.iloc[i]['sp_era_l3']
            if not np.isnan(actual) and not np.isnan(prior_mean):
                np.testing.assert_allclose(
                    actual, prior_mean, rtol=1e-5,
                    err_msg=(
                        f"Leakage: sp_era_l3 for pitcher {pitcher_id} "
                        f"at row {i} uses game-N data"
                    )
                )


def test_sp_era_l3_cold_start(sample_games: pd.DataFrame) -> None:
    """sp_era_l3 must be NaN for a pitcher's first game (no prior data)."""
    df = build_features(sample_games.copy())
    first_games = (
        df.sort_values('date')
        .groupby('pitcher_id')
        .head(1)
    )
    assert first_games['sp_era_l3'].isna().all(), (
        "sp_era_l3 should be NaN for a pitcher's first career game"
    )
```
