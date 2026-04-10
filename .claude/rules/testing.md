# Testing Rules
**Active when:** editing files in tests/ or writing new tests

---

## Framework & Commands

```bash
pytest tests/ -v                              # full suite
pytest tests/unit/test_features.py -v        # after any feature change
pytest tests/unit/ -v                         # unit tests only
pytest tests/integration/ -v                  # integration tests only
pytest tests/ -v --tb=short -q               # concise output
pytest tests/ -v -k "test_leakage"           # run tests matching name
```

## Fixtures — Scope Rules

```python
import pytest
import sqlite3

# Session-scoped: expensive read-only resources
@pytest.fixture(scope="session")
def db_conn():
    """Read-only DB connection shared across the entire test session."""
    conn = sqlite3.connect("data/mlb.db")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()

@pytest.fixture(scope="session")
def sample_games(db_conn):
    """Load a fixed set of historical games once for the session."""
    return db_conn.execute(
        "SELECT * FROM games WHERE season = 2023 LIMIT 500"
    ).fetchall()

# Function-scoped: anything that modifies state
@pytest.fixture(scope="function")
def test_db(tmp_path):
    """Isolated in-memory DB for tests that write data."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    # apply schema
    from mlb.db import create_schema
    create_schema(conn)
    yield conn
    conn.close()
```

## Required Tests for New Features

Every new feature added to mlb/features.py requires two tests in
tests/unit/test_features.py:

**1. Leakage test** — verifies the feature excludes today's game:
```python
def test_sp_siera_l5_no_leakage(sample_games):
    """sp_siera_l5 must not use data from the current game."""
    df = build_features(sample_games)
    # Feature at row N must equal rolling mean of rows 0..N-1 only
    for i in range(1, len(df)):
        game_n = df.iloc[i]
        prior = df.iloc[:i]
        expected = prior[prior['pitcher_id'] == game_n['pitcher_id']]['siera_game'].tail(5).mean()
        np.testing.assert_allclose(game_n['sp_siera_l5'], expected, rtol=1e-5)
```

**2. Cold-start test** — verifies NaN at game 0 of a team's season:
```python
def test_sp_siera_l5_cold_start(sample_games):
    """Feature must be NaN for a pitcher's first game (no prior data)."""
    df = build_features(sample_games)
    first_game_idx = df.groupby('pitcher_id').head(1).index
    assert df.loc[first_game_idx, 'sp_siera_l5'].isna().all()
```

## Required Tests for Math Functions

Every new function in mlb/betting.py, mlb/calibration.py, mlb/model.py
requires at least one correctness test in tests/unit/test_betting.py or
tests/unit/test_calibration.py:

```python
def test_kelly_bet_known_values():
    """Verify Kelly formula against manual calculation."""
    # win_prob=0.6, price=0.50 → b=1.0, f* = (0.6*1 - 0.4)/1 = 0.20
    # fractional at 0.25x = 0.05, capped at max_pct=0.05
    result = kelly_bet(win_prob=0.6, kalshi_price=0.50)
    np.testing.assert_allclose(result, 0.05, rtol=1e-6)

def test_kelly_bet_no_edge():
    """Kelly must return 0 when there is no edge."""
    result = kelly_bet(win_prob=0.50, kalshi_price=0.50)
    assert result == 0.0

def test_p_over_known_line():
    """Convolution check: high λs should produce high P(over low line)."""
    prob = p_over_vectorised(lam_home=6.0, lam_away=5.0, line=4.5)
    assert prob > 0.95   # ~11 expected runs vs 4.5 line

def test_p_over_symmetry():
    """P(over) + P(under) + P(exact) ≈ 1.0."""
    lam_h, lam_a, line = 4.5, 4.0, 8.5
    p_over = p_over_vectorised(lam_h, lam_a, line)
    p_under = p_over_vectorised(lam_a, lam_h, 9.5 - 0.001)  # approximate
    assert 0.0 < p_over < 1.0
    assert 0.0 < p_under < 1.0
```

## Elo Zero-Sum Test

Must run after every batch of Elo updates:

```python
def test_elo_zero_sum(db_conn):
    """Total Elo across all teams must be constant after updates."""
    before = db_conn.execute(
        "SELECT SUM(elo) FROM elo_ratings WHERE date = '2024-04-01'"
    ).fetchone()[0]
    after = db_conn.execute(
        "SELECT SUM(elo) FROM elo_ratings WHERE date = '2024-04-02'"
    ).fetchone()[0]
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-4)
```

## Integration Test Structure

tests/integration/test_pipeline.py must cover:
- Full pipeline runs end-to-end on a small date range (3 games)
- Output written to predictions table
- No exceptions raised

tests/integration/test_temporal.py must cover:
- Walk-forward folds never overlap
- Feature matrix contains no future data in any fold
