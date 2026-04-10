# Python Coding Standards
**Active when:** editing any .py file in the mlb/ directory or tests/

---

## Formatter / Linter

Ruff is the only formatter and linter. It replaces Black, Flake8, and isort.

```bash
ruff check . && ruff format .   # run before every commit
ruff check --fix .              # auto-fix safe issues
```

Never manually format imports or line breaks — let Ruff handle it.

## Type Hints

Required on all public functions and methods:

```python
# CORRECT
def p_over_vectorised(lam_home: float, lam_away: float,
                      line: float, max_runs: int = 30) -> float:
    ...

# WRONG — no type hints
def p_over_vectorised(lam_home, lam_away, line, max_runs=30):
    ...
```

Use `Optional[X]` for nullable, `list[X]` not `List[X]` (Python 3.10+ style).

## Docstrings

NumPy style on all public functions:

```python
def kelly_bet(win_prob: float, kalshi_price: float,
              kelly_mult: float = 0.25) -> float:
    """
    Compute fractional Kelly bet size.

    Parameters
    ----------
    win_prob : float
        Calibrated probability the bet wins (0–1).
    kalshi_price : float
        Kalshi mid-price for the bet side in dollars (0–1).
    kelly_mult : float, optional
        Fractional Kelly multiplier. Default 0.25x.

    Returns
    -------
    float
        Fraction of bankroll to bet (0–0.05).
    """
```

## Error Handling

No bare `except:` — always catch specific exceptions:

```python
# CORRECT
try:
    games = Scoreboard(sport="MLB", date=date).games
except requests.HTTPError as e:
    logger.error("SBR request failed: %s", e)
    return []
except ValueError as e:
    logger.error("SBR parse error on %s: %s", date, e)
    return []

# WRONG
try:
    games = Scoreboard(sport="MLB", date=date).games
except:
    return []
```

## Float Comparisons in Tests

```python
# CORRECT
np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-4)

# WRONG
assert result == expected        # fragile for floats
assert abs(result - expected) < 0.001  # inconsistent threshold
```

## Imports

Standard order (Ruff enforces this automatically):
1. Standard library
2. Third-party
3. Local (`from mlb.db import ...`)

No wildcard imports (`from module import *`).

## Constants

Module-level constants in UPPER_SNAKE_CASE at the top of the file:

```python
MAX_RUNS = 30          # practical upper bound for Poisson convolution
MIN_EDGE = 0.03        # minimum Kalshi edge threshold
KELLY_MULT = 0.25      # fractional Kelly multiplier
MAX_BET_PCT = 0.05     # max fraction of bankroll per game
```

## Logging

Use the standard `logging` module — never `print()` in production code:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Scraped %d games for %s", len(games), date)
logger.warning("Low liquidity on %s: open_interest=%.0f", ticker, oi)
logger.error("Kalshi API error: %s", e)
```

`print()` is acceptable only in notebooks and CLI scripts.

## SQLite Connections

Always enable WAL mode and use context managers:

```python
import sqlite3
from contextlib import contextmanager

@contextmanager
def get_conn(db_path: str = "data/mlb.db"):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
```
