"""
Generate equity curve charts for the README.
Run from project root: python scripts/gen_charts.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from mlb.betting import simulate_structural

# ── Config ────────────────────────────────────────────────────────────────────

FILTERS = ["day_k9_park", "high_line", "summer_hot_wind_out"]
START = "2021-04-01"
END = "2025-10-01"
BANKROLL = 100.0
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(OUT_DIR, exist_ok=True)

SIZINGS = [
    {
        "label": "Flat 5%",
        "sizing": "flat",
        "flat_bet_pct": 0.05,
        "kelly_mult": 0.25,
        "kelly_cap": 0.05,
        "color": "#888888",
        "ls": "--",
        "lw": 1.6,
    },
    {
        "label": "Quarter Kelly  (0.25x, 5% cap)",
        "sizing": "quarter_kelly",
        "flat_bet_pct": 0.02,
        "kelly_mult": 0.25,
        "kelly_cap": 0.05,
        "color": "#5B9BD5",
        "ls": ":",
        "lw": 1.8,
    },
    {
        "label": "Half Kelly  (0.50x, 15% cap)  [production]",
        "sizing": "quarter_kelly",
        "flat_bet_pct": 0.02,
        "kelly_mult": 0.50,
        "kelly_cap": 0.15,
        "color": "#70AD47",
        "ls": "-",
        "lw": 2.8,
    },
    {
        "label": "Full Kelly  (1.0x, 15% cap)",
        "sizing": "quarter_kelly",
        "flat_bet_pct": 0.02,
        "kelly_mult": 1.0,
        "kelly_cap": 0.15,
        "color": "#ED7D31",
        "ls": "-.",
        "lw": 1.8,
    },
]


# ── Run simulations ───────────────────────────────────────────────────────────

print("Running simulations…")
results = []
for cfg in SIZINGS:
    tmp = f"/tmp/eq_{cfg['label'][:8].replace(' ', '_')}.csv"
    r = simulate_structural(
        start=START,
        end=END,
        filters=FILTERS,
        sizing=cfg["sizing"],
        flat_bet_pct=cfg.get("flat_bet_pct", 0.02),
        kelly_mult=cfg["kelly_mult"],
        kelly_cap=cfg["kelly_cap"],
        initial_bankroll=BANKROLL,
        output_path=tmp,
    )
    df = pd.read_csv(tmp)
    df["date"] = pd.to_datetime(df["date"])
    results.append({"cfg": cfg, "df": df, "r": r})
    print(
        f"  {cfg['label'][:40]:<40} "
        f"${r['bankroll_final']:>8,.0f}  "
        f"Sharpe {r['sharpe_annualised']:.2f}  "
        f"DD {r['max_drawdown']:.1f}%"
    )


# ── Chart 1 — Equity curves ───────────────────────────────────────────────────

fig, axes = plt.subplots(
    2, 1, figsize=(12, 9),
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
)

ax_eq, ax_dd = axes

for res in results:
    cfg = res["cfg"]
    df = res["df"]
    r = res["r"]

    dates = [pd.to_datetime(START)] + df["date"].tolist()
    bankroll = np.array([BANKROLL] + df["bankroll_after"].tolist())
    peak = np.maximum.accumulate(bankroll)
    dd = (peak - bankroll) / peak * 100

    label = (
        f"{cfg['label']}\n"
        f"  Final ${r['bankroll_final']:,.0f} · "
        f"Sharpe {r['sharpe_annualised']:.2f} · "
        f"DD -{r['max_drawdown']:.1f}%"
    )
    ax_eq.plot(dates, bankroll, label=label,
               color=cfg["color"], linestyle=cfg["ls"], linewidth=cfg["lw"])
    ax_dd.plot(dates, -dd,
               color=cfg["color"], linestyle=cfg["ls"], linewidth=cfg["lw"] * 0.8, alpha=0.85)

ax_eq.axhline(BANKROLL, color="black", linestyle=":", linewidth=0.9, alpha=0.35)
ax_eq.set_ylabel("Bankroll ($)", fontsize=11)
ax_eq.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
ax_eq.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
ax_eq.grid(True, alpha=0.25)
ax_eq.set_title(
    "Structural Filter Strategy — Equity Curves by Bet Sizing\n"
    "2021–2025 · day_k9_park + high_line + summer_hot_wind_out · DraftKings Closing Lines",
    fontsize=12, pad=10,
)

ax_dd.set_ylabel("Drawdown (%)", fontsize=10)
ax_dd.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}%"))
ax_dd.set_xlabel("Date", fontsize=11)
ax_dd.grid(True, alpha=0.25)
ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_dd.xaxis.set_major_locator(mdates.YearLocator())

fig.tight_layout()
out1 = os.path.join(OUT_DIR, "equity_curves.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out1}")


# ── Chart 2 — Per-filter breakdown (half Kelly only) ─────────────────────────

FILTER_CONFIGS = [
    {"name": "day_k9_park",          "label": "day_k9_park\n(UNDER · 56.4%)",          "color": "#5B9BD5"},
    {"name": "high_line",             "label": "high_line\n(UNDER · 57.5%)",             "color": "#70AD47"},
    {"name": "summer_hot_wind_out",   "label": "summer_hot_wind_out\n(OVER · 63.1%)",    "color": "#ED7D31"},
    {"name": "all",                   "label": "All Three Combined",                     "color": "#7030A0"},
]

fig2, ax2 = plt.subplots(figsize=(12, 5.5))

for fc in FILTER_CONFIGS:
    filt = [fc["name"]] if fc["name"] != "all" else FILTERS
    tmp = f"/tmp/filter_{fc['name']}.csv"
    r2 = simulate_structural(
        start=START, end=END,
        filters=filt,
        sizing="quarter_kelly",
        kelly_mult=0.50,
        kelly_cap=0.15,
        initial_bankroll=BANKROLL,
        output_path=tmp,
    )
    df2 = pd.read_csv(tmp)
    df2["date"] = pd.to_datetime(df2["date"])
    dates = [pd.to_datetime(START)] + df2["date"].tolist()
    bk = [BANKROLL] + df2["bankroll_after"].tolist()
    lw = 2.8 if fc["name"] == "all" else 1.6
    ls = "-" if fc["name"] == "all" else "--"
    label = (
        f"{fc['label'].replace(chr(10), ' · ')}\n"
        f"  Final ${r2['bankroll_final']:,.0f} · "
        f"Win rate {r2['win_rate']:.1f}% · "
        f"n={r2['bets_placed']}"
    )
    ax2.plot(dates, bk, label=label,
             color=fc["color"], linestyle=ls, linewidth=lw)

ax2.axhline(BANKROLL, color="black", linestyle=":", linewidth=0.9, alpha=0.35)
ax2.set_ylabel("Bankroll ($)", fontsize=11)
ax2.set_xlabel("Date", fontsize=11)
ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
ax2.grid(True, alpha=0.25)
ax2.set_title(
    "Equity Curves by Filter — Half Kelly (0.50×, 15% cap)\n"
    "2021–2025 · DraftKings Closing Lines",
    fontsize=12, pad=10,
)
fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "filter_breakdown.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")


# ── Chart 3 — Win rate bar chart ──────────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(8, 4.5))

filters_display = ["day_k9_park\n(UNDER)", "high_line\n(UNDER)", "summer_hot_wind_out\n(OVER)", "All Three\nCombined"]
win_rates = [56.4, 57.5, 63.1, 57.4]
samples   = [906, 373, 134, 1413]
colors    = ["#5B9BD5", "#70AD47", "#ED7D31", "#7030A0"]

bars = ax3.bar(filters_display, win_rates, color=colors, width=0.55, edgecolor="white", linewidth=1.2)
ax3.axhline(52.38, color="#C00000", linestyle="--", linewidth=1.4, label="Break-even at −110 (52.38%)")
ax3.axhline(50.0, color="black", linestyle=":", linewidth=0.9, alpha=0.4, label="Coin flip (50%)")

for bar, wr, n in zip(bars, win_rates, samples):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{wr:.1f}%\nn={n:,}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

ax3.set_ylim(45, 68)
ax3.set_ylabel("Win Rate (%)", fontsize=11)
ax3.set_title("Filter Win Rates vs Break-Even · 2021–2025 · DraftKings", fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
fig3.tight_layout()
out3 = os.path.join(OUT_DIR, "win_rates.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"Saved: {out3}")

print("\nAll charts generated.")
