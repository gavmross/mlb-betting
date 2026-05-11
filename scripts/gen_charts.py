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

PRODUCTION = {
    "sizing": "quarter_kelly",
    "kelly_mult": 0.50,
    "kelly_cap": 0.15,
}

FILTER_CONFIGS = [
    {"name": "day_k9_park",        "label": "day_k9_park  (UNDER · 56.4%)",        "color": "#5B9BD5"},
    {"name": "high_line",           "label": "high_line  (UNDER · 57.5%)",           "color": "#70AD47"},
    {"name": "summer_hot_wind_out", "label": "summer_hot_wind_out  (OVER · 63.1%)", "color": "#ED7D31"},
    {"name": "all",                 "label": "All Three Combined",                   "color": "#7030A0"},
]


# ── Chart 1 — Production equity curve + drawdown ──────────────────────────────

print("Running combined simulation (production)...")
r_all = simulate_structural(
    start=START, end=END,
    filters=FILTERS,
    output_path="/tmp/prod_all.csv",
    initial_bankroll=BANKROLL,
    **PRODUCTION,
)
df_all = pd.read_csv("/tmp/prod_all.csv")
df_all["date"] = pd.to_datetime(df_all["date"])

dates = [pd.to_datetime(START)] + df_all["date"].tolist()
bankroll = np.array([BANKROLL] + df_all["bankroll_after"].tolist())
peak = np.maximum.accumulate(bankroll)
dd = (peak - bankroll) / peak * 100

print(
    f"  Combined: ${r_all['bankroll_final']:,.0f}  "
    f"Sharpe {r_all['sharpe_annualised']:.2f}  "
    f"Win rate {r_all['win_rate']:.1f}%  "
    f"DD -{r_all['max_drawdown']:.1f}%  "
    f"n={r_all['bets_placed']}"
)

fig, (ax_eq, ax_dd) = plt.subplots(
    2, 1, figsize=(12, 8),
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
)

ax_eq.fill_between(dates, BANKROLL, bankroll,
                   where=np.array(bankroll) >= BANKROLL,
                   alpha=0.12, color="#70AD47")
ax_eq.fill_between(dates, BANKROLL, bankroll,
                   where=np.array(bankroll) < BANKROLL,
                   alpha=0.15, color="#C00000")
ax_eq.plot(dates, bankroll, color="#4472C4", linewidth=2.0)
ax_eq.axhline(BANKROLL, color="black", linestyle=":", linewidth=0.9, alpha=0.4)

# Annotate final value
ax_eq.annotate(
    f"${r_all['bankroll_final']:,.0f}",
    xy=(dates[-1], bankroll[-1]),
    xytext=(-60, 12), textcoords="offset points",
    fontsize=11, fontweight="bold", color="#4472C4",
    arrowprops=dict(arrowstyle="-", color="#4472C4", lw=1),
)

ax_eq.set_ylabel("Bankroll ($)", fontsize=11)
ax_eq.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
ax_eq.grid(True, alpha=0.22)
ax_eq.set_title(
    "Combined Structural Filter Strategy — Half Kelly (0.50x, 15% cap)\n"
    "2021–2025  ·  DraftKings Closing Lines  ·  "
    f"$100 start  |  Win rate {r_all['win_rate']:.1f}%  |  "
    f"Sharpe {r_all['sharpe_annualised']:.2f}  |  "
    f"{r_all['bets_placed']:,} bets",
    fontsize=11, pad=8,
)
ax_eq.tick_params(labelbottom=False)

ax_dd.fill_between(dates, 0, -dd, color="#C00000", alpha=0.35)
ax_dd.plot(dates, -dd, color="#C00000", linewidth=1.0, alpha=0.7)
ax_dd.set_ylabel("Drawdown", fontsize=10)
ax_dd.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}%"))
ax_dd.set_xlabel("Date", fontsize=11)
ax_dd.grid(True, alpha=0.22)
ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_dd.xaxis.set_major_locator(mdates.YearLocator())
ax_dd.set_ylim(top=5)

out1 = os.path.join(OUT_DIR, "equity_curve.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out1}")


# ── Chart 2 — Per-filter breakdown (production sizing) ────────────────────────

print("\nRunning per-filter simulations...")
fig2, ax2 = plt.subplots(figsize=(12, 5.5))

for fc in FILTER_CONFIGS:
    filt = [fc["name"]] if fc["name"] != "all" else FILTERS
    tmp = f"/tmp/filter_{fc['name']}.csv"
    r2 = simulate_structural(
        start=START, end=END,
        filters=filt,
        initial_bankroll=BANKROLL,
        output_path=tmp,
        **PRODUCTION,
    )
    df2 = pd.read_csv(tmp)
    df2["date"] = pd.to_datetime(df2["date"])
    dates2 = [pd.to_datetime(START)] + df2["date"].tolist()
    bk2 = [BANKROLL] + df2["bankroll_after"].tolist()
    lw = 2.5 if fc["name"] == "all" else 1.5
    ls = "-" if fc["name"] == "all" else "--"
    label = (
        f"{fc['label']}   "
        f"${r2['bankroll_final']:,.0f}  |  "
        f"Win {r2['win_rate']:.1f}%  |  "
        f"n={r2['bets_placed']}"
    )
    ax2.plot(dates2, bk2, label=label,
             color=fc["color"], linestyle=ls, linewidth=lw)
    print(f"  {fc['label'][:35]:<35} ${r2['bankroll_final']:>7,.0f}  Win {r2['win_rate']:.1f}%  n={r2['bets_placed']}")

ax2.axhline(BANKROLL, color="black", linestyle=":", linewidth=0.9, alpha=0.35)
ax2.set_ylabel("Bankroll ($)", fontsize=11)
ax2.set_xlabel("Date", fontsize=11)
ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.legend(fontsize=9, loc="upper left", framealpha=0.92)
ax2.grid(True, alpha=0.22)
ax2.set_title(
    "Equity by Filter — Half Kelly (0.50x, 15% cap)  ·  2021–2025  ·  DraftKings Closing Lines",
    fontsize=11, pad=8,
)
fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "filter_breakdown.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")


# ── Chart 3 — Win rate bar chart ──────────────────────────────────────────────

print("\nGenerating win rate chart...")
fig3, ax3 = plt.subplots(figsize=(9, 5))

labels   = ["day_k9_park\n(UNDER)", "high_line\n(UNDER)", "summer_hot_wind_out\n(OVER)", "All Three\nCombined"]
win_rates = [56.4, 57.5, 63.1, 57.4]
samples   = [906,  373,  134,  1413]
colors    = ["#5B9BD5", "#70AD47", "#ED7D31", "#7030A0"]

bars = ax3.bar(labels, win_rates, color=colors, width=0.55,
               edgecolor="white", linewidth=1.2, zorder=3)
ax3.axhline(52.38, color="#C00000", linestyle="--", linewidth=1.6,
            label="Break-even at -110  (52.38%)", zorder=4)
ax3.axhline(50.0, color="black", linestyle=":", linewidth=0.9,
            alpha=0.4, label="Coin flip  (50.0%)", zorder=4)

for bar, wr, n in zip(bars, win_rates, samples):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.35,
        f"{wr:.1f}%\nn={n:,}",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )

ax3.set_ylim(45, 70)
ax3.set_ylabel("Win Rate (%)", fontsize=11)
ax3.set_title(
    "Filter Win Rates vs Break-Even at -110\n2021–2025  ·  DraftKings Closing Lines",
    fontsize=12, pad=8,
)
ax3.legend(fontsize=9.5, loc="upper left")
ax3.grid(axis="y", alpha=0.25, zorder=0)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
fig3.tight_layout()
out3 = os.path.join(OUT_DIR, "win_rates.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"Saved: {out3}")

print("\nAll charts generated.")
