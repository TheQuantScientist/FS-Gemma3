import json
import numpy as np

# ==============================
# CONFIG
# ==============================
FILE_PATH       = "/home/nckh2/qa/SLM/results/AAPL_lb14_fh28_predictions_20260221_100228.json"
STEP_AHEAD      = 28
INITIAL_CAPITAL = 100_000
COST_PER_TRADE  = 0.0005   # per side (entry + exit = 2x)
# ==============================


with open(FILE_PATH, "r") as f:
    data = json.load(f)

capital = INITIAL_CAPITAL
equity_curve = [capital]
returns = []
trades = 0

i = 0
while i < len(data):

    entry = data[i]

    # Ensure this entry has full horizon
    if len(entry["actual"]) < STEP_AHEAD:
        i += 1
        continue

    entry_price = entry["actual"][0]                  # price at t+1
    exit_price  = entry["actual"][STEP_AHEAD - 1]     # price at t+STEP
    pred_price  = entry["predicted"][STEP_AHEAD - 1]

    # Long if predicted exit > entry
    if pred_price > entry_price:

        ret = (exit_price - entry_price) / entry_price

        # round-trip cost
        ret -= 2 * COST_PER_TRADE

        capital *= (1 + ret)
        equity_curve.append(capital)
        returns.append(ret)
        trades += 1

        # Skip forward to avoid overlap
        i += STEP_AHEAD

    else:
        i += 1


# ==============================
# STATISTICS
# ==============================

total_return_pct = (capital / INITIAL_CAPITAL - 1) * 100
total_profit     = capital - INITIAL_CAPITAL

if len(returns) > 1:
    mean_ret = np.mean(returns)
    std_ret  = np.std(returns)
    sharpe   = (mean_ret / std_ret) * np.sqrt(252 / STEP_AHEAD)
else:
    sharpe = np.nan

# ==============================
# OUTPUT
# ==============================

print("\n" + "═"*60)
print(f" NON-OVERLAPPING BACKTEST | HORIZON = {STEP_AHEAD}")
print("═"*60)
print(f"Initial capital      : ${INITIAL_CAPITAL:,.0f}")
print(f"Final capital        : ${capital:,.2f}")
print(f"Total P&L            : ${total_profit:+,.2f}")
print(f"Total return         : {total_return_pct:+.2f}%")
print(f"Number of trades     : {trades}")
print(f"Sharpe (annualized)  : {sharpe:.3f}" if not np.isnan(sharpe) else "Sharpe: —")
print("═"*60 + "\n")