import asyncio
import json
import time
import ollama
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
from datetime import datetime
import re
from typing import List, Optional, Dict, Any

# ────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────

DATA_PATH = Path("/home/nckh2/qa/SLM/data/AVGO_1d_full.csv")

MODEL_NAME = "gemma3:4b"
# MODEL_NAME = "qwen3:4b"
# MODEL_NAME = "phi3:3.8b"
# MODEL_NAME = "atla/selene-mini"

TEST_START_DATE = "2025-05-28"

LOOKBACKS = [1]
FORECAST_HORIZONS = [7]

TEMPERATURE = 0.1
TOP_P = 0.90

# Tune this based on your GPU VRAM (start with 4–8)
MAX_CONCURRENT = 8

# ────────────────────────────────────────────────
#  SYSTEM PROMPT TEMPLATE (made dynamic)
# ────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are a stock analyst expert
Your task is to predict the next {horizon} daily Closing prices.
Input data includes historical Open, High, Low, Volume of the past {lookback} days. Use only the provided data.
Before forecasting, understand the underlying trend, momentum, volatility, and volume changes carefully to make realistic predictions.
Predictions must be as close as possible to real-world closing prices.
Output exactly {horizon} numbers separated by semicolon with 3 decimal places.

Follow this exact closing price template: {example_template}
Example: {example_values}

No text, no words, no explanations, no brackets, no newlines.
"""

# ────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────

def prepare_input_json(df: pd.DataFrame, lookback: int) -> str:
    recent = df.tail(lookback).copy()
    recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")
    
    data_list = recent[["Date", "open", "high", "low", "close", "volume"]].to_dict("records")
    
    payload = {
        "symbol": "ORCL",
        "timeframe": "1d",
        "lookback_days": lookback,
        "data": data_list
    }
    
    return json.dumps(payload, separators=(",", ":"), indent=None)


def parse_prediction(text: str, horizon: int) -> Optional[List[float]]:
    if not text:
        return None

    text = text.strip()
    text = re.sub(r'^[^0-9.;\-]+', '', text)
    text = re.sub(r'[^0-9.;\-]+$', '', text)

    parts = [p.strip() for p in text.split(';') if p.strip()]
    
    if len(parts) < horizon:
        for sep in [',', '\n', ' ']:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if len(parts) >= horizon:
                break

    if len(parts) < horizon:
        parts = re.findall(r'-?\d+\.\d{1,6}', text)

    if len(parts) < horizon:
        return None

    try:
        preds = []
        for s in parts[:horizon]:
            s = s.replace(',', '.')
            val = float(s)
            preds.append(round(val, 3))
        return preds
    except (ValueError, TypeError):
        return None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.nan

    return {"mae": mae, "rmse": rmse, "mape": mape}


# ────────────────────────────────────────────────
#  ASYNC PREDICTION WORKER
# ────────────────────────────────────────────────

async def predict_one(
    idx: int,
    df: pd.DataFrame,
    semaphore: asyncio.Semaphore,
    client: ollama.AsyncClient,
    lookback: int,
    horizon: int
) -> Optional[Dict[str, Any]]:
    async with semaphore:
        start = time.perf_counter()

        window = df.iloc[idx - lookback : idx]
        actual = df["close"].iloc[idx : idx + horizon].values
        json_input = prepare_input_json(window, lookback)

        # Dynamically generate system prompt
        example_template = ";".join(["number"] * horizon)
        example_values = ";".join(["142.350"] * horizon)  # Placeholder example, adjust if needed for longer horizons
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            horizon=horizon,
            lookback=lookback,
            example_template=example_template,
            example_values=example_values
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"ORCL daily data (JSON):\n\n{json_input}\n\nNext {horizon} closing prices:"}
        ]

        try:
            response = await client.chat(
                model=MODEL_NAME,
                messages=messages,
                options={
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                }
            )
            raw = response["message"]["content"].strip()
        except Exception as e:
            print(f"[{df['Date'].iloc[idx].date()}] LLM error: {e}")
            return None

        preds = parse_prediction(raw, horizon)
        date_str = df["Date"].iloc[idx].strftime("%Y-%m-%d")

        duration = time.perf_counter() - start

        if preds is None or len(preds) != horizon:
            print(f"[{date_str}] Parse failed ({duration:.2f}s)")
            return {
                "date": date_str,
                "raw_output": raw,
                "parse_failed": True
            }

        actual_rounded = [round(x, 3) for x in actual]

        print(f"[{date_str}] ({duration:.2f}s)")
        print(f"  Actual   : {actual_rounded}")
        print(f"  Predicted: {preds}")

        return {
            "date": date_str,
            "actual": actual_rounded,
            "predicted": preds,
            "raw_output": raw,
        }


# ────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────

async def main_async():
    print(f"Loading: {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    except Exception as e:
        print(f"Load failed: {e}")
        return

    print(f"Range: {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}")
    print(f"Rows: {len(df)}\n")

    # Find the starting index for test period
    test_mask = df["Date"] >= pd.to_datetime(TEST_START_DATE)
    if not test_mask.any():
        print("No data after test start date.")
        return
    test_start_idx = test_mask.idxmax()  # First row >= TEST_START_DATE

    # Ensure enough pre-data for max lookback
    max_lookback = max(LOOKBACKS)
    if test_start_idx < max_lookback:
        print("Not enough historical data before test start.")
        return

    overall_start = time.perf_counter()

    client = ollama.AsyncClient()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    all_results = {}
    for lookback in LOOKBACKS:
        for horizon in FORECAST_HORIZONS:
            print(f"\n══ Testing lookback={lookback}, horizon={horizon} ══")

            min_samples_for_test = lookback + horizon
            test_len = len(df) - test_start_idx
            if test_len < min_samples_for_test:
                print("Not enough test data for this combo.")
                continue

            tasks = []
            for i in range(test_start_idx, len(df) - horizon + 1):
                tasks.append(predict_one(i, df, semaphore, client, lookback, horizon))

            results_raw = await asyncio.gather(*tasks, return_exceptions=True)

            # ── Process results ───────────────────────────────────────
            results = []
            parse_fail_count = 0
            failed_examples = []

            for res in results_raw:
                if isinstance(res, Exception):
                    parse_fail_count += 1
                    continue
                if res is None or res.get("parse_failed"):
                    parse_fail_count += 1
                    if res and len(failed_examples) < 3:
                        failed_examples.append({"date": res["date"], "raw": res.get("raw_output", "")[:350]})
                    continue
                results.append(res)

            if not results:
                print("No valid predictions collected.")
                if failed_examples:
                    print(f"Parse failures: {parse_fail_count}")
                    for ex in failed_examples:
                        print(f"[{ex['date']}] {ex['raw']}\n")
                continue

            df_res = pd.DataFrame(results)
            n_valid = len(df_res)
            print(f"Valid predictions : {n_valid}")
            print(f"Parse failures    : {parse_fail_count}")
            print(f"Date range        : {df_res['date'].min()} → {df_res['date'].max()}")

            # ── SUMMARY ───────────────────────────────────────────────
            print("\n" + "═" * 80)
            print(f"BACKTEST SUMMARY for lookback={lookback}, horizon={horizon}")
            print("═" * 80 + "\n")

            for h in range(1, horizon + 1):
                y_true = df_res["actual"].apply(lambda x: x[h-1] if len(x) >= h else np.nan).dropna().values
                y_pred = df_res["predicted"].apply(lambda x: x[h-1] if len(x) >= h else np.nan).dropna().values
                if len(y_true) == 0:
                    continue
                m = compute_metrics(y_true, y_pred)
                print(f"+{h:2d}d   MAE: {m['mae']:8.4f}   RMSE: {m['rmse']:8.4f}   MAPE: {m['mape']:6.2f}%")

            # Save per combo
            key = f"lb{lookback}_fh{horizon}"
            all_results[key] = results
            out_file = Path(f"ORCL_slm_pred_{key}_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nSaved → {out_file}")

    total_time = time.perf_counter() - overall_start
    print(f"\nTotal time for all combos: {total_time:.1f} seconds")

if __name__ == "__main__":
    # Recommended environment variables (set before running python):
    # export OLLAMA_NUM_PARALLEL=6          # or 8, 10, … depending on VRAM
    # export OLLAMA_FLASH_ATTENTION=true    # if supported
    # export OLLAMA_MAX_QUEUE=512           # optional

    asyncio.run(main_async())