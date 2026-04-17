# FS-Gemma3: Stock Price Prediction with Small Language Model

This project investigates the capability of small language models (SLMs) for financial time series forecasting. Using Gemma 3 (4B parameters), the system predicts 28-day ahead daily closing prices given structured historical OHLCV data. Predictions are evaluated using standard regression metrics and translated into trading performance via a non-overlapping long-only backtester.

## Project Structure

```bash
thequantscientist-fs-gemma3/
├── src/
│   ├── main.py              # Main asynchronous forecasting pipeline
│   ├── backtest.py          # Non-overlapping backtesting engine
│   ├── analytics.py         # Results aggregation and insight generation
│   ├── tradition.py         # Traditional baseline models
│   ├── transformers.py      # Transformer models benchmarking
└── results/
    ├── gemma/
    │   ├── result/          # Raw predictions and per-configuration metrics (JSON)
    │   └── insights/
    │       ├── backtest/    # Per-symbol backtest summaries
    │       └── ultimate/    # Global best configurations by metric and profit
    ├── traditional/         # Classical ML/statistical benchmark results
    └── transformers/        # Transformer-based model benchmarks
```

## Methodology

### Forecasting Pipeline
- **Model**: Gemma 3 4B (via Ollama)
- **Input**: Structured JSON containing the most recent `lookback` days of OHLCV data
- **Output**: Exactly 28 future daily closing prices
- **Lookback windows**: 1, 7, 14, 21, 28 days
- **Forecast horizon**: 28 days
- **Evaluation steps**: 1, 7, 14, 21, 28 days ahead

### Backtesting
- Non-overlapping trades based on 28-day horizon predictions
- Long positions only when the predicted price at horizon exceeds current price
- Transaction cost: 0.05% per side (0.10% round-trip)
- Initial capital: $100,000
- Performance metrics: Total return, P&L, number of trades, annualized Sharpe ratio

## Results

Detailed results are available in the `results/gemma/` directory:

- Raw prediction files and per-step error metrics (MAE, RMSE, MAPE)
- Backtest performance for each symbol and lookback configuration
- Aggregated insights identifying best-performing lookback windows per symbol and globally

Stocks evaluated include major U.S. equities across technology, healthcare, financials, and other sectors.

## Requirements

- Python 3.10+
- Ollama with `gemma3:4b` model
- Required packages:
  ```bash
  pip install ollama pandas numpy scikit-learn
  ```

## Usage

1. Place daily OHLCV data files (`SYMBOL_1d_full.csv`) in the `data/` directory.
2. Ensure Ollama is running and the `gemma3:4b` model is available.
3. Run the forecasting pipeline:
   ```bash
   python src/main.py
   ```
4. Run backtesting and analysis (see `src/backtest.py` and `src/analytics.py`).

## Configuration

Key parameters are defined at the top of `src/main.py` and `src/backtest.py`:
- Test period start: `2025-05-28`
- Temperature: 0.1
- Top-p: 0.90
- Maximum concurrent requests: 12

## Repository Contents

- `src/main.py`: Core forecasting loop with async parallelization and output parsing
- `src/backtest.py`: Deterministic non-overlapping backtester
- `results/`: All generated predictions, metrics, and insights