# ================== 28-STEP AHEAD STOCK FORECASTING - FIXED VERSION ==================
import os
import warnings
import math
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Configuration ────────────────────────────────────────────────────
DATA_ROOT = "/home/nckh2/qa/IntraFormer/data"
OUTPUT_DIR = "/home/nckh2/qa/SLM/results/traditional"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STOCK_FILES = {
    'AAPL':  f'{DATA_ROOT}/AAPL_1d_full.csv',
    'ABBV':  f'{DATA_ROOT}/ABBV_1d_full.csv',
    'AMD':   f'{DATA_ROOT}/AMD_1d_full.csv',
    'AMGN':  f'{DATA_ROOT}/AMGN_1d_full.csv',
    'AMZN':  f'{DATA_ROOT}/AMZN_1d_full.csv',
    'AVGO':  f'{DATA_ROOT}/AVGO_1d_full.csv',
    'AXP':   f'{DATA_ROOT}/AXP_1d_full.csv',
    'BAC':   f'{DATA_ROOT}/BAC_1d_full.csv',
    'BLK':   f'{DATA_ROOT}/BLK_1d_full.csv',
    'BMY':   f'{DATA_ROOT}/BMY_1d_full.csv',
    'C':     f'{DATA_ROOT}/C_1d_full.csv',
    'DHR':   f'{DATA_ROOT}/DHR_1d_full.csv',
    'GOOGL': f'{DATA_ROOT}/GOOGL_1d_full.csv',
    'GS':    f'{DATA_ROOT}/GS_1d_full.csv',
    'INTC':  f'{DATA_ROOT}/INTC_1d_full.csv',
    'JNJ':   f'{DATA_ROOT}/JNJ_1d_full.csv',
    'JPM':   f'{DATA_ROOT}/JPM_1d_full.csv',
    'LLY':   f'{DATA_ROOT}/LLY_1d_full.csv',
    'META':  f'{DATA_ROOT}/META_1d_full.csv',
    'MRK':   f'{DATA_ROOT}/MRK_1d_full.csv',
    'MS':    f'{DATA_ROOT}/MS_1d_full.csv',
    'MSFT':  f'{DATA_ROOT}/MSFT_1d_full.csv',
    'NVDA':  f'{DATA_ROOT}/NVDA_1d_full.csv',
    'ORCL':  f'{DATA_ROOT}/ORCL_1d_full.csv',
    'PFE':   f'{DATA_ROOT}/PFE_1d_full.csv',
    'SCHW':  f'{DATA_ROOT}/SCHW_1d_full.csv',
    'SPGI':  f'{DATA_ROOT}/SPGI_1d_full.csv',
    'TMO':   f'{DATA_ROOT}/TMO_1d_full.csv',
    'UNH':   f'{DATA_ROOT}/UNH_1d_full.csv',
    'WFC':   f'{DATA_ROOT}/WFC_1d_full.csv',
}

SEQ_LEN = 60
PRED_LEN = 28
BATCH_SIZE = 128
EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 120
MIN_DELTA = 5e-7
TEST_DAYS = 182
VAL_FRACTION = 0.20

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOSS_FN = nn.MSELoss()

FEATURES = ['open', 'high', 'low', 'close', 'volume']
CLOSE_IDX = FEATURES.index('close')

# ─── Dataset ─────────────────────────────────────────────────────────
class MultiStepDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, CLOSE_IDX]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ─── Models (unchanged) ──────────────────────────────────────────────
class CNNLSTM(nn.Module):
    def __init__(self, input_size=5, pred_len=PRED_LEN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(256, pred_len)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)


class LSTM(nn.Module):
    def __init__(self, input_size=5, pred_len=PRED_LEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, pred_len)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1])


class GRU(nn.Module):
    def __init__(self, input_size=5, pred_len=PRED_LEN):
        super().__init__()
        self.gru = nn.GRU(input_size, 256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, pred_len)

    def forward(self, x):
        o, _ = self.gru(x)
        return self.fc(o[:, -1])


class CLAM(nn.Module):
    def __init__(self, input_size=5, pred_len=PRED_LEN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(128, 200, num_layers=2, batch_first=True)
        self.attn = nn.Linear(200, 1)
        self.fc = nn.Linear(200, pred_len)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm(x)
        w = torch.softmax(self.attn(x), dim=1)
        x = (w * x).sum(dim=1)
        return self.fc(x)


MODELS = {
    'CNN-LSTM': CNNLSTM,
    'LSTM':     LSTM,
    'GRU':      GRU,
    'CLAM':     CLAM
}


# ─── Train Function (slightly cleaned) ───────────────────────────────
def train_model(model, train_loader, val_loader):
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x + torch.randn_like(x) * 0.015

            optimizer.zero_grad()
            loss = LOSS_FN(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += LOSS_FN(model(x), y).item()
                n_batches += 1

        val_loss = val_loss / n_batches if n_batches > 0 else float('inf')
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == EPOCHS - 1:
            print(f"    Epoch {epoch+1:3d} | Val Loss: {val_loss:.6f}")

        if val_loss < best_loss - MIN_DELTA:
            best_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ─── Correct Rolling Test Evaluation (Fixed) ─────────────────────────────
def evaluate(model, full_scaled, close_scaler, df_index, symbol, model_name):
    model.eval()
    preds_list = []
    trues_list = []

    test_start = len(full_scaled) - TEST_DAYS   # index where test begins

    print(f"  → Evaluating {model_name} on {symbol} (rolling test)")

    with torch.no_grad():
        for i in range(TEST_DAYS):
            idx = test_start - SEQ_LEN + i
            if idx < 0:
                continue   # skip if not enough history

            # Get input and true future
            x = full_scaled[idx : idx + SEQ_LEN]
            y = full_scaled[idx + SEQ_LEN : idx + SEQ_LEN + PRED_LEN, CLOSE_IDX]

            # Important: only use if we have full PRED_LEN true values
            if len(y) < PRED_LEN:
                continue

            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            pred = model(x_tensor).cpu().numpy()[0]          # shape (28,)

            preds_list.append(pred)
            trues_list.append(y)                             # shape (28,)

    if len(preds_list) == 0:
        print(f"  Not enough history for test on {symbol}")
        return None, None

    # Now both lists have the same length and full shape
    preds = np.array(preds_list)   # (N, 28)
    trues = np.array(trues_list)   # (N, 28)

    # ─── Horizon metrics ─────────────────────────────────────────────
    horizons = [1, 7, 14, 21, 28]
    metrics = {'symbol': symbol, 'model': model_name}

    print(f"    Evaluated on {len(preds)} test days")

    for h in horizons:
        if h > PRED_LEN:
            continue
        idx = h - 1
        p = preds[:, idx]
        t = trues[:, idx]

        p_real = close_scaler.inverse_transform(p.reshape(-1, 1)).ravel()
        t_real = close_scaler.inverse_transform(t.reshape(-1, 1)).ravel()

        rmse = np.sqrt(np.mean((p_real - t_real)**2))
        mae  = np.mean(np.abs(p_real - t_real))
        mape = np.mean(np.abs((t_real - p_real) / (t_real + 1e-8))) * 100

        metrics[f'rmse_h{h}'] = round(rmse, 4)
        metrics[f'mae_h{h}']  = round(mae, 4)
        metrics[f'mape_h{h}'] = round(mape, 2)

        print(f"    +{h:2d}d → RMSE:{rmse:.4f}  MAE:{mae:.4f}  MAPE:{mape:.2f}%")

    # ─── Save predictions ────────────────────────────────────────────
    # Use only the dates that actually have predictions
    valid_test_dates = df_index[-len(preds):]

    df_pred = pd.DataFrame({
        'symbol': symbol,
        'model': model_name,
        'date': valid_test_dates,
        'true_close': close_scaler.inverse_transform(trues[:, 0].reshape(-1, 1)).ravel(),
        'pred_close': close_scaler.inverse_transform(preds[:, 0].reshape(-1, 1)).ravel(),
    })

    for h in [7, 14, 28]:
        if h <= PRED_LEN:
            df_pred[f'true_t+{h}'] = close_scaler.inverse_transform(trues[:, h-1].reshape(-1, 1)).ravel()
            df_pred[f'pred_t+{h}'] = close_scaler.inverse_transform(preds[:, h-1].reshape(-1, 1)).ravel()

    return metrics, df_pred


# ─── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_metrics = []
    all_preds = []

    for symbol, path in STOCK_FILES.items():
        print(f"\n{'='*70}")
        print(f"Processing {symbol}")
        print(f"{'='*70}")

        try:
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')[FEATURES].sort_index().ffill().bfill()
        except Exception as e:
            print(f"Failed to load {symbol}: {e}")
            continue

        if len(df) < SEQ_LEN + PRED_LEN + TEST_DAYS + 100:
            print(f"Skipping {symbol} (not enough data)")
            continue

        train_df = df.iloc[:-TEST_DAYS]
        test_df  = df.iloc[-TEST_DAYS:]

        # Per-feature scaling (only on train)
        scalers = {f: MinMaxScaler().fit(train_df[[f]]) for f in FEATURES}
        close_scaler = scalers['close']

        train_s = np.hstack([scalers[f].transform(train_df[[f]]) for f in FEATURES])
        test_s  = np.hstack([scalers[f].transform(test_df[[f]]) for f in FEATURES])

        val_size = int(len(train_s) * VAL_FRACTION)
        val_s    = train_s[-val_size:]
        train_s  = train_s[:-val_size]

        train_ds = MultiStepDataset(train_s, SEQ_LEN, PRED_LEN)
        val_ds   = MultiStepDataset(val_s,   SEQ_LEN, PRED_LEN)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

        full_scaled = np.concatenate([train_s, val_s, test_s])

        for model_name, ModelClass in MODELS.items():
            print(f"\n→ Training {model_name} for {symbol} ...")
            
            model = ModelClass().to(DEVICE)
            model = train_model(model, train_loader, val_loader)

            metrics, df_pred = evaluate(model, full_scaled, close_scaler, df.index, symbol, model_name)

            if metrics is not None:
                all_metrics.append(metrics)
                all_preds.append(df_pred)

            torch.cuda.empty_cache()
            gc.collect()

    # ─── Save outputs ────────────────────────────────────────────────
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(OUTPUT_DIR, "traditional_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n✅ Saved metrics → {metrics_path}")

    if all_preds:
        predictions_path = os.path.join(OUTPUT_DIR, "traditional_predictions.csv")
        pd.concat(all_preds, ignore_index=True).to_csv(predictions_path, index=False)
        print(f"✅ Saved predictions → {predictions_path}")

    print("\n🎉 Traditional models benchmark finished successfully!")