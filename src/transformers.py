import os
import sys
import argparse
import warnings
import math
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore", category=UserWarning)

# Add Time-Series-Library path
sys.path.append('/home/nckh2/qa/Time-Series-Library')
from utils.timefeatures import time_features

from models.Autoformer import Model as Autoformer
from models.FEDformer import Model as FEDformer
from models.Informer import Model as Informer

try:
    from models.iTransformer import Model as iTransformer
    HAS_iTRANSFORMER = True
except ImportError:
    HAS_iTRANSFORMER = False
    print("iTransformer not found – skipping.")

# ─── Configuration ───────────────────────────────────────────────────────────────

DATA_ROOT = "/home/nckh2/qa/IntraFormer/data"

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

FEATURES = ['open', 'high', 'low', 'close', 'volume']
CLOSE_IDX = FEATURES.index('close')

SEQ_LEN      = 60
LABEL_LEN    = 30
PRED_LEN     = 28
EPOCHS       = 1          # Change back to 200 when you want full training
BATCH_SIZE   = 128
LR           = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE     = 120
MIN_DELTA    = 5e-7
TEST_DAYS    = 182
VAL_FRACTION = 0.20

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOSS_FN = nn.MSELoss()

PREFIX = "stock_transformer_per_stock_"
RESULTS_DIR = "/home/nckh2/qa/SLM/results/transformers"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Dataset ────────────────────────────────────────────────────────────────────
class PerStockForecastDataset(Dataset):
    def __init__(self, data: np.ndarray, time_features: np.ndarray,
                 seq_len: int, label_len: int, pred_len: int = 1):
        self.data = data
        self.time_features = time_features
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx):
        s_begin = idx
        s_end   = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end   = s_end + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end, CLOSE_IDX:CLOSE_IDX+1]
        seq_x_mark = self.time_features[s_begin:s_end]
        seq_y_mark = self.time_features[r_begin:r_end]

        return (
            torch.from_numpy(seq_x).float(),
            torch.from_numpy(seq_y).float(),
            torch.from_numpy(seq_x_mark).float(),
            torch.from_numpy(seq_y_mark).float()
        )

# ─── Model Config ───────────────────────────────────────────────────────────────
def get_model_configs():
    common = {
        'seq_len': SEQ_LEN,
        'label_len': LABEL_LEN,
        'pred_len': PRED_LEN,
        'enc_in': 5,
        'dec_in': 5,
        'c_out': 1,
        'd_model': 256,
        'n_heads': 8,
        'e_layers': 3,
        'd_layers': 1,
        'd_ff': 512,
        'dropout': 0.12,
        'activation': 'gelu',
        'embed': 'timeF',
        'freq': 'd',
    }

    models = [
        {'name': 'Autoformer', 'class': Autoformer, 'configs': {**common, 'task_name': 'long_term_forecast', 'factor': 3, 'moving_avg': 25}},
        {'name': 'FEDformer',  'class': FEDformer,  'configs': {**common, 'task_name': 'long_term_forecast', 'factor': 3, 'moving_avg': 25, 'version': 'Fourier', 'mode_select': 'random', 'modes': 32}},
        {'name': 'Informer',   'class': Informer,   'configs': {**common, 'task_name': 'long_term_forecast', 'factor': 5, 'distil': True, 'output_attention': False}},
    ]

    if HAS_iTRANSFORMER:
        models.append({'name': 'iTransformer', 'class': iTransformer, 'configs': {**common, 'task_name': 'long_term_forecast', 'e_layers': 3, 'factor': 3}})

    return models

# ─── Horizon Metrics ────────────────────────────────────────────────────────────
def compute_horizon_metrics(pred_horizon: np.ndarray, true_horizon: np.ndarray):
    if len(pred_horizon) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'mape_pct': np.nan}

    mae = mean_absolute_error(true_horizon, pred_horizon)
    rmse = np.sqrt(mean_squared_error(true_horizon, pred_horizon))
    mape = np.mean(np.abs((true_horizon - pred_horizon) / (true_horizon + 1e-8))) * 100

    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'mape_pct': round(mape, 2)
    }

# ─── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_metrics = []
    all_predictions = []

    model_configs = get_model_configs()

    for symbol, filepath in STOCK_FILES.items():
        print(f"\n{'='*70}\nProcessing {symbol}\n{'='*70}")

        try:
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')[FEATURES].sort_index().ffill().bfill()
        except Exception as e:
            print(f"Load error for {symbol}: {e}")
            continue

        if len(df) < SEQ_LEN + TEST_DAYS + 100:
            print(f"Skipping {symbol} — not enough data")
            continue

        test_df  = df.iloc[-TEST_DAYS:]
        pre_test = df.iloc[:-TEST_DAYS]
        val_size = int(len(pre_test) * VAL_FRACTION)
        val_df   = pre_test.iloc[-val_size:]
        train_df = pre_test.iloc[:-val_size]

        scalers = {col: MinMaxScaler().fit(train_df[[col]]) for col in FEATURES}

        def scale_data(frame):
            return np.hstack([scalers[col].transform(frame[[col]]) for col in FEATURES]).astype(np.float32)

        train_scaled = scale_data(train_df)
        val_scaled   = scale_data(val_df)
        test_scaled  = scale_data(test_df)
        full_scaled  = np.concatenate([train_scaled, val_scaled, test_scaled])

        # Time features
        dates = df.index
        time_feat = time_features(dates, freq='d').T.astype(np.float32)

        pre_test_values = np.concatenate([train_scaled, val_scaled])

        for cfg in model_configs:
            model_name = cfg['name']
            print(f"  → {model_name}")

            args = argparse.Namespace(**cfg['configs'])
            model = cfg['class'](args).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

            train_ds = PerStockForecastDataset(train_scaled, time_feat[:len(train_scaled)], SEQ_LEN, LABEL_LEN, PRED_LEN)
            val_ds   = PerStockForecastDataset(val_scaled,   time_feat[len(train_scaled):len(pre_test_values)], SEQ_LEN, LABEL_LEN, PRED_LEN)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

            best_val_loss = float('inf')
            patience_counter = 0
            best_state = None

            for epoch in range(EPOCHS):
                model.train()
                train_loss = 0.0
                train_batch_count = 0
                for batch in train_loader:
                    x, y, x_mark, y_mark = [t.to(DEVICE) for t in batch]

                    x = x + torch.randn_like(x) * 0.015

                    dec_inp = torch.cat([x[:, -LABEL_LEN:, :],
                                         torch.zeros_like(x[:, -PRED_LEN:, :])], dim=1)

                    out = model(x, x_mark, dec_inp, y_mark)
                    pred = out[:, -PRED_LEN:, :] if out.shape[1] > PRED_LEN else out

                    loss = LOSS_FN(pred, y[:, -PRED_LEN:, :])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()

                    train_loss += loss.item()
                    train_batch_count += 1

                train_avg = train_loss / train_batch_count if train_batch_count > 0 else float('inf')

                # Validation
                model.eval()
                val_loss_sum = 0.0
                n_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        x, y, x_mark, y_mark = [t.to(DEVICE) for t in batch]
                        dec_inp = torch.cat([x[:, -LABEL_LEN:, :],
                                             torch.zeros_like(x[:, -PRED_LEN:, :])], dim=1)
                        out = model(x, x_mark, dec_inp, y_mark)
                        pred = out[:, -PRED_LEN:, :] if out.shape[1] > PRED_LEN else out
                        val_loss_sum += LOSS_FN(pred, y[:, -PRED_LEN:, :]).item()
                        n_batches += 1

                val_loss = val_loss_sum / n_batches if n_batches > 0 else float('inf')
                scheduler.step()

                if (epoch + 1) % 1 == 0 or epoch == EPOCHS - 1:   # print every epoch when EPOCHS=1
                    print(f"    Epoch {epoch+1:3d} | Train {train_avg:.6f} | Val {val_loss:.6f}")

                if val_loss < best_val_loss - MIN_DELTA:
                    best_val_loss = val_loss
                    best_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            # ─── Robust Rolling Test Evaluation ─────────────────────────────────────
            model.eval()
            preds_list = []
            trues_list = []

            test_start = len(pre_test_values)

            print(f"  → Evaluating {model_name} on {symbol} (rolling test)")

            with torch.no_grad():
                for i in range(TEST_DAYS):
                    idx = test_start - SEQ_LEN + i
                    if idx < 0:
                        continue

                    s_begin = idx
                    s_end   = s_begin + SEQ_LEN
                    r_begin = s_end - LABEL_LEN
                    r_end   = s_end + PRED_LEN

                    if r_end > len(full_scaled):
                        continue

                    seq_x      = full_scaled[s_begin:s_end]
                    seq_y      = full_scaled[r_begin:r_end, CLOSE_IDX:CLOSE_IDX+1]
                    seq_x_mark = time_feat[s_begin:s_end]
                    seq_y_mark = time_feat[r_begin:r_end]

                    if seq_x_mark.shape[0] != SEQ_LEN or seq_y_mark.shape[0] != (LABEL_LEN + PRED_LEN):
                        continue

                    x      = torch.from_numpy(seq_x).float().unsqueeze(0).to(DEVICE)
                    y      = torch.from_numpy(seq_y).float().unsqueeze(0).to(DEVICE)
                    x_mark = torch.from_numpy(seq_x_mark).float().unsqueeze(0).to(DEVICE)
                    y_mark = torch.from_numpy(seq_y_mark).float().unsqueeze(0).to(DEVICE)

                    dec_inp = torch.cat([x[:, -LABEL_LEN:, :],
                                         torch.zeros_like(x[:, -PRED_LEN:, :])], dim=1)

                    out = model(x, x_mark, dec_inp, y_mark)
                    pred = out[:, -PRED_LEN:, :] if out.shape[1] > PRED_LEN else out

                    # Robust numpy conversion
                    pred_np = pred.cpu().numpy().squeeze()      # handles (1,28,1), (28,1), (28,)
                    true_np = y[:, -PRED_LEN:, :].cpu().numpy().squeeze()

                    if pred_np.ndim == 1:
                        preds_list.append(pred_np)
                        trues_list.append(true_np)

            if len(preds_list) == 0:
                print(f"  Not enough history for test on {symbol}")
                continue

            preds_test = np.array(preds_list)   # (N, 28)
            trues_test = np.array(trues_list)   # (N, 28)

            print(f"    Evaluated on {len(preds_test)} test days")

            # ─── Horizon metrics ─────────────────────────────────────────────────
            horizons = [1, 7, 14, 21, 28]
            metrics = {'symbol': symbol, 'model': model_name}

            for h in horizons:
                if h > PRED_LEN:
                    continue
                idx = h - 1
                p = preds_test[:, idx]
                t = trues_test[:, idx]

                p_real = scalers['close'].inverse_transform(p.reshape(-1, 1)).ravel()
                t_real = scalers['close'].inverse_transform(t.reshape(-1, 1)).ravel()

                hm = compute_horizon_metrics(p_real, t_real)

                metrics[f'mae_h{h}']   = hm['mae']
                metrics[f'rmse_h{h}']  = hm['rmse']
                metrics[f'mape_h{h}']  = hm['mape_pct']

                print(f"    +{h:2d}d → RMSE:{hm['rmse']:.4f}  MAE:{hm['mae']:.4f}  MAPE:{hm['mape_pct']:.2f}%")

            all_metrics.append(metrics)

            # ─── Save predictions ────────────────────────────────────────────────
            valid_test_dates = test_df.index[:len(preds_test)]

            df_pred = pd.DataFrame({
                'symbol': symbol,
                'model': model_name,
                'date': valid_test_dates,
                'true_close': scalers['close'].inverse_transform(trues_test[:, 0].reshape(-1, 1)).ravel(),
                'pred_close': scalers['close'].inverse_transform(preds_test[:, 0].reshape(-1, 1)).ravel(),
            })

            for h in [7, 14, 28]:
                if h <= PRED_LEN:
                    df_pred[f'true_t+{h}'] = scalers['close'].inverse_transform(trues_test[:, h-1].reshape(-1, 1)).ravel()
                    df_pred[f'pred_t+{h}'] = scalers['close'].inverse_transform(preds_test[:, h-1].reshape(-1, 1)).ravel()

            all_predictions.append(df_pred)

            torch.cuda.empty_cache()
            gc.collect()

    # ─── Save Results ───────────────────────────────────────────────────────────────
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(RESULTS_DIR, f"{PREFIX}metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n✅ Saved detailed metrics → {metrics_path}")

    if all_predictions:
        predictions_path = os.path.join(RESULTS_DIR, f"{PREFIX}predictions.csv")
        pd.concat(all_predictions, ignore_index=True).to_csv(predictions_path, index=False)
        print(f"✅ Saved predictions → {predictions_path}")

    print("\n🎉 Transformer benchmark finished successfully!")