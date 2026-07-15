"""
Main experiment script: trains Hidformer + 4 baselines (CNN, RNN, LSTM, DNN)
on 6 Consumer-Goods tickers, replicating the protocol from Section 5 of
"Hidformer: Transformer-Style Neural Network in Stock Price Forecasting"
(arXiv:2412.19932). TensorFlow / Keras implementation.

Usage:
    python train.py --tickers AAPL KO --epochs 100 --runs 5

Notes on faithfulness to the paper:
  * seq_len = pred_len = 128, batch_size = 64, Adam lr=1e-4, epochs=100 (Sec 5.2)
  * Hidformer: N_T=4, N_E=1... (paper states N_E=1, but that collapses the
    embedding to a scalar per token; we default to a slightly larger
    embed_dim for numerical stability, adjustable via --embed-dim.)
  * loss = weighted MSE (weights T_y..1)
  * 5 independent runs per model; report mean/SE of MAE, MSE, MAPE
  * net-value backtest + Mann-Whitney U significance test across runs
"""

from __future__ import annotations
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import Hidformer
from baselines import BASELINES
from data import build_datasets, make_tf_dataset, TICKERS
from metrics import weighted_mse_loss, prediction_accuracy, net_value_series, significance_test


def build_model(name: str, seq_len: int, pred_len: int, in_channels: int, embed_dim: int):
    if name == "Hidformer":
        return Hidformer(seq_len=seq_len, pred_len=pred_len, in_channels=in_channels,
                          n_tokens=4, embed_dim=embed_dim, n_blocks=3, n_decoder_layers=2)
    return BASELINES[name](seq_len=seq_len, pred_len=pred_len, in_channels=in_channels)


def train_one_run(model_name, X_train, y_train, X_val, y_val, seq_len, pred_len,
                   in_channels, embed_dim, epochs, batch_size, lr, seed):
    tf.keras.utils.set_random_seed(seed)
    model = build_model(model_name, seq_len, pred_len, in_channels, embed_dim)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=weighted_mse_loss)

    train_ds = make_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True,
                                drop_remainder=True)
    val_ds = make_tf_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)

    ckpt_path = f"/tmp/{model_name}_best_{seed}.weights.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True,
                                         save_weights_only=True, verbose=0),
    ]
    has_val = len(X_val) > 0
    model.fit(train_ds, validation_data=val_ds if has_val else None, epochs=epochs,
              callbacks=callbacks if has_val else [], verbose=0)

    if has_val:
        try:
            model.load_weights(ckpt_path)
        except Exception:
            pass  # fall back to last-epoch weights if checkpoint wasn't written
    return model


def evaluate(model, X_val, y_val, batch_size=64):
    if len(X_val) == 0:
        return None, None
    pred = model.predict(X_val, batch_size=batch_size, verbose=0)
    return np.asarray(pred), np.asarray(y_val)


def run_experiment(tickers, epochs=100, runs=5, seq_len=128, pred_len=128,
                    embed_dim=16, batch_size=64, lr=1e-4, in_channels=6,
                    start="1980-12-12", end="2023-12-29"):
    model_names = ["CNN", "RNN", "LSTM", "DNN", "Hidformer"]
    results = {m: {"MAE": [], "MSE": [], "MAPE": []} for m in model_names}
    nv_by_model_ticker = {m: {} for m in model_names}

    for ticker in tickers:
        print(f"\n=== {ticker} ===")
        X_train, y_train, X_val, y_val, stats, df = build_datasets(
            ticker, seq_len=seq_len, pred_len=pred_len, start=start, end=end)

        for model_name in model_names:
            run_maes, run_mses, run_mapes = [], [], []
            last_pred, last_true = None, None
            for run in range(runs):
                model = train_one_run(model_name, X_train, y_train, X_val, y_val,
                                       seq_len, pred_len, in_channels, embed_dim,
                                       epochs, batch_size, lr, seed=run)
                pred, true = evaluate(model, X_val, y_val, batch_size)
                if pred is None:
                    continue
                mae, mse, mape = prediction_accuracy(pred, true)
                run_maes.append(mae); run_mses.append(mse); run_mapes.append(mape)
                last_pred, last_true = pred, true
                print(f"  [{model_name} run {run+1}/{runs}] MAE={mae:.4f} MSE={mse:.4f} MAPE={mape:.2f}%")

            results[model_name]["MAE"].append(np.mean(run_maes))
            results[model_name]["MSE"].append(np.mean(run_mses))
            results[model_name]["MAPE"].append(np.mean(run_mapes))

            if last_pred is not None:
                # one-step-ahead close prediction = first column of the pred_len forecast
                pred_next = last_pred[:, 0]
                true_close_seq = np.concatenate([[last_true[0, 0]], last_true[:, 0]])
                nv = net_value_series(pred_next, true_close_seq)
                nv_by_model_ticker[model_name][ticker] = nv

    print("\n=== Prediction accuracy (mean over tickers) ===")
    print(f"{'Model':<10}{'MAE':>10}{'MSE':>10}{'MAPE':>10}")
    for m in model_names:
        print(f"{m:<10}{np.mean(results[m]['MAE']):>10.3f}"
              f"{np.mean(results[m]['MSE']):>10.3f}{np.mean(results[m]['MAPE']):>10.1f}")

    print("\n=== Net value at end of backtest, per ticker ===")
    header = "Model     " + "".join(f"{t:>10}" for t in tickers)
    print(header)
    for m in model_names:
        row = f"{m:<10}"
        for t in tickers:
            nv = nv_by_model_ticker[m].get(t)
            row += f"{(nv[-1] - 1) if nv is not None else float('nan'):>10.3f}"
        print(row)

    # significance test: Hidformer vs each baseline on MAE across tickers
    print("\n=== Mann-Whitney U test: Hidformer vs baselines (MAE across tickers) ===")
    hid_mae = np.array(results["Hidformer"]["MAE"])
    for m in model_names:
        if m == "Hidformer":
            continue
        base_mae = np.array(results[m]["MAE"])
        if len(hid_mae) > 0 and len(base_mae) > 0:
            stat, p = significance_test(hid_mae, base_mae)
            print(f"  Hidformer vs {m}: U={stat:.2f}, p={p:.4f}")

    return results, nv_by_model_ticker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hidformer stock-forecasting experiments (TF)")
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--pred-len", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--start", type=str, default="1980-12-12")
    parser.add_argument("--end", type=str, default="2023-12-29")
    args = parser.parse_args()

    run_experiment(
        tickers=args.tickers, epochs=args.epochs, runs=args.runs,
        seq_len=args.seq_len, pred_len=args.pred_len, embed_dim=args.embed_dim,
        batch_size=args.batch_size, lr=args.lr, start=args.start, end=args.end,
    )
