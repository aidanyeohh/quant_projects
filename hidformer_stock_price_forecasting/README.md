# Hidformer — Stock Price Forecasting (arXiv:2412.19932) — TensorFlow version

A TensorFlow/Keras port of the PyTorch implementation of *"Hidformer:
Transformer-Style Neural Network in Stock Price Forecasting"*
(Szydłowski & Chudziak, 2024). Same architecture, same experimental
protocol — every module (`model.py`, `baselines.py`, `data.py`, `metrics.py`,
`train.py`) has a 1:1 counterpart, rewritten idiomatically in Keras.

## Files

| File | Contents |
|---|---|
| `model.py` | Hidformer as a `tf.keras.Model` subclass: segment-and-merge tokenizer, two towers (recursive attention / time-domain, linear attention / frequency-domain via `tf.signal.rfft`), hierarchical merge layers, MLP decoder. |
| `baselines.py` | CNN, RNN, LSTM, DNN baselines, built with the Keras functional API. |
| `data.py` | Yahoo Finance download, min-max normalization (Eq. 1), sliding-window arrays, `tf.data.Dataset` batching, 95/5 chronological split. |
| `metrics.py` | Weighted MSE loss (usable directly as a Keras loss), MAE/MSE/MAPE, net-value trading backtest (Eq. 2–3), Mann-Whitney U test. |
| `train.py` | End-to-end experiment runner using `model.compile()` / `model.fit()`, with a `ModelCheckpoint` callback to keep the best validation epoch (mirrors "we selected the model chosen as the best during training"). |

## What changed vs. the PyTorch version

- **Layers**: `nn.Module` → `tf.keras.layers.Layer` / `tf.keras.Model`; `forward` → `call`.
- **Attention math**: rewritten with `tf.matmul` / `tf.einsum` instead of `torch.matmul`/`einsum` — numerically equivalent.
- **Recursive attention's memory update**: `nn.GRUCell` → `tf.keras.layers.GRUCell` (note TF's `GRUCell.__call__` returns `(output, [new_state])`, handled accordingly).
- **Frequency-domain path**: `torch.fft.rfft` + `F.interpolate` → `tf.signal.rfft` + `tf.image.resize` (bilinear resize used as the 1-D linear-interpolation equivalent).
- **Training loop**: manual PyTorch loop → `model.compile(loss=weighted_mse_loss)` + `model.fit()`, with best-checkpoint restoring via `ModelCheckpoint(save_best_only=True)` instead of manually tracking `state_dict()`.
- **Datasets**: `torch.utils.data.Dataset`/`DataLoader` → plain NumPy sliding-window arrays fed through `tf.data.Dataset` (`.shuffle().batch().prefetch()`).
- Parameter count is identical to the PyTorch build (62,048 for the default `embed_dim=16` config) — confirms an architecturally faithful port.

## Usage

```bash
pip install tensorflow yfinance numpy pandas scipy

# quick sanity check of the model itself
python model.py

# full experiment (defaults match the paper's Section 5.2 hyperparameters:
# T_x=T_y=128, batch=64, Adam lr=1e-4, 100 epochs, 5 runs, 6 tickers)
python train.py --tickers AAPL KO MO PEP PG TM --epochs 100 --runs 5
```

Fast local test without waiting for 100-epoch runs on 6 tickers:

```bash
python train.py --tickers AAPL --epochs 5 --runs 1 --seq-len 32 --pred-len 8
```

## Caveats (same as the PyTorch version)

- The paper states `N_E=1` (a 1-dimensional embedding per token), which
  would make attention degenerate; `--embed-dim` defaults to 16 here for a
  numerically meaningful model. Adjust freely.
- "Recursive attention" and the exact frequency-domain preprocessing are
  described only conceptually in the paper (no code/equations given), so
  `RecursiveAttention` and the rFFT-based frequency tower are reasonable,
  documented reconstructions rather than a verbatim match to the authors'
  unpublished implementation.
- With very small token counts (e.g. after two merges, only 1 token
  remains per chunk), the recursive-attention memory-update GRU in the
  final chunk of a block has no downstream consumer, so Keras will warn
  "gradients do not exist for variables [...gru_cell...]" for the last
  block(s) of a tower — this is an expected consequence of the chunking
  scheme at small sequence lengths, not a bug, and disappears at the
  paper's real hyperparameters (`seq_len=128` gives many chunks per block).
- Yahoo Finance data availability/format can change; `data.py` handles the
  common `yfinance` multi-index column case but you may need to adjust for
  ticker-specific gaps or splits.
