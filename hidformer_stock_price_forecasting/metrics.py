"""Evaluation criteria from Section 5.3 of the paper:
  - Weighted MSE training loss (weights decay linearly from T_y down to 1).
  - MAE, MSE, MAPE prediction-accuracy metrics.
  - Net value trading strategy, Eq. (2)-(3).
  - Mann-Whitney U significance test between models.
"""

import numpy as np
import tensorflow as tf
from scipy.stats import mannwhitneyu


def weighted_mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Weighted MSE: weight for horizon step i (1-indexed, i=1..T_y) is
    (T_y - i + 1), i.e. decaying linearly from T_y down to 1, so nearer-term
    predictions are penalized more heavily. Usable directly as a
    tf.keras loss function."""
    pred_len = tf.shape(y_pred)[-1]
    weights = tf.linspace(tf.cast(pred_len, tf.float32), 1.0, pred_len)
    weights = weights / tf.reduce_sum(weights)
    se = tf.square(tf.cast(y_pred, tf.float32) - tf.cast(y_true, tf.float32))
    return tf.reduce_mean(tf.reduce_sum(se * weights, axis=-1))


def prediction_accuracy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8):
    """Returns MAE, MSE, MAPE(%) over all predicted points."""
    mae = np.mean(np.abs(pred - target))
    mse = np.mean((pred - target) ** 2)
    mape = np.mean(np.abs((pred - target) / (target + eps))) * 100
    return mae, mse, mape


def net_value_series(pred_next: np.ndarray, true_close: np.ndarray) -> np.ndarray:
    """
    Implements Eq. (2)-(3): a simple long/short-one-unit strategy based on
    the model's one-step-ahead prediction.
      pred_next[t]  : predicted close at t+1, made using info up to t
      true_close[t] : true close price sequence (t = 0..N-1)
    Returns the cumulative net-value series NV_t (Eq. 3), starting at 1.
    """
    y_t = true_close[:-1]
    y_tp1 = true_close[1:]
    y_hat_tp1 = pred_next[:-1]

    signs = np.sign(y_hat_tp1 - y_t)
    returns = np.log(y_tp1 / y_t) * signs  # R_{t+1}, Eq. (2)

    nv = np.ones(len(returns) + 1)
    nv[1:] = 1 + np.cumsum(returns)
    return nv  # NV_0 = 1, NV_1..NV_{t}


def significance_test(metric_a: np.ndarray, metric_b: np.ndarray):
    """Mann-Whitney U test between two models' per-run metric values
    (e.g. 5 independent-run MAE scores each), as used in Section 5.4."""
    stat, p_value = mannwhitneyu(metric_a, metric_b, alternative="two-sided")
    return stat, p_value
