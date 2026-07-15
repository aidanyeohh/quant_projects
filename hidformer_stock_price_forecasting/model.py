"""
Hidformer: Hierarchical Dual-Tower Transformer for stock price forecasting.
TensorFlow / Keras implementation.

Implementation follows the architecture described in:
Szydlowski, K.L. & Chudziak, J.A. "Hidformer: Transformer-Style Neural
Network in Stock Price Forecasting" (arXiv:2412.19932), which adapts
Liu et al.'s Hidformer (Expert Systems with Applications, 2024) to the
stock-forecasting setting.

Key architectural components (Section 4.2 of the paper):
  1. Two-Tower architecture: one encoder operates on the raw (time-domain)
     sequence, the other on its frequency-domain representation (FFT).
  2. Segment-and-merge: the input sequence is chopped into overlapping
     subsequences ("tokens"); after every encoder block, adjacent tokens
     are merged (downsampled) to build a hierarchical, multi-scale
     representation.
  3. Recursive attention (time tower) / Linear attention (frequency tower):
     classic O(L^2) multi-head attention is replaced by a recursive
     (segment-recurrent) attention in the time tower, and a linear-time
     attention approximation in the frequency tower.
  4. MLP-type decoder: concatenated tower outputs are decoded by a plain
     multi-layer perceptron in one shot (generative-style, cf. Informer).
"""

from __future__ import annotations
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --------------------------------------------------------------------------
# 1. Segment-and-merge tokenization
# --------------------------------------------------------------------------
class SegmentTokenizer(layers.Layer):
    """Splits an input sequence [B, T, C] into N_T non-overlapping (padded)
    segments and flattens each segment into a single N_E-dimensional
    embedding token, producing [B, N_T, N_E]."""

    def __init__(self, seq_len: int, n_tokens: int, in_channels: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.n_tokens = n_tokens
        self.in_channels = in_channels
        self.seg_len = math.ceil(seq_len / n_tokens)
        padded_len = self.seg_len * n_tokens
        self.pad = padded_len - seq_len
        self.proj = layers.Dense(embed_dim)

    def call(self, x):
        # x: [B, T, C]
        if self.pad > 0:
            x = tf.pad(x, [[0, 0], [0, self.pad], [0, 0]])
        B = tf.shape(x)[0]
        x = tf.reshape(x, (B, self.n_tokens, self.seg_len * self.in_channels))
        return self.proj(x)  # [B, N_T, N_E]


# --------------------------------------------------------------------------
# 2. Merge layer (hierarchical downsampling between blocks)
# --------------------------------------------------------------------------
class MergeLayer(layers.Layer):
    """Merges pairs of adjacent tokens: [B, N, E] -> [B, ceil(N/2), E].
    Used between encoder blocks to build a multi-scale representation
    while keeping the last dimension fixed (via a linear projection)."""

    def __init__(self, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.norm = layers.LayerNormalization()
        self.proj = layers.Dense(embed_dim)

    def call(self, x):
        B = tf.shape(x)[0]
        N = x.shape[1]
        E = self.embed_dim
        if N % 2 == 1:
            x = tf.pad(x, [[0, 0], [0, 1], [0, 0]])
            N += 1
        x = tf.reshape(x, (B, N // 2, 2 * E))
        x = self.norm(x)
        return self.proj(x)  # [B, N/2, E]


# --------------------------------------------------------------------------
# 3a. Recursive attention (time tower)
# --------------------------------------------------------------------------
class RecursiveAttention(layers.Layer):
    """A segment-recurrent multi-head attention: the token sequence is
    processed chunk-by-chunk, carrying a compressed running memory state
    forward (similar in spirit to Transformer-XL / recurrent attention),
    which gives the time tower a recurrence-flavoured inductive bias while
    still allowing attention over the current chunk plus the memory."""

    def __init__(self, embed_dim: int, n_heads: int = 4, chunk_size: int = 2,
                 mem_len: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.chunk_size = chunk_size
        self.mem_len = mem_len

        self.q_proj = layers.Dense(embed_dim)
        self.k_proj = layers.Dense(embed_dim)
        self.v_proj = layers.Dense(embed_dim)
        self.out_proj = layers.Dense(embed_dim)
        self.dropout = layers.Dropout(dropout)

        # GRU cell compresses each processed chunk into the running memory
        self.mem_update = layers.GRUCell(embed_dim)

    def _split_heads(self, x, seq_len):
        B = tf.shape(x)[0]
        x = tf.reshape(x, (B, seq_len, self.n_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # [B,H,N,Dh]

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        N = x.shape[1]
        E = self.embed_dim

        memory = tf.zeros((B, self.mem_len, E), dtype=x.dtype)
        outputs = []

        n_chunks = math.ceil(N / self.chunk_size)
        for c in range(n_chunks):
            start, end = c * self.chunk_size, min((c + 1) * self.chunk_size, N)
            c_len = end - start
            chunk = x[:, start:end, :]  # [B, c_len, E]

            kv_input = tf.concat([memory, chunk], axis=1)
            kv_len = self.mem_len + c_len
            q = self._split_heads(self.q_proj(chunk), c_len)
            k = self._split_heads(self.k_proj(kv_input), kv_len)
            v = self._split_heads(self.v_proj(kv_input), kv_len)

            scores = tf.matmul(q, k, transpose_b=True) / math.sqrt(self.head_dim)
            attn = self.dropout(tf.nn.softmax(scores, axis=-1), training=training)
            out = tf.matmul(attn, v)  # [B,H,c_len,Dh]
            out = tf.transpose(out, perm=[0, 2, 1, 3])
            out = tf.reshape(out, (B, c_len, E))
            out = self.out_proj(out)
            outputs.append(out)

            # update memory recurrently with the mean-pooled chunk output
            chunk_summary = tf.reduce_mean(out, axis=1)  # [B, E]
            mem_flat = tf.reshape(memory, (B * self.mem_len, E))
            summary_rep = tf.repeat(chunk_summary[:, None, :], self.mem_len, axis=1)
            summary_rep = tf.reshape(summary_rep, (B * self.mem_len, E))
            new_mem, _ = self.mem_update(summary_rep, [mem_flat])
            memory = tf.reshape(new_mem, (B, self.mem_len, E))

        return tf.concat(outputs, axis=1)  # [B, N, E]


# --------------------------------------------------------------------------
# 3b. Linear attention (frequency tower)
# --------------------------------------------------------------------------
class LinearAttention(layers.Layer):
    """Linear-time attention (Katharopoulos et al., 2020 style): replaces
    softmax(QK^T)V with phi(Q)(phi(K)^T V), reducing complexity from
    O(N^2) to O(N). Used in the frequency-domain tower, which is well
    suited to a smoother/lower-frequency-biased kernel featuremap."""

    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.q_proj = layers.Dense(embed_dim)
        self.k_proj = layers.Dense(embed_dim)
        self.v_proj = layers.Dense(embed_dim)
        self.out_proj = layers.Dense(embed_dim)
        self.dropout = layers.Dropout(dropout)

    @staticmethod
    def _feature_map(x):
        # elu(x) + 1 keeps values positive, a standard linear-attention kernel
        return tf.nn.elu(x) + 1.0

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        N = x.shape[1]
        E = self.embed_dim

        def split_heads(t):
            t = tf.reshape(t, (B, N, self.n_heads, self.head_dim))
            return tf.transpose(t, perm=[0, 2, 1, 3])  # [B,H,N,Dh]

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        q, k = self._feature_map(q), self._feature_map(k)

        # linear attention: O(N) instead of O(N^2)
        kv = tf.einsum("bhnd,bhne->bhde", k, v)               # [B,H,Dh,Dh]
        k_sum = tf.reduce_sum(k, axis=2)                       # [B,H,Dh]
        numerator = tf.einsum("bhnd,bhde->bhne", q, kv)        # [B,H,N,Dh]
        denominator = tf.einsum("bhnd,bhd->bhn", q, k_sum)[..., None] + 1e-6
        out = numerator / denominator
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (B, N, E))
        out = self.dropout(out, training=training)
        return self.out_proj(out)


# --------------------------------------------------------------------------
# 4. Encoder block (shared structure, pluggable attention)
# --------------------------------------------------------------------------
class EncoderBlock(layers.Layer):
    def __init__(self, embed_dim: int, attn_layer: layers.Layer, ff_mult: int = 4,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = attn_layer
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.ff = keras.Sequential([
            layers.Dense(ff_mult * embed_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
        ])
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        x = x + self.dropout(self.attn(self.norm1(x), training=training), training=training)
        x = x + self.dropout(self.ff(self.norm2(x), training=training), training=training)
        return x


# --------------------------------------------------------------------------
# 5. A full tower: N_B x (encoder block + merge), with block outputs
#    concatenated at the tower output (per Section 4.2 / Fig. 3).
# --------------------------------------------------------------------------
class Tower(layers.Layer):
    def __init__(self, embed_dim: int, n_blocks: int, attn_type: str,
                 n_heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        assert attn_type in ("recursive", "linear")
        self.n_blocks = n_blocks
        self.blocks = []
        self.merges = []
        for i in range(n_blocks):
            if attn_type == "recursive":
                attn = RecursiveAttention(embed_dim, n_heads=n_heads, dropout=dropout)
            else:
                attn = LinearAttention(embed_dim, n_heads=n_heads, dropout=dropout)
            self.blocks.append(EncoderBlock(embed_dim, attn, dropout=dropout))
            if i < n_blocks - 1:
                self.merges.append(MergeLayer(embed_dim))

    def call(self, x, training=False):
        # Each block's (mean-pooled) output is kept and concatenated at the
        # end, giving a multi-scale summary of the tower as described in
        # the paper ("at the output of the tower there are concatenated
        # outputs of each block").
        block_outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x, training=training)
            block_outputs.append(tf.reduce_mean(x, axis=1))  # [B, E] pooled summary
            if i < len(self.merges):
                x = self.merges[i](x)
        return tf.concat(block_outputs, axis=-1)  # [B, E * n_blocks]


# --------------------------------------------------------------------------
# 6. MLP-type decoder
# --------------------------------------------------------------------------
class MLPDecoder(layers.Layer):
    def __init__(self, hidden_dim: int, out_len: int, n_layers: int = 2,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        net = []
        for _ in range(n_layers - 1):
            net += [layers.Dense(hidden_dim, activation="gelu"), layers.Dropout(dropout)]
        net.append(layers.Dense(out_len))
        self.net = keras.Sequential(net)

    def call(self, x, training=False):
        return self.net(x, training=training)  # [B, out_len]


# --------------------------------------------------------------------------
# 7. Full Hidformer model
# --------------------------------------------------------------------------
class Hidformer(keras.Model):
    """
    Args (mirrors the paper's hyperparameter notation, Section 5.2):
        seq_len   : T_x, length of the input history window
        pred_len  : T_y, number of future steps predicted
        in_channels : number of input features per day (O,H,L,C,AdjC,Vol = 6)
        n_tokens  : N_T, number of segment tokens
        embed_dim : N_E, per-token embedding dimension
        n_blocks  : N_B, number of blocks per tower
        n_decoder_layers : N_D, number of MLP decoder layers
    """

    def __init__(self, seq_len: int = 128, pred_len: int = 128, in_channels: int = 6,
                 n_tokens: int = 4, embed_dim: int = 16, n_blocks: int = 3,
                 n_heads: int = 4, n_decoder_layers: int = 2, decoder_hidden: int = 128,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_channels = in_channels

        # --- time-domain tower ---
        self.time_tokenizer = SegmentTokenizer(seq_len, n_tokens, in_channels, embed_dim)
        self.time_tower = Tower(embed_dim, n_blocks, attn_type="recursive", n_heads=n_heads,
                                 dropout=dropout)

        # --- frequency-domain tower (rFFT magnitude, resampled to seq_len) ---
        self.freq_tokenizer = SegmentTokenizer(seq_len, n_tokens, in_channels, embed_dim)
        self.freq_tower = Tower(embed_dim, n_blocks, attn_type="linear", n_heads=n_heads,
                                 dropout=dropout)

        self.decoder = MLPDecoder(decoder_hidden, pred_len, n_layers=n_decoder_layers,
                                   dropout=dropout)

    def _to_frequency_domain(self, x):
        # x: [B, T, C] -> real-valued spectral representation, same shape,
        # via rFFT along time then resampling back to seq_len so it can be
        # tokenized identically to the raw series.
        x_t = tf.transpose(x, perm=[0, 2, 1])           # [B, C, T]
        freq = tf.signal.rfft(x_t)                       # [B, C, F] complex
        mag = tf.abs(freq)                                # [B, C, F]
        mag = mag[..., None]                              # [B, C, F, 1] for image resize
        mag = tf.image.resize(mag, size=(mag.shape[1], self.seq_len), method="bilinear")
        mag = tf.squeeze(mag, axis=-1)                    # [B, C, T]
        mag = tf.transpose(mag, perm=[0, 2, 1])           # [B, T, C]
        return mag

    def call(self, x, training=False):
        """x: [B, seq_len, in_channels] -> returns [B, pred_len] (close price)."""
        time_tokens = self.time_tokenizer(x)
        time_repr = self.time_tower(time_tokens, training=training)

        freq_input = self._to_frequency_domain(x)
        freq_tokens = self.freq_tokenizer(freq_input)
        freq_repr = self.freq_tower(freq_tokens, training=training)

        fused = tf.concat([time_repr, freq_repr], axis=-1)
        return self.decoder(fused, training=training)


if __name__ == "__main__":
    # quick smoke test
    model = Hidformer(seq_len=128, pred_len=128, in_channels=6,
                       n_tokens=4, embed_dim=16, n_blocks=3, n_decoder_layers=2)
    dummy = tf.random.normal((8, 128, 6))
    out = model(dummy, training=False)
    n_params = sum(int(tf.size(w)) for w in model.trainable_variables)
    print("output shape:", out.shape)
    print("parameter count:", n_params)
