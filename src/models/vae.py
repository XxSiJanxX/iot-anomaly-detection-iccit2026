"""
vae.py
------
Variational Autoencoder (VAE) for unsupervised IoT anomaly detection.

Architecture (from proposal):
  Input -> Encoder(128->64) -> μ, log_σ² -> Reparameterise -> z (32-dim)
         -> Decoder(64->128) -> Output (Sigmoid)

Loss: ELBO = BCE(reconstruction) + beta * KL-divergence(q(z|x) || N(0,I))
beta is tuned on the validation set.

Anomaly score: per-sample ELBO reconstruction loss at inference time.
A record is flagged as anomalous if its score > threshold.

Threshold (REVISION, Reviewer 3 comment 5):
  * Main threshold = a percentile p of the BENIGN validation scores, with
    p in {90..99.9} chosen by validation MCC. Choosing p by MCC uses the
    validation labels of both classes, so this is SUPERVISED,
    validation-based calibration, and is reported as such.
  * A second, LABEL-FREE threshold is also stored: the 99th percentile of
    the benign TRAINING scores (no attack labels involved). It is reported
    as "VAE (label-free threshold)".
Early stopping monitors the loss on BENIGN validation rows only.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import joblib

from utils.logger import get_logger
from utils.timer import Timer
from configs.config import (
    VAE_LATENT_DIM, VAE_HIDDEN_DIMS, VAE_DROPOUT_RATE, VAE_L2_REG,
    VAE_LEARNING_RATE, VAE_BATCH_SIZE, VAE_MAX_EPOCHS, VAE_PATIENCE,
    VAE_BETA_RANGE, OUT_MODELS, RANDOM_SEED,
)

log = get_logger(__name__)
tf.random.set_seed(RANDOM_SEED)


# ------------------------------------------------------------------------------
# SAMPLING LAYER (reparameterisation trick)
# ------------------------------------------------------------------------------

class Sampling(layers.Layer):
    """z = μ + ε * exp(log_σ² / 2),  ε ~ N(0, I)"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ------------------------------------------------------------------------------
# VAE MODEL
# ------------------------------------------------------------------------------

class VAEModel(keras.Model):
    """Full VAE as a Keras Model subclass."""

    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dims: list, dropout_rate: float, l2_reg: float,
                 beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.input_dim   = input_dim
        self.latent_dim  = latent_dim
        self.beta        = beta
        self.total_loss_tracker       = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker       = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker          = keras.metrics.Mean(name="kl_loss")

        reg = regularizers.l2(l2_reg)

        # -- ENCODER -----------------------------------------------
        encoder_inputs = keras.Input(shape=(input_dim,), name="encoder_input")
        x = encoder_inputs
        for i, units in enumerate(hidden_dims):
            x = layers.Dense(units, activation="relu",
                              kernel_regularizer=reg,
                              name=f"enc_{i}")(x)
            if i == len(hidden_dims) - 1:          # dropout on last encoder layer
                x = layers.Dropout(dropout_rate)(x)

        z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z         = Sampling(name="z")([z_mean, z_log_var])

        self.encoder = keras.Model(
            encoder_inputs, [z_mean, z_log_var, z], name="encoder"
        )

        # -- DECODER -----------------------------------------------
        latent_inputs = keras.Input(shape=(latent_dim,), name="decoder_input")
        x = latent_inputs
        for i, units in enumerate(reversed(hidden_dims)):
            x = layers.Dense(units, activation="relu",
                              kernel_regularizer=reg,
                              name=f"dec_{i}")(x)
            if i == 0:                              # dropout on first decoder layer
                x = layers.Dropout(dropout_rate)(x)

        decoder_outputs = layers.Dense(input_dim, activation="sigmoid",
                                       name="decoder_output")(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.recon_loss_tracker,
                self.kl_loss_tracker]

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstruction = self.decoder(z, training=training)
        return reconstruction, z_mean, z_log_var

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            recon, z_mean, z_log_var = self(x, training=True)
            recon_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(x, recon)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        recon, z_mean, z_log_var = self(x, training=False)
        recon_loss = tf.reduce_mean(keras.losses.binary_crossentropy(x, recon))
        kl_loss    = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        total_loss = recon_loss + self.beta * kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Per-sample ELBO reconstruction loss (higher = more anomalous).
        Returns 1-D array of floats.

        IMPORTANT: Uses z_mean directly (NOT the stochastically sampled z)
        for the reconstruction. The Sampling layer injects fresh random
        noise (tf.random.normal) on every call, which means anomaly_score()
        previously returned a DIFFERENT result each time it was called on
        the same data -- even at inference time. This made any threshold
        tuned on one call of anomaly_score(X_val) invalid for a later call
        of anomaly_score(X_test), since the two calls were scoring different
        random noise realizations, not just different data. Using z_mean
        (the encoder's deterministic best estimate of the latent code)
        makes anomaly_score() deterministic and reproducible, which is the
        standard practice for VAE inference/scoring (sampling is only
        needed during training for the reparameterisation trick).
        """
        X_tensor = tf.cast(X, tf.float32)
        z_mean, z_log_var, _ = self.encoder(X_tensor, training=False)
        recon = self.decoder(z_mean, training=False)   # deterministic reconstruction
        # REVISION (bug fix): the old code wrapped keras' binary_crossentropy
        # -- which ALREADY averages over the feature axis and returns one
        # value per record -- in a second reduce_mean(axis=-1). That second
        # mean collapsed the per-record losses into ONE batch-level number,
        # so every record received the same reconstruction term and only the
        # KL term differed between records; the score of a record also
        # depended on which other records were scored with it. Per-record
        # BCE is now computed explicitly over the feature axis.
        r = tf.clip_by_value(recon, 1e-7, 1.0 - 1e-7)
        recon_loss = -tf.reduce_mean(
            X_tensor * tf.math.log(r) + (1.0 - X_tensor) * tf.math.log(1.0 - r),
            axis=-1,
        ).numpy()
        kl = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
        ).numpy()
        return recon_loss + self.beta * kl


# ------------------------------------------------------------------------------
# TRAINER WRAPPER
# ------------------------------------------------------------------------------

class TrainedVAE:
    """Wrapper that holds the trained VAE, optimal beta, threshold, and metadata."""

    def __init__(self):
        self.vae:           VAEModel  = None
        self.best_beta:     float     = 1.0
        self.threshold:     float     = None
        self.train_time_s:  float     = 0.0
        self.model_size_mb: float     = 0.0
        self.history:       dict      = {}
        self.input_dim:     int       = 0
        self.threshold_percentile: float = None
        self.threshold_labelfree:  float = None
        self.val_mcc:       float     = None
        self.epochs_trained: int      = 0
        self.n_benign_train: int      = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions using tuned threshold."""
        scores = self.vae.anomaly_score(X)
        return (scores > self.threshold).astype(int)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return self.vae.anomaly_score(X)

    def predict_labelfree(self, X: np.ndarray) -> np.ndarray:
        return (self.vae.anomaly_score(X) > self.threshold_labelfree).astype(int)

    def meta(self) -> dict:
        return {
            "best_beta":    self.best_beta,
            "threshold":    self.threshold,
            "threshold_percentile": self.threshold_percentile,
            "threshold_labelfree":  self.threshold_labelfree,
            "val_mcc":      self.val_mcc,
            "epochs_trained": self.epochs_trained,
            "n_benign_train": self.n_benign_train,
            "train_time_s": self.train_time_s,
            "model_size_mb":self.model_size_mb,
            "input_dim":    self.input_dim,
        }

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        wpath = os.path.join(save_dir, "vae_weights.weights.h5")
        self.vae.save_weights(wpath)
        # REVISION: model size = size of the saved weights file (was never set)
        self.model_size_mb = round(os.path.getsize(wpath) / (1024 ** 2), 3)
        meta = self.meta()
        joblib.dump(meta, os.path.join(save_dir, "vae_meta.pkl"))
        log.info(f"VAE saved -> {save_dir}")

    @classmethod
    def load(cls, save_dir: str, input_dim: int) -> "TrainedVAE":
        obj = cls()
        meta = joblib.load(os.path.join(save_dir, "vae_meta.pkl"))
        obj.best_beta    = meta["best_beta"]
        obj.threshold    = meta["threshold"]
        obj.train_time_s = meta["train_time_s"]
        obj.input_dim    = meta["input_dim"]
        obj.model_size_mb = meta.get("model_size_mb", 0.0)
        obj.threshold_percentile = meta.get("threshold_percentile")
        obj.threshold_labelfree  = meta.get("threshold_labelfree")
        obj.val_mcc        = meta.get("val_mcc")
        obj.epochs_trained = meta.get("epochs_trained", 0)
        obj.n_benign_train = meta.get("n_benign_train", 0)
        obj.vae = VAEModel(
            input_dim=meta["input_dim"],
            latent_dim=VAE_LATENT_DIM,
            hidden_dims=VAE_HIDDEN_DIMS,
            dropout_rate=VAE_DROPOUT_RATE,
            l2_reg=VAE_L2_REG,
            beta=meta["best_beta"],
        )
        obj.vae.build((None, meta["input_dim"]))
        obj.vae.load_weights(os.path.join(save_dir, "vae_weights.weights.h5"))
        return obj


# ------------------------------------------------------------------------------
# TRAINING FUNCTION
# ------------------------------------------------------------------------------

def train_vae(
    X_train_benign: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    dataset_name: str,
) -> TrainedVAE:
    """
    Train VAE on benign-only data, tune beta and anomaly threshold on val set.

    Parameters
    ----------
    X_train_benign : scaled benign training samples (output of vae pipeline)
    X_val          : scaled full validation set
    y_val          : validation labels (0/1)
    dataset_name   : for logging and saving

    Returns
    -------
    TrainedVAE object with tuned model, beta, and threshold
    """
    from sklearn.metrics import f1_score
    input_dim = X_train_benign.shape[1]
    best_f1   = -1.0
    best_tv   = None

    log.info(f"[{dataset_name}] VAE beta-tuning over: {VAE_BETA_RANGE}")
    X_val_benign = X_val[np.asarray(y_val) == 0]

    for beta in VAE_BETA_RANGE:
        log.info(f"[{dataset_name}] Training VAE with beta={beta}...")

        vae = VAEModel(
            input_dim=input_dim,
            latent_dim=VAE_LATENT_DIM,
            hidden_dims=VAE_HIDDEN_DIMS,
            dropout_rate=VAE_DROPOUT_RATE,
            l2_reg=VAE_L2_REG,
            beta=beta,
        )
        vae.compile(optimizer=keras.optimizers.Adam(VAE_LEARNING_RATE))

        cb = [
            keras.callbacks.EarlyStopping(
                monitor="val_total_loss",
                patience=VAE_PATIENCE,
                restore_best_weights=True,
                mode="min",
            )
        ]

        with Timer(f"VAE beta={beta}") as t:
            history = vae.fit(
                X_train_benign,
                X_train_benign,   # target = input (reconstruction)
                epochs=VAE_MAX_EPOCHS,
                batch_size=VAE_BATCH_SIZE,
                validation_data=(X_val_benign, X_val_benign),
                callbacks=cb,
                verbose=0,
            )

        # Tune threshold on validation set (optimises MCC, not F1 --
        # see _tune_threshold docstring for why this matters on imbalanced data)
        val_scores = vae.anomaly_score(X_val)
        best_thresh, best_val_f1_at_mcc, best_pct = _tune_threshold(val_scores, y_val)

        # Compute MCC at the chosen threshold for selection + logging
        from sklearn.metrics import matthews_corrcoef as _skmcc
        val_preds_at_thresh = (val_scores > best_thresh).astype(int)
        val_mcc = _skmcc(y_val, val_preds_at_thresh)

        log.info(f"[{dataset_name}] beta={beta} | "
                 f"Val MCC={val_mcc:.4f} | Val F1={best_val_f1_at_mcc:.4f} | "
                 f"Threshold={best_thresh:.6f} | "
                 f"Time={t.elapsed:.1f}s")

        # Select best beta by MCC, not F1 -- avoids picking a degenerate
        # "predict majority class" configuration on imbalanced val sets.
        if val_mcc > best_f1:   # best_f1 variable now tracks best MCC seen
            best_f1 = val_mcc
            tv = TrainedVAE()
            tv.vae          = vae
            tv.best_beta    = beta
            tv.threshold    = best_thresh
            tv.train_time_s = round(t.elapsed, 2)
            tv.history      = {
                "loss":     history.history.get("total_loss", []),
                "val_loss": history.history.get("val_total_loss", []),
            }
            tv.input_dim = input_dim
            tv.threshold_percentile = best_pct
            tv.val_mcc = round(float(val_mcc), 4)
            tv.epochs_trained = len(history.history.get("total_loss", []))
            tv.n_benign_train = int(len(X_train_benign))
            tv.threshold_labelfree = float(np.percentile(
                vae.anomaly_score(X_train_benign), 99))
            best_tv = tv

    log.info(f"[{dataset_name}] Best VAE: beta={best_tv.best_beta} | "
             f"Val MCC={best_f1:.4f} | Threshold={best_tv.threshold:.6f} "
             f"(benign-val percentile {best_tv.threshold_percentile}) | "
             f"label-free threshold (p99 benign train)="
             f"{best_tv.threshold_labelfree:.6f}")

    # Save
    save_dir = os.path.join(OUT_MODELS, f"{dataset_name}_VAE")
    best_tv.save(save_dir)

    return best_tv


def _tune_threshold(scores: np.ndarray, labels: np.ndarray,
                    percentiles: tuple = (90, 91, 92, 93, 94, 95, 96, 97,
                                           98, 99, 99.5, 99.9)) -> tuple:
    """
    Select the anomaly threshold as a high percentile of the BENIGN-ONLY
    validation scores, choosing whichever percentile (from `percentiles`)
    gives the best validation MCC (Matthews Correlation Coefficient).

    WHY PERCENTILE-OF-BENIGN, NOT AN EXACT MCC-OPTIMAL SWEEP:
    An earlier version of this function swept every threshold between
    scores.min() and scores.max() and picked whichever exact cutpoint
    maximised MCC. That approach failed catastrophically on N-BaIoT:
    Val MCC=0.9875 but Test MCC=0.0000 / FPR=1.0000 (diagnosed in
    diagnose_vae.py). The root cause: benign reconstruction scores are
    compressed into an extremely narrow band (val-normal range was
    ~40.045-40.130, a spread of ~0.08 units). An exact MCC-optimal sweep
    finds a cutpoint with near-zero margin inside that narrow band --
    it overfits to the specific noise of the validation split. A small,
    statistically normal shift between val and test splits (test-normal
    range was ~40.121-40.205) was then enough to push the ENTIRE
    test-normal population above that threshold.

    Selecting a high percentile of benign-only scores instead builds in
    a deliberate safety margin: the threshold sits above most of the
    benign distribution's mass rather than balanced on its exact edge,
    so it tolerates small val->test distribution shift. We still choose
    *which* percentile via validation MCC, so the selection remains
    data-driven rather than an arbitrary fixed percentile.

    WHY MCC AND NOT F1 for scoring each candidate:
    On imbalanced validation sets (e.g. N-BaIoT val is ~84-92% attack),
    a threshold near the minimum score flags nearly everything as "attack".
    This produces a deceptively HIGH F1 score purely because most samples
    genuinely are attacks (high recall dominates F1 when positive class is
    the majority) -- NOT because the model actually discriminates between
    normal and anomalous reconstruction error. MCC penalises this
    degenerate "predict everything as positive" behaviour because it
    accounts for both classes symmetrically (true negatives matter).

    Returns (best_threshold, best_f1_at_that_threshold, percentile).
    """
    from sklearn.metrics import f1_score as skf1, matthews_corrcoef as skmcc

    benign_scores = scores[labels == 0]
    if len(benign_scores) < 10:
        # Not enough benign validation samples to get a stable percentile;
        # fall back to the median of all scores as a safe non-degenerate default.
        best_t = float(np.median(scores))
        preds = (scores > best_t).astype(int)
        return best_t, skf1(labels, preds, zero_division=0), None

    best_t, best_mcc, best_f_at_best_mcc, best_p = None, -1.0, 0.0, None
    for p in percentiles:
        t = float(np.percentile(benign_scores, p))
        preds = (scores > t).astype(int)
        # Skip degenerate thresholds: everything flagged, or nothing flagged
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        mcc = skmcc(labels, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_t = t
            best_p = p
            best_f_at_best_mcc = skf1(labels, preds, zero_division=0)

    # Fallback: if every percentile candidate was degenerate (shouldn't
    # normally happen), use the 99th percentile of benign scores directly.
    if best_t is None:
        best_t = float(np.percentile(benign_scores, 99))
        best_p = 99
        preds = (scores > best_t).astype(int)
        best_f_at_best_mcc = skf1(labels, preds, zero_division=0)

    return best_t, best_f_at_best_mcc, best_p
