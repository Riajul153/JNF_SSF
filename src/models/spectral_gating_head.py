"""
Spectral Gating Head (SGH) — lightweight post-processing module for
JNF-SSF beamformer refinement.

The SGH takes the complex STFT output of the frozen JNF-SSF model,
computes a log-magnitude representation, and predicts a [0, 1] frequency
mask via a small MLP.  The mask is applied element-wise to the complex
spectrogram so that phase is preserved while artifact-heavy frequency
bands are suppressed.

Reference
---------
See the accompanying LaTeX document
"Frequency-Selective Spectral Gating for JNF-SSF Beamformer Refinement".
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Spectral Gating Head (the trainable "Stage 2" module)
# ---------------------------------------------------------------------------

class SpectralGatingHead(nn.Module):
    """
    3-layer MLP that predicts a sigmoid mask M(t, f) from log-magnitude
    features.  Designed to be plugged after a frozen JNF-SSF backbone.

    Parameters
    ----------
    n_freqs : int
        Number of STFT frequency bins (nfft // 2 + 1).
    hidden_dim : int
        Width of the two hidden layers (default 256).
    eps : float
        Small constant for numerical stability in log (default 1e-7).
    """

    def __init__(self, n_freqs: int, hidden_dim: int = 256, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

        self.mlp = nn.Sequential(
            nn.Linear(n_freqs, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, n_freqs),
            nn.Sigmoid(),  # ensures M(t,f) ∈ [0, 1]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : complex Tensor  [B, F, T]
            Complex STFT output of the frozen JNF-SSF model.

        Returns
        -------
        Y_hat : complex Tensor  [B, F, T]
            Masked (enhanced) complex spectrogram.
        """
        mag = torch.abs(X)                              # [B, F, T]
        log_mag = torch.log(mag + self.eps)              # [B, F, T]

        # MLP operates per-frame across frequencies
        log_mag_t = log_mag.permute(0, 2, 1)             # [B, T, F]
        M_t = self.mlp(log_mag_t)                        # [B, T, F]
        M = M_t.permute(0, 2, 1)                         # [B, F, T]

        Y_hat = X * M                                    # phase preserved
        return Y_hat

    def predict_mask(self, X: torch.Tensor) -> torch.Tensor:
        """Return only the mask (useful for visualisation / debugging)."""
        mag = torch.abs(X)
        log_mag = torch.log(mag + self.eps)
        log_mag_t = log_mag.permute(0, 2, 1)
        M_t = self.mlp(log_mag_t)
        return M_t.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Band-Weighted MSE loss
# ---------------------------------------------------------------------------

class BandWeightedLoss(nn.Module):
    """
    Frequency-weighted magnitude MSE loss that up-weights aliasing bands.

    Parameters
    ----------
    n_freqs : int
        Number of frequency bins.
    fs : int
        Sampling rate in Hz.
    nfft : int
        FFT size.
    alias_bands : list[tuple[float, float]]
        List of (low_hz, high_hz) ranges to up-weight.
    lambda_high : float
        Weight multiplier for the aliasing bands (default 10.0).
    """

    def __init__(
        self,
        n_freqs: int,
        fs: int = 16000,
        nfft: int = 512,
        alias_bands: list = None,
        lambda_high: float = 10.0,
    ):
        super().__init__()
        if alias_bands is None:
            alias_bands = [(2000.0, 2500.0)]  # default: the 2.1 kHz alias region

        W = torch.ones(n_freqs)
        freq_resolution = fs / nfft  # Hz per bin
        for lo, hi in alias_bands:
            bin_lo = int(lo / freq_resolution)
            bin_hi = min(int(hi / freq_resolution) + 1, n_freqs)
            W[bin_lo:bin_hi] = lambda_high

        # register as buffer so it moves with .to(device) but isn't a parameter
        self.register_buffer("W", W)

    def forward(
        self,
        Y_hat: torch.Tensor,
        Y_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        Y_hat : complex Tensor [B, F, T]
            Enhanced output from the SGH.
        Y_target : complex Tensor [B, F, T]
            Clean anechoic reference.

        Returns
        -------
        loss : scalar Tensor
        """
        mag_hat = torch.abs(Y_hat)
        mag_target = torch.abs(Y_target)

        # W is [F] → broadcast to [1, F, 1]
        W = self.W.unsqueeze(0).unsqueeze(-1)

        loss = torch.mean(W * (mag_target - mag_hat) ** 2)
        return loss
