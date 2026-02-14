import torch
from typing import Optional


def cyc_pos_encoding(
    target_dirs_deg: torch.Tensor,
    n_cond_emb_dim: int,
    d_alpha: int = 20,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    DSENet-style cyclic positional encoding for angles in degrees.

    Args:
        target_dirs_deg: Tensor of angles in degrees. Shape [B] or [B, 1] or scalar.
        n_cond_emb_dim: Output embedding dim. Must be even.
        d_alpha: Scaling factor (DSENet uses 20 by default).
        device: Optional torch.device override.

    Returns:
        Tensor of shape [B, n_cond_emb_dim].
    """
    if n_cond_emb_dim % 2 != 0:
        raise ValueError("n_cond_emb_dim must be even for cyclic positional encoding")

    if not torch.is_tensor(target_dirs_deg):
        target_dirs_deg = torch.tensor(target_dirs_deg)

    if device is None:
        device = target_dirs_deg.device

    theta = torch.deg2rad(target_dirs_deg.to(device=device, dtype=torch.float32))

    if theta.ndim == 0:
        theta = theta.view(1)
    elif theta.ndim == 2 and theta.shape[-1] == 1:
        theta = theta.squeeze(-1)

    pe = torch.zeros(theta.shape[0], n_cond_emb_dim, device=theta.device, dtype=theta.dtype)
    for j in range(n_cond_emb_dim // 2):
        angle = d_alpha / (10000 ** (2 * j / n_cond_emb_dim))
        pe[:, 2 * j] = torch.sin(torch.sin(theta) * angle)
        pe[:, 2 * j + 1] = torch.sin(torch.cos(theta) * angle)

    return pe
