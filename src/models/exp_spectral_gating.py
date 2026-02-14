"""
Experiment class for training the Spectral Gating Head (SGH) on top of a
**frozen** JNF-SSF beamformer.

Usage (standalone)
------------------
    python exp_spectral_gating.py \
        --ssf-checkpoint path/to/ssf-best.ckpt \
        --config        path/to/ssf_flac_config.yaml

Or integrate it into your existing train_jnf_ssf.py workflow — see the
`build_sgh_experiment` helper at the bottom of this file.
"""

import os
import sys
import argparse
from typing import Literal, Optional

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.aggregation import RunningMean

from models.exp_enhancement import EnhancementExp
from models.models import JNF_SSF
from models.spectral_gating_head import SpectralGatingHead, BandWeightedLoss


# ────────────────────────────────────────────────────────────────────────────
# Lightning Experiment
# ────────────────────────────────────────────────────────────────────────────

class SpectralGatingExp(EnhancementExp):
    """
    Two-stage pipeline:
        1. Frozen JNF-SSF  → produces complex spectrogram  X(t,f)
        2. Trainable SGH   → predicts mask M(t,f) and outputs  Y = X ⊙ M
    """

    def __init__(
        self,
        # --- backbone (frozen) ---
        model: nn.Module,                    # JNF_SSF instance (will be frozen)
        # --- SGH hyper-params ---
        sgh_hidden_dim: int = 256,
        sgh_eps: float = 1e-7,
        # --- loss ---
        alias_bands: list = None,
        lambda_high: float = 10.0,
        loss_type: str = "band_weighted",    # "band_weighted" | "l1" | "sisdr"
        # --- optimiser ---
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        # --- STFT ---
        stft_length: int = 512,
        stft_shift: int = 256,
        # --- existing SSF params ---
        cirm_comp_K: float = 10.0,
        cirm_comp_C: float = 0.1,
        n_cond_emb_dim: int = 72,
        condition_enc_type: Literal["index", "arange"] = "arange",
        cond_arange_params: tuple = None,
        reference_channel: int = 0,
        loss_alpha: float = 1.0,
        # --- scheduler ---
        scheduler_type: str = None,
        scheduler_params: dict = None,
    ):
        super().__init__(
            model=model,
            cirm_comp_K=cirm_comp_K,
            cirm_comp_C=cirm_comp_C,
            scheduler_type=scheduler_type,
            scheduler_params=scheduler_params,
        )

        # ── Backbone (frozen) ──────────────────────────────────────────────
        self.model = model
        self._freeze_backbone()

        # ── STFT settings ──────────────────────────────────────────────────
        self.stft_length = stft_length
        self.stft_shift = stft_shift
        n_freqs = stft_length // 2 + 1

        # ── Spectral Gating Head (trainable) ───────────────────────────────
        self.sgh = SpectralGatingHead(
            n_freqs=n_freqs,
            hidden_dim=sgh_hidden_dim,
            eps=sgh_eps,
        )

        # ── Loss ───────────────────────────────────────────────────────────
        self.loss_type = loss_type
        self.loss_alpha = loss_alpha
        if loss_type == "band_weighted":
            self.bw_loss = BandWeightedLoss(
                n_freqs=n_freqs,
                fs=16000,
                nfft=stft_length,
                alias_bands=alias_bands,
                lambda_high=lambda_high,
            )

        # ── Condition encoding (same as SSFExp) ────────────────────────────
        self.n_cond_emb_dim = n_cond_emb_dim
        self.condition_enc_type = condition_enc_type
        self.cond_arange_params = cond_arange_params
        if condition_enc_type == "arange":
            assert cond_arange_params is not None
            start, stop, step = cond_arange_params
            angles = range(start, stop, step)
            self.angle_index_map = dict(zip(angles, range(len(angles))))

        # ── Misc ───────────────────────────────────────────────────────────
        self.cirm_K = cirm_comp_K
        self.cirm_C = cirm_comp_C
        self.reference_channel = reference_channel
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.running_loss = RunningMean(window=20)

    # ── Freeze the backbone ────────────────────────────────────────────────

    def _freeze_backbone(self):
        """Freeze all parameters of the JNF-SSF model."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        print(f"[SpectralGatingExp] Backbone frozen — "
              f"{sum(p.numel() for p in self.model.parameters()):,} params fixed.")

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, noisy_stft: torch.Tensor, target_dirs: torch.Tensor):
        """
        Full pipeline: frozen JNF-SSF → SGH.

        Parameters
        ----------
        noisy_stft : complex Tensor [B, C, F, T]
        target_dirs : Tensor [B]

        Returns
        -------
        enhanced_stft : complex Tensor [B, F, T]
        """
        stacked = torch.cat(
            [noisy_stft.real, noisy_stft.imag], dim=1
        )  # [B, 2C, F, T]

        target_dirs_enc = self.encode_condition(target_dirs)

        with torch.no_grad():
            if self.model.output_type == "IRM":
                mask = self.model(stacked, target_dirs_enc, device=self.device)
                speech_mask = mask
            elif self.model.output_type == "CRM":
                stacked_mask = self.model(stacked, target_dirs_enc, device=self.device)
                speech_mask, _ = self.get_complex_masks_from_stacked(stacked_mask)

        # JNF-SSF output (complex spectrogram)
        X = noisy_stft[:, self.reference_channel, ...] * speech_mask  # [B, F, T]

        # Spectral Gating Head (trainable)
        Y_hat = self.sgh(X)
        return Y_hat

    # ── Training / Validation steps ────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.shared_step(batch, batch_idx, stage="val",
                                dataloader_idx=dataloader_idx)

    def shared_step(self, batch, batch_idx, stage="train", dataloader_idx=0):
        noisy_td = batch["noisy_td"]
        clean_td = batch["clean_td"]
        target_dirs = batch["target_dir"]

        noisy_stft, clean_stft = self.get_stft_rep(noisy_td, clean_td)

        # Forward through frozen backbone + trainable SGH
        Y_hat = self.forward(noisy_stft, target_dirs)

        # Target: clean reference channel
        Y_target = clean_stft[:, self.reference_channel, ...]

        # ── Compute loss ───────────────────────────────────────────────────
        if self.loss_type == "band_weighted":
            loss = self.bw_loss(Y_hat, Y_target)
        elif self.loss_type == "l1":
            loss = torch.mean(torch.abs(torch.abs(Y_target) - torch.abs(Y_hat)))
        elif self.loss_type == "sisdr":
            est_td = self.get_td_rep(Y_hat)[0]
            ref_td = self.get_td_rep(Y_target)[0]
            loss = -torch.mean(self.compute_global_si_sdr(est_td, ref_td))
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # ── SI-SDR metric ──────────────────────────────────────────────────
        with torch.no_grad():
            est_clean_td = self.get_td_rep(Y_hat)[0]
            ref_clean_td = self.get_td_rep(Y_target)[0]
            si_sdr = torch.mean(self.compute_global_si_sdr(est_clean_td, ref_clean_td))

        # ── Logging ────────────────────────────────────────────────────────
        add_dl_idx = stage != "train" and dataloader_idx != 0
        self.running_loss(loss)
        on_step = stage == "train"
        self.log(f"{stage}/loss", self.running_loss.compute(),
                 on_step=on_step, on_epoch=True, logger=True,
                 add_dataloader_idx=add_dl_idx, sync_dist=True, prog_bar=True)
        self.log(f"{stage}/si_sdr", si_sdr,
                 on_step=False, on_epoch=True, logger=True,
                 prog_bar=True, add_dataloader_idx=add_dl_idx, sync_dist=True)

        if stage == "val" and dataloader_idx == 0:
            self.log("monitor_loss", loss, on_step=False, on_epoch=True,
                     logger=True, sync_dist=True)

        if batch_idx < 1:
            noisy_ref = noisy_stft[:, self.reference_channel, ...]
            self.log_batch_detailed_spectrograms(
                [noisy_ref, Y_target, Y_hat], batch_idx, stage, n_samples=10
            )
            self.log_batch_detailed_audio(
                noisy_td[:, 0, ...], est_clean_td, batch_idx, stage
            )

        return loss

    # ── Optimiser (only SGH params) ────────────────────────────────────────

    def configure_optimizers(self):
        # Only optimise the SGH parameters
        opt = torch.optim.Adam(
            self.sgh.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.scheduler_type in {"ReduceLROnPLateau", "ReduceLROnPlateau"}:
            sched = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, **(self.scheduler_params or {})
                ),
                "name": "lr_schedule",
                "monitor": "monitor_loss",
            }
            return {"optimizer": opt, "lr_scheduler": sched}
        if self.scheduler_type == "MultiStepLR":
            sched = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    opt, **(self.scheduler_params or {})
                ),
                "name": "lr_schedule",
            }
            return {"optimizer": opt, "lr_scheduler": sched}
        return opt

    # ── Condition encoder (reused from SSFExp) ─────────────────────────────

    def encode_condition(self, target_dirs):
        if self.condition_enc_type == "index":
            return torch.nn.functional.one_hot(
                target_dirs, self.n_cond_emb_dim
            ).float()
        elif self.condition_enc_type == "arange":
            mapped = target_dirs.cpu().apply_(self.angle_index_map.get).to(self.device)
            return torch.nn.functional.one_hot(
                mapped, self.n_cond_emb_dim
            ).float()

    # ── Ensure backbone stays in eval mode ─────────────────────────────────

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()   # backbone always eval (frozen BN, dropout off)
        return self

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage="test")


# ────────────────────────────────────────────────────────────────────────────
# Helper: build experiment from an SSF checkpoint + config
# ────────────────────────────────────────────────────────────────────────────

def build_sgh_experiment(
    ssf_config: dict,
    ssf_checkpoint_path: str,
    stft_length: int,
    stft_shift: int,
    sgh_hidden_dim: int = 256,
    alias_bands: list = None,
    lambda_high: float = 10.0,
    loss_type: str = "band_weighted",
    learning_rate: float = 1e-3,
) -> SpectralGatingExp:
    """
    Convenience function: loads a trained JNF-SSF from a Lightning checkpoint
    and wraps it with a fresh SGH for fine-tuning.
    """
    from models.exp_ssf import SSFExp

    # ── Load the trained SSF experiment ────────────────────────────────────
    ssf_model = JNF_SSF(**ssf_config["network"])
    ssf_exp = SSFExp(
        model=ssf_model,
        stft_length=stft_length,
        stft_shift=stft_shift,
        **ssf_config["experiment"],
    )
    ckpt = torch.load(ssf_checkpoint_path, map_location="cpu")
    ssf_exp.load_state_dict(ckpt["state_dict"])
    print(f"[build_sgh_experiment] Loaded SSF weights from {ssf_checkpoint_path}")

    # ── Extract the trained JNF_SSF model ──────────────────────────────────
    trained_backbone = ssf_exp.model

    # ── Build the SGH experiment ───────────────────────────────────────────
    exp_params = dict(ssf_config["experiment"])
    sgh_exp = SpectralGatingExp(
        model=trained_backbone,
        sgh_hidden_dim=sgh_hidden_dim,
        alias_bands=alias_bands,
        lambda_high=lambda_high,
        loss_type=loss_type,
        learning_rate=learning_rate,
        stft_length=stft_length,
        stft_shift=stft_shift,
        **{k: v for k, v in exp_params.items()
           if k in {"cirm_comp_K", "cirm_comp_C", "n_cond_emb_dim",
                     "condition_enc_type", "cond_arange_params",
                     "reference_channel", "weight_decay", "loss_alpha",
                     "scheduler_type", "scheduler_params"}},
    )
    return sgh_exp


# ────────────────────────────────────────────────────────────────────────────
# Standalone training script
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary

    parser = argparse.ArgumentParser(description="Train Spectral Gating Head on frozen JNF-SSF")
    parser.add_argument("--ssf-checkpoint", default=None, help="Path to trained SSF .ckpt file")
    parser.add_argument("--config", required=True, help="Path to ssf_flac_config.yaml")
    parser.add_argument("--sgh-hidden-dim", type=int, default=256)
    parser.add_argument("--lambda-high", type=float, default=10.0)
    parser.add_argument("--alias-lo", type=float, default=2000.0, help="Alias band lower freq (Hz)")
    parser.add_argument("--alias-hi", type=float, default=2500.0, help="Alias band upper freq (Hz)")
    parser.add_argument("--loss-type", default="band_weighted",
                        choices=["band_weighted", "l1", "sisdr"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--gpus", type=int, nargs="*", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        ssf_config = yaml.safe_load(f)

    pl.seed_everything(ssf_config.get("seed", 0), workers=True)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    ssf_checkpoint = args.ssf_checkpoint or ssf_config.get("training", {}).get(
        "ssf_resume_ckpt"
    )
    if not ssf_checkpoint:
        ssf_checkpoint = ssf_config.get("training", {}).get("resume_ckpt")
    if not ssf_checkpoint:
        raise ValueError(
            "Missing SSF checkpoint: pass --ssf-checkpoint or set training.ssf_resume_ckpt in config."
        )
    if not os.path.isfile(ssf_checkpoint):
        raise FileNotFoundError(f"SSF checkpoint not found: {ssf_checkpoint}")

    # Data
    data_config = dict(ssf_config["data"])
    stft_length = data_config.get("stft_length_samples", 512)
    stft_shift = data_config.get("stft_shift_samples", 256)

    data_source = data_config.get("source", "hdf5").lower()
    if data_source == "flac":
        from data.flac_datamodule import FlacDataModule
        dm = FlacDataModule(**data_config)
    else:
        from data.datamodule import HDF5DataModule
        dm = HDF5DataModule(**data_config)

    # Experiment
    sgh_exp = build_sgh_experiment(
        ssf_config=ssf_config,
        ssf_checkpoint_path=ssf_checkpoint,
        stft_length=stft_length,
        stft_shift=stft_shift,
        sgh_hidden_dim=args.sgh_hidden_dim,
        alias_bands=[(args.alias_lo, args.alias_hi)],
        lambda_high=args.lambda_high,
        loss_type=args.loss_type,
        learning_rate=args.lr,
    )

    # Trainer
    log_dir = os.path.join(os.path.dirname(ssf_checkpoint), "..", "tb_logs")
    ckpt_dir = os.path.dirname(ssf_checkpoint)
    logger = pl_loggers.TensorBoardLogger(log_dir, name="SGH")
    training_cfg = ssf_config.get("training", {})
    if args.gpus:
        accelerator = "gpu"
        devices = args.gpus
    elif "accelerator" in training_cfg or "devices" in training_cfg:
        accelerator = training_cfg.get("accelerator", "auto")
        devices = training_cfg.get("devices", 1)
    elif "gpus" in training_cfg:
        gpus = training_cfg.get("gpus")
        if gpus:
            accelerator = "gpu"
            devices = gpus
        else:
            accelerator = "cpu"
            devices = 1
    elif torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1
    max_epochs = (
        args.max_epochs
        if args.max_epochs is not None
        else training_cfg.get("max_epochs", 50)
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[
            EarlyStopping(monitor="val/loss", patience=10, mode="min", verbose=True),
            ModelCheckpoint(monitor="val/loss", save_top_k=1, mode="min",
                            dirpath=ckpt_dir, filename="sgh-best"),
            ModelCheckpoint(save_top_k=-1, dirpath=ckpt_dir,
                            filename="sgh-{epoch:02d}"),
            ModelSummary(max_depth=2),
        ],
        gradient_clip_val=training_cfg.get("gradient_clip_val", 1.0),
        gradient_clip_algorithm=training_cfg.get("gradient_clip_algorithm", "norm"),
        strategy=training_cfg.get("strategy", "auto"),
        precision=training_cfg.get(
            "precision", "16-mixed" if accelerator == "gpu" else "32-true"
        ),
        accelerator=accelerator,
        devices=devices,
    )

    trainer.fit(sgh_exp, dm)
