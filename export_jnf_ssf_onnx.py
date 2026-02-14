#!/usr/bin/env python3
"""
Export JNF-SSF checkpoint to ONNX.

This exports a wrapper with signature:
  forward(x, target_dir_idx) -> mask_out

where:
  x:              [B, 2*C, F, T] float32
  target_dir_idx: [B] int64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import yaml


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")
    if any(k.startswith("model.") for k in state.keys()):
        state = {k.replace("model.", "", 1): v for k, v in state.items()}
    return state


def _import_jnf_ssf(repo_root: Path):
    src_dir = repo_root / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"JNF-SSF src directory not found: {src_dir}")
    sys.path.insert(0, str(src_dir))
    from models.models import JNF_SSF  # pylint: disable=import-error

    return JNF_SSF


class SSFOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, n_cond_emb_dim: int):
        super().__init__()
        self.model = model
        self.n_cond_emb_dim = int(n_cond_emb_dim)

    def forward(self, x: torch.Tensor, target_dir_idx: torch.Tensor) -> torch.Tensor:
        if target_dir_idx.dtype != torch.long:
            target_dir_idx = target_dir_idx.to(torch.long)
        if target_dir_idx.ndim > 1:
            target_dir_idx = target_dir_idx.reshape(target_dir_idx.shape[0])
        target_dir_oh = F.one_hot(target_dir_idx, self.n_cond_emb_dim).to(dtype=x.dtype)
        return self.model(x, target_dir_oh, device="cpu")


def _build_model(repo_root: Path, ckpt: Path, cfg: Path) -> Tuple[torch.nn.Module, int]:
    JNF_SSF = _import_jnf_ssf(repo_root)
    config = _load_config(cfg)
    if "network" not in config or "experiment" not in config:
        raise ValueError(f"Missing required sections in config: {cfg}")

    base = JNF_SSF(**config["network"])
    state = _load_state_dict(ckpt)
    missing, unexpected = base.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"Warning: missing keys={len(missing)} unexpected keys={len(unexpected)}")
    base.eval()

    n_cond = int(config["experiment"]["n_cond_emb_dim"])
    return base, n_cond


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export JNF-SSF checkpoint to ONNX")
    p.add_argument("--ckpt", type=str, required=True, help="Path to ssf-best-sisdr.ckpt")
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to JNF-SSF YAML config",
    )
    p.add_argument("--output", type=str, required=True, help="Output ONNX path")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset")
    p.add_argument("--batch", type=int, default=1, help="Dummy batch for export")
    p.add_argument("--freq-bins", type=int, default=257, help="Dummy F bins")
    p.add_argument("--time-frames", type=int, default=188, help="Dummy T frames")
    p.add_argument("--channels", type=int, default=2, help="Number of microphones/channels")
    p.add_argument("--target-idx", type=int, default=90, help="Dummy target direction index")
    p.add_argument("--dynamic-batch", action="store_true", help="Enable dynamic batch axis")
    p.add_argument("--dynamic-time", action="store_true", help="Enable dynamic time axis")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    ckpt = Path(args.ckpt).expanduser().resolve()
    if str(args.config).strip():
        cfg = Path(args.config).expanduser().resolve()
    else:
        cfg = (repo_root / "src" / "config" / "ssf_flac_config.yaml").resolve()
    out_path = Path(args.output).expanduser().resolve()

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not cfg.exists():
        raise FileNotFoundError(f"Config not found: {cfg}")

    base, n_cond = _build_model(repo_root, ckpt, cfg)
    wrapper = SSFOnnxWrapper(base, n_cond).eval()

    in_ch = int(args.channels) * 2
    x = torch.randn(int(args.batch), in_ch, int(args.freq_bins), int(args.time_frames), dtype=torch.float32)
    target = torch.full((int(args.batch),), int(args.target_idx), dtype=torch.long)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if args.dynamic_batch or args.dynamic_time:
        dynamic_axes = {
            "x": {},
            "target_dir": {},
            "mask_out": {},
        }
        if args.dynamic_batch:
            dynamic_axes["x"][0] = "batch"
            dynamic_axes["target_dir"][0] = "batch"
            dynamic_axes["mask_out"][0] = "batch"
        if args.dynamic_time:
            dynamic_axes["x"][3] = "time"
            dynamic_axes["mask_out"][3] = "time"

    print(f"Exporting JNF-SSF ONNX:\n  ckpt:   {ckpt}\n  config: {cfg}\n  output: {out_path}")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (x, target),
            str(out_path),
            input_names=["x", "target_dir"],
            output_names=["mask_out"],
            dynamic_axes=dynamic_axes,
            opset_version=int(args.opset),
            export_params=True,
            do_constant_folding=True,
            dynamo=False,
        )

    print(f"Saved ONNX: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
