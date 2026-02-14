#!/usr/bin/env python3
"""
Evaluate JNF-SSF ONNX model on paired Clean/Noisy test set.

Metrics:
  - SI-SDR
  - OSINR
  - PESQ
  - STOI

Outputs:
  - metrics_progress.csv
  - metrics_averages.txt
  - metrics_means.csv
Optionally:
  - top-K submission bundles with target/interference/processed audio
  - TaskX_*.mat from rank-1 sample
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from scipy.signal import resample_poly
from tqdm import tqdm

try:
    from pesq import pesq as pesq_fn  # type: ignore
except Exception:
    pesq_fn = None

try:
    from pystoi import stoi as stoi_fn  # type: ignore
except Exception:
    stoi_fn = None

try:
    from scipy.io import savemat  # type: ignore
except Exception:
    savemat = None


CSV_HEADER = ["file", "si_sdr", "osinr", "pesq", "stoi", "status", "error"]


@dataclass
class JNFOnnxInfo:
    x_name: str
    target_name: str
    output_name: str
    feat_channels: int
    f_bins: int
    providers: List[str]


def si_sdr(reference: np.ndarray, estimation: np.ndarray, eps: float = 1e-8) -> float:
    ref = reference.astype(np.float64, copy=False)
    est = estimation.astype(np.float64, copy=False)
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    ref_energy = float(np.dot(ref, ref) + eps)
    scale = float(np.dot(est, ref) / ref_energy)
    s_target = scale * ref
    e_noise = est - s_target
    num = float(np.dot(s_target, s_target) + eps)
    den = float(np.dot(e_noise, e_noise) + eps)
    return 10.0 * math.log10(num / den)


def osinr(reference: np.ndarray, estimation: np.ndarray, eps: float = 1e-8) -> float:
    ref = reference.astype(np.float64, copy=False)
    est = estimation.astype(np.float64, copy=False)
    scale = float(np.dot(est, ref) / (np.dot(ref, ref) + eps))
    target = scale * ref
    error = est - target
    p_target = float(np.mean(target**2))
    p_error = float(np.mean(error**2) + eps)
    return 10.0 * math.log10(p_target / p_error)


def compute_pesq(reference: np.ndarray, estimation: np.ndarray, sr: int) -> float:
    if pesq_fn is None:
        return float("nan")
    if sr not in (8000, 16000):
        return float("nan")
    mode = "wb" if sr == 16000 else "nb"
    ref = np.clip(reference.astype(np.float32, copy=False), -1.0, 1.0)
    est = np.clip(estimation.astype(np.float32, copy=False), -1.0, 1.0)
    n = min(ref.shape[0], est.shape[0])
    ref = ref[:n]
    est = est[:n]
    if n < int(0.3 * sr):
        pad = int(0.3 * sr) - n
        ref = np.pad(ref, (0, pad))
        est = np.pad(est, (0, pad))
    try:
        return float(pesq_fn(sr, ref, est, mode))
    except Exception:
        return float("nan")


def compute_stoi(reference: np.ndarray, estimation: np.ndarray, sr: int) -> float:
    if stoi_fn is None:
        return float("nan")
    try:
        return float(stoi_fn(reference.astype(np.float32), estimation.astype(np.float32), sr, extended=False))
    except Exception:
        return float("nan")


def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    wav_tc, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if wav_tc.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    return wav_tc, int(sr)


def resample_1d(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = math.gcd(int(sr_in), int(sr_out))
    up = int(sr_out) // g
    down = int(sr_in) // g
    y = resample_poly(x, up=up, down=down, axis=-1)
    return y.astype(np.float32, copy=False)


def enforce_channels(wav_tc: np.ndarray, n_channels: int) -> np.ndarray:
    c = wav_tc.shape[1]
    if c < n_channels:
        rep = np.repeat(wav_tc[:, c - 1 : c], n_channels - c, axis=1)
        wav_tc = np.concatenate([wav_tc, rep], axis=1)
    elif c > n_channels:
        wav_tc = wav_tc[:, :n_channels]
    return wav_tc


def align_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(a.shape[0], b.shape[0])
    return a[:n], b[:n]


def collect_pairs(clean_dir: Path, noisy_dir: Path) -> List[Tuple[Path, Path]]:
    exts = {".wav", ".flac", ".ogg"}
    clean_map: Dict[str, Path] = {}
    for p in clean_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            clean_map[p.name] = p

    pairs: List[Tuple[Path, Path]] = []
    for p in noisy_dir.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in exts):
            continue
        c = clean_map.get(p.name)
        if c is not None:
            pairs.append((c, p))
    pairs.sort(key=lambda x: x[1].name)
    return pairs


def parse_jnf_info(sess: ort.InferenceSession) -> JNFOnnxInfo:
    inputs = list(sess.get_inputs())
    if len(inputs) < 2:
        raise ValueError("Expected at least two ONNX inputs: x and target_dir")

    x_in = None
    for i in inputs:
        if len(i.shape) == 4:
            x_in = i
            break
    if x_in is None:
        raise ValueError("Could not find rank-4 input x [B,2C,F,T]")

    target_in = None
    for i in inputs:
        if i.name != x_in.name:
            target_in = i
            break
    if target_in is None:
        raise ValueError("Could not find target direction input")

    feat = x_in.shape[1]
    f_bins = x_in.shape[2]
    if not isinstance(feat, int) or feat <= 0:
        raise ValueError(f"Expected static feature channels in ONNX input shape, got {x_in.shape}")
    if not isinstance(f_bins, int) or f_bins <= 0:
        raise ValueError(f"Expected static frequency bins in ONNX input shape, got {x_in.shape}")

    return JNFOnnxInfo(
        x_name=x_in.name,
        target_name=target_in.name,
        output_name=sess.get_outputs()[0].name,
        feat_channels=int(feat),
        f_bins=int(f_bins),
        providers=sess.get_providers(),
    )


def create_session(model_path: Path, prefer_cuda: bool) -> Tuple[ort.InferenceSession, JNFOnnxInfo]:
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.intra_op_num_threads = max(1, min(16, os.cpu_count() or 8))
    sess_opts.inter_op_num_threads = 1
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if prefer_cuda else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)
    info = parse_jnf_info(sess)
    return sess, info


def run_jnf_onnx_enhancement(
    sess: ort.InferenceSession,
    info: JNFOnnxInfo,
    noisy_path: Path,
    sample_rate: int,
    n_fft: int,
    hop: int,
    target_dir_idx: int,
) -> np.ndarray:
    wav_tc, sr = read_audio(noisy_path)
    if sr != sample_rate:
        wav_tc = np.stack([resample_1d(wav_tc[:, c], sr, sample_rate) for c in range(wav_tc.shape[1])], axis=1)
    n_mics = info.feat_channels // 2
    wav_tc = enforce_channels(wav_tc, n_mics).astype(np.float32, copy=False)

    wav_ct = torch.from_numpy(wav_tc.T.copy())  # [C, T]
    win = torch.sqrt(torch.hann_window(n_fft, dtype=torch.float32))
    noisy_stft = torch.stft(
        wav_ct,
        n_fft=n_fft,
        hop_length=hop,
        window=win,
        center=True,
        onesided=True,
        return_complex=True,
    )  # [C, F, TT]
    if noisy_stft.shape[1] != info.f_bins:
        raise ValueError(f"Model expects F={info.f_bins}, STFT produced F={noisy_stft.shape[1]}")

    noisy_stft = noisy_stft.unsqueeze(0)  # [1, C, F, TT]
    x = torch.cat([noisy_stft.real, noisy_stft.imag], dim=1).numpy().astype(np.float32)  # [1, 2C, F, TT]
    target = np.array([int(target_dir_idx)], dtype=np.int64)

    y_out = sess.run([info.output_name], {info.x_name: x, info.target_name: target})[0]
    if y_out.ndim != 4:
        raise ValueError(f"Unexpected ONNX output shape: {y_out.shape}")
    if y_out.shape[1] == 1:
        speech_mask = y_out[:, 0, :, :].astype(np.complex64)
    elif y_out.shape[1] >= 2:
        speech_mask = y_out[:, 0, :, :] + 1j * y_out[:, 1, :, :]
    else:
        raise ValueError(f"Unexpected ONNX output channel shape: {y_out.shape}")

    est_clean_stft = noisy_stft[:, 0, :, :].numpy() * speech_mask
    est_td = torch.istft(
        torch.from_numpy(est_clean_stft[0]),
        n_fft=n_fft,
        hop_length=hop,
        window=win,
        center=True,
        onesided=True,
        length=int(wav_tc.shape[0]),
        return_complex=False,
    ).numpy()
    return est_td.astype(np.float32, copy=False)


def ensure_metrics_csv(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)
        return
    with path.open("r", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    if header != CSV_HEADER:
        raise ValueError(f"CSV header mismatch at {path}. Expected {CSV_HEADER}, found {header}")


def append_metrics_row(path: Path, row: Sequence[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def load_done_set(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            status = (row.get("status") or "").strip().upper()
            name = (row.get("file") or "").strip()
            if name and status == "OK":
                done.add(name)
    return done


def _parse_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _nan_stats(arr: np.ndarray) -> Tuple[int, float, float]:
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return 0, float("nan"), float("nan")
    return int(valid.size), float(np.mean(valid)), float(np.std(valid))


def summarize_metrics(csv_path: Path, txt_path: Path, mean_csv_path: Path) -> None:
    rows: List[dict] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("status") or "").strip().upper() == "OK":
                rows.append(row)

    si = np.array([_parse_float(r.get("si_sdr", "nan")) for r in rows], dtype=np.float64)
    osn = np.array([_parse_float(r.get("osinr", "nan")) for r in rows], dtype=np.float64)
    pe = np.array([_parse_float(r.get("pesq", "nan")) for r in rows], dtype=np.float64)
    st = np.array([_parse_float(r.get("stoi", "nan")) for r in rows], dtype=np.float64)

    si_n, si_mean, si_std = _nan_stats(si)
    os_n, os_mean, os_std = _nan_stats(osn)
    pe_n, pe_mean, pe_std = _nan_stats(pe)
    st_n, st_mean, st_std = _nan_stats(st)

    txt_lines = [
        "JNF ONNX Evaluation Summary",
        f"Rows considered (status=OK): {len(rows)}",
        "",
        f"SI_SDR  count={si_n} mean={si_mean:.6f} std={si_std:.6f}" if si_n else "SI_SDR  count=0 mean=NaN std=NaN",
        f"OSINR   count={os_n} mean={os_mean:.6f} std={os_std:.6f}" if os_n else "OSINR   count=0 mean=NaN std=NaN",
        f"PESQ    count={pe_n} mean={pe_mean:.6f} std={pe_std:.6f}" if pe_n else "PESQ    count=0 mean=NaN std=NaN",
        f"STOI    count={st_n} mean={st_mean:.6f} std={st_std:.6f}" if st_n else "STOI    count=0 mean=NaN std=NaN",
    ]
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    mean_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with mean_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rows_ok",
                "si_sdr_mean",
                "osinr_mean",
                "pesq_mean",
                "stoi_mean",
                "si_sdr_count",
                "osinr_count",
                "pesq_count",
                "stoi_count",
            ]
        )
        w.writerow(
            [
                len(rows),
                f"{si_mean:.6f}" if np.isfinite(si_mean) else "NaN",
                f"{os_mean:.6f}" if np.isfinite(os_mean) else "NaN",
                f"{pe_mean:.6f}" if np.isfinite(pe_mean) else "NaN",
                f"{st_mean:.6f}" if np.isfinite(st_mean) else "NaN",
                si_n,
                os_n,
                pe_n,
                st_n,
            ]
        )


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    out = []
    for ch in stem:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _save_topk(
    sess: ort.InferenceSession,
    info: JNFOnnxInfo,
    pairs_map: Dict[str, Tuple[Path, Path]],
    csv_path: Path,
    sample_rate: int,
    n_fft: int,
    hop: int,
    target_dir_idx: int,
    rank_metric: str,
    top_k: int,
    topk_dir: Path,
    submission_mat_name: str,
) -> None:
    rows: List[dict] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("status") or "").strip().upper() != "OK":
                continue
            v = _parse_float(row.get(rank_metric, "nan"))
            if not np.isfinite(v):
                continue
            row["_rank_value"] = v
            rows.append(row)

    if not rows:
        print("Top-K skipped: no valid metric rows.")
        return

    rows.sort(key=lambda r: float(r["_rank_value"]), reverse=True)
    rows = rows[: max(1, int(top_k))]
    topk_dir.mkdir(parents=True, exist_ok=True)
    out_rows: List[Dict] = []
    rank1_payload: Optional[Dict] = None

    for rank, row in enumerate(rows, start=1):
        name = (row.get("file") or "").strip()
        pair = pairs_map.get(name)
        if pair is None:
            print(f"[top {rank}] missing pair in map: {name}")
            continue
        clean_path, noisy_path = pair

        enhanced = run_jnf_onnx_enhancement(
            sess=sess,
            info=info,
            noisy_path=noisy_path,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop=hop,
            target_dir_idx=target_dir_idx,
        )
        clean_tc, clean_sr = read_audio(clean_path)
        noisy_tc, noisy_sr = read_audio(noisy_path)
        clean_mono = clean_tc.mean(axis=1).astype(np.float32, copy=False)
        noisy_mono = noisy_tc.mean(axis=1).astype(np.float32, copy=False)
        if clean_sr != sample_rate:
            clean_mono = resample_1d(clean_mono, clean_sr, sample_rate)
        if noisy_sr != sample_rate:
            noisy_mono = resample_1d(noisy_mono, noisy_sr, sample_rate)
        est, ref = align_pair(enhanced, clean_mono)
        est, noisy_aligned = align_pair(est, noisy_mono)

        si_val = si_sdr(ref, est)
        os_val = osinr(ref, est)
        pe_val = compute_pesq(ref, est, sample_rate)
        st_val = compute_stoi(ref, est, sample_rate)

        bundle_dir = topk_dir / f"rank_{rank:02d}_{_safe_stem(name)}"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(bundle_dir / "target_signal.wav"), ref, sample_rate, subtype="FLOAT")
        sf.write(str(bundle_dir / "interference_signal1.wav"), noisy_aligned, sample_rate, subtype="FLOAT")
        sf.write(str(bundle_dir / "processed_signal.wav"), est, sample_rate, subtype="FLOAT")

        meta = {
            "rank": rank,
            "file": name,
            "rank_metric": rank_metric,
            "rank_metric_value_from_log": float(row["_rank_value"]),
            "si_sdr": float(si_val),
            "osinr": float(os_val),
            "pesq": float(pe_val),
            "stoi": float(st_val),
        }
        (bundle_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        out_rows.append(
            {
                "rank": rank,
                "file": name,
                "si_sdr": f"{si_val:.6f}",
                "osinr": f"{os_val:.6f}",
                "pesq": f"{pe_val:.6f}" if np.isfinite(pe_val) else "NaN",
                "stoi": f"{st_val:.6f}" if np.isfinite(st_val) else "NaN",
                "bundle_dir": str(bundle_dir),
            }
        )

        if rank == 1:
            rank1_payload = {
                "target_signal": ref.astype(np.float32, copy=False),
                "interference_signal": noisy_aligned.astype(np.float32, copy=False),
                "mixture_signal": noisy_aligned.astype(np.float32, copy=False),
                "processed_signal": est.astype(np.float32, copy=False),
                "metrics": {
                    "OSINR": float(os_val),
                    "PESQ": float(pe_val),
                    "STOI": float(st_val),
                    "SI_SDR": float(si_val),
                },
                "params": {
                    "sample_rate": int(sample_rate),
                    "source_file": name,
                    "rank_metric": rank_metric,
                    "note": "interference_signal and mixture_signal are taken from the noisy mixture due dataset format",
                },
            }

    topk_csv = topk_dir / "topk_results.csv"
    with topk_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "file", "si_sdr", "osinr", "pesq", "stoi", "bundle_dir"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"Saved top-K summary: {topk_csv}")

    if submission_mat_name.strip():
        if savemat is None:
            print("scipy.io.savemat unavailable; skipping .mat creation.")
        elif rank1_payload is None:
            print("No rank-1 payload available; skipping .mat creation.")
        else:
            mat_path = topk_dir / submission_mat_name
            payload = dict(rank1_payload)
            payload["rir_data"] = np.zeros((1, 1), dtype=np.float32)
            savemat(str(mat_path), payload, do_compression=True)
            print(f"Saved submission mat: {mat_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate JNF ONNX model on paired test set.")
    p.add_argument("--onnx_model", type=str, required=True)
    p.add_argument("--test_dir", type=str, default=r"Dataset/Audio_Dataset/Test")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--n_fft", type=int, default=512)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--target_dir_idx", type=int, default=90)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--prefer_cuda", action="store_true", default=True)
    p.add_argument("--cpu_only", dest="prefer_cuda", action="store_false")
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--rank_metric", type=str, default="si_sdr", choices=["si_sdr", "osinr", "pesq", "stoi"])
    p.add_argument("--save_top_k", type=int, default=0, help="If >0, save top-K audio bundles")
    p.add_argument("--topk_dir", type=str, default="", help="Top-K output dir (default: output_dir/top_5)")
    p.add_argument("--submission_mat_name", type=str, default="", help="Optional TaskX_*.mat filename")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.onnx_model).expanduser().resolve()
    test_dir = Path(args.test_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    clean_dir = test_dir / "Clean"
    noisy_dir = test_dir / "Noisy"

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if not clean_dir.exists() or not noisy_dir.exists():
        raise FileNotFoundError(f"Missing Clean/Noisy dirs under: {test_dir}")

    sess, info = create_session(model_path, prefer_cuda=bool(args.prefer_cuda))
    print(f"ONNX providers in use: {info.providers}")
    print(f"ONNX I/O: x='{info.x_name}', target='{info.target_name}', output='{info.output_name}', feat={info.feat_channels}, F={info.f_bins}")

    pairs = collect_pairs(clean_dir, noisy_dir)
    if args.max_files and args.max_files > 0:
        pairs = pairs[: int(args.max_files)]
    if not pairs:
        raise RuntimeError("No Clean/Noisy matching file pairs found.")
    pair_map = {n.name: (c, n) for c, n in pairs}

    metrics_csv = out_dir / "metrics_progress.csv"
    summary_txt = out_dir / "metrics_averages.txt"
    means_csv = out_dir / "metrics_means.csv"
    ensure_metrics_csv(metrics_csv)

    done = load_done_set(metrics_csv) if args.resume else set()
    to_process = [(c, n) for c, n in pairs if n.name not in done]
    print(f"Total pairs: {len(pairs)} | Already done: {len(done)} | Remaining: {len(to_process)}")

    for clean_path, noisy_path in tqdm(to_process, desc="Evaluating", unit="file"):
        try:
            enhanced = run_jnf_onnx_enhancement(
                sess=sess,
                info=info,
                noisy_path=noisy_path,
                sample_rate=int(args.sample_rate),
                n_fft=int(args.n_fft),
                hop=int(args.hop),
                target_dir_idx=int(args.target_dir_idx),
            )

            clean_tc, clean_sr = read_audio(clean_path)
            clean_mono = clean_tc.mean(axis=1).astype(np.float32, copy=False)
            if clean_sr != int(args.sample_rate):
                clean_mono = resample_1d(clean_mono, clean_sr, int(args.sample_rate))

            est, ref = align_pair(enhanced.astype(np.float32, copy=False), clean_mono)
            si_val = si_sdr(ref, est)
            os_val = osinr(ref, est)
            pe_val = compute_pesq(ref, est, int(args.sample_rate))
            st_val = compute_stoi(ref, est, int(args.sample_rate))

            append_metrics_row(
                metrics_csv,
                [
                    noisy_path.name,
                    f"{si_val:.6f}",
                    f"{os_val:.6f}",
                    f"{pe_val:.6f}" if np.isfinite(pe_val) else "NaN",
                    f"{st_val:.6f}" if np.isfinite(st_val) else "NaN",
                    "OK",
                    "",
                ],
            )
        except Exception as exc:
            append_metrics_row(metrics_csv, [noisy_path.name, "NaN", "NaN", "NaN", "NaN", "FAILED", str(exc)])

    summarize_metrics(metrics_csv, summary_txt, means_csv)
    print(f"Saved per-file metrics: {metrics_csv}")
    print(f"Saved text summary:     {summary_txt}")
    print(f"Saved means CSV:        {means_csv}")

    if int(args.save_top_k) > 0:
        topk_dir = Path(args.topk_dir).expanduser().resolve() if args.topk_dir else (out_dir / f"top_{int(args.save_top_k)}")
        _save_topk(
            sess=sess,
            info=info,
            pairs_map=pair_map,
            csv_path=metrics_csv,
            sample_rate=int(args.sample_rate),
            n_fft=int(args.n_fft),
            hop=int(args.hop),
            target_dir_idx=int(args.target_dir_idx),
            rank_metric=args.rank_metric,
            top_k=int(args.save_top_k),
            topk_dir=topk_dir,
            submission_mat_name=args.submission_mat_name,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
