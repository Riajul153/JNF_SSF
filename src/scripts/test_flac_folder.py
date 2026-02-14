import os
import sys
import argparse
import glob
import yaml
import torch
import soundfile as sf
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility as stoi

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.exp_jnf import JNFExp
from models.exp_ssf import SSFExp
from models.models import JNF_SSF


def load_experiment(config, ckpt_path, model_type, device):
    if model_type == "jnf":
        init_params = JNFExp.get_init_params(config)
        exp = JNFExp.load_from_checkpoint(ckpt_path, **init_params)
    elif model_type == "ssf":
        stft_length = config["data"]["stft_length_samples"]
        stft_shift = config["data"]["stft_shift_samples"]
        model = JNF_SSF(**config["network"])
        init_params = {
            "model": model,
            "stft_length": stft_length,
            "stft_shift": stft_shift,
            **config["experiment"],
        }
        if ckpt_path.lower().endswith(".pt"):
            exp = SSFExp(**init_params)
            state = torch.load(ckpt_path, map_location=device)
            missing, unexpected = exp.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"Warning: missing keys={len(missing)} unexpected keys={len(unexpected)}")
        else:
            exp = SSFExp.load_from_checkpoint(ckpt_path, **init_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    exp.eval()
    exp.to(device)
    return exp


def load_audio(path, fs, n_channels):
    audio, sr = sf.read(path, always_2d=True)
    if sr != fs:
        raise ValueError(f"Expected {fs} Hz, got {sr} Hz for {path}")
    audio = audio.T

    if audio.shape[0] < n_channels:
        pad = n_channels - audio.shape[0]
        audio = torch.from_numpy(audio)
        audio = torch.cat([audio, audio[-1:, :].repeat(pad, 1)], dim=0)
    elif audio.shape[0] > n_channels:
        audio = torch.from_numpy(audio[:n_channels, :])
    else:
        audio = torch.from_numpy(audio)

    audio = audio.float().unsqueeze(0)
    return audio


def infer_file(exp, audio_td, model_type, target_dir=None):
    device = exp.device
    audio_td = audio_td.to(device)

    noisy_stft = exp.get_stft_rep(audio_td)[0]
    stacked_noisy_stft = torch.concat(
        (torch.real(noisy_stft), torch.imag(noisy_stft)), dim=1
    )

    with torch.no_grad():
        if model_type == "ssf":
            if target_dir is None:
                raise ValueError("target_dir is required for SSF inference")
            target_dir = torch.tensor([target_dir], device=device, dtype=torch.long)
            mask_out = exp(stacked_noisy_stft, target_dir)
        else:
            mask_out = exp(stacked_noisy_stft)

        if exp.model.output_type == "IRM":
            speech_mask = mask_out
        elif exp.model.output_type == "CRM":
            speech_mask, _ = exp.get_complex_masks_from_stacked(mask_out)
        else:
            raise ValueError(f"Unsupported output_type: {exp.model.output_type}")

        if speech_mask.dim() == 4 and speech_mask.shape[1] == 1:
            speech_mask = speech_mask[:, 0, ...]

        est_clean_stft = noisy_stft[:, exp.reference_channel, ...] * speech_mask
        est_clean_td = exp.get_td_rep(est_clean_stft)[0]

    return est_clean_td.cpu().squeeze(0)


def compute_si_sdr(est, ref):
    est = est - torch.mean(est)
    ref = ref - torch.mean(ref)
    alpha = torch.dot(est, ref) / (torch.dot(ref, ref) + 1e-8)
    scaled_ref = alpha * ref
    num = torch.sum(scaled_ref ** 2)
    den = torch.sum((scaled_ref - est) ** 2) + 1e-8
    return 10 * torch.log10(num / den)


def compute_out_snr(est, ref):
    min_len = min(est.shape[-1], ref.shape[-1])
    est = est[:min_len]
    ref = ref[:min_len]
    scale = torch.dot(est, ref) / (torch.dot(ref, ref) + 1e-8)
    target = scale * ref
    error = est - target
    p_target = torch.mean(target ** 2)
    p_error = torch.mean(error ** 2) + 1e-8
    return 10 * torch.log10(p_target / p_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "ssf_flac_config.yaml"
        ),
    )
    parser.add_argument(
        "--ckpt",
        default=r"checkpoints/Anechoic/JNF_SSF/ssf-best-sisdr.ckpt",
    )
    parser.add_argument(
        "--input_dir",
        default=r"Dataset/Audio_Dataset/Test/Noisy",
    )
    parser.add_argument(
        "--clean_dir",
        default=r"Dataset/Audio_Dataset/Test/Clean",
        help="Optional clean folder to compute metrics",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("outputs_anechoic", "test_first_1000"),
        help="Base directory to save first N outputs",
    )
    parser.add_argument(
        "--best_dir",
        default=os.path.join("outputs", "best_30"),
        help="(Deprecated) Base directory to save best K results sorted by OSNR then STOI",
    )
    parser.add_argument(
        "--best_sisdr_dir",
        default=os.path.join("outputs_anechoic", "best_30_sisdr"),
        help="Base directory to save best K results sorted by SI-SDR",
    )
    parser.add_argument(
        "--best_stoi_dir",
        default=os.path.join("outputs_anechoic", "best_30_stoi"),
        help="Base directory to save best K results sorted by STOI",
    )
    parser.add_argument(
        "--save_limit",
        type=int,
        default=10,
        help="Save outputs for only the first N files in sorted order",
    )
    parser.add_argument(
        "--best_k",
        type=int,
        default=30,
        help="Save best K results sorted by OSNR and STOI",
    )
    parser.add_argument("--model", choices=["jnf", "ssf"], default="ssf")
    parser.add_argument("--target-dir", type=int, default=0, help="Required for SSF")
    parser.add_argument(
        "--device",
        default=None,
        help="Force device: cpu or cuda. Defaults to cuda if available.",
    )
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    fs = config["data"].get("fs", 16000)
    n_channels = config["data"]["n_channels"]

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    exp = load_experiment(config, args.ckpt, args.model, device)

    patterns = [os.path.join(args.input_dir, "*.flac"), os.path.join(args.input_dir, "*.wav")]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files.sort()

    if not files:
        raise FileNotFoundError(f"No audio files found in {args.input_dir}")

    output_enh_dir = os.path.join(args.output_dir, "enhanced")
    output_noisy_dir = os.path.join(args.output_dir, "noisy")
    output_clean_dir = os.path.join(args.output_dir, "clean")
    os.makedirs(output_enh_dir, exist_ok=True)
    os.makedirs(output_noisy_dir, exist_ok=True)
    os.makedirs(output_clean_dir, exist_ok=True)

    si_sdr_values = []
    stoi_values = []
    osnr_values = []
    scored_results = []
    save_set = set(files[: min(args.save_limit, len(files))])

    for path in files:
        audio_td = load_audio(path, fs, n_channels)
        enhanced = infer_file(exp, audio_td, args.model, args.target_dir)
        base = os.path.splitext(os.path.basename(path))[0]
        clean_td = None
        if args.clean_dir is not None:
            clean_path = os.path.join(args.clean_dir, os.path.basename(path))
            if not os.path.isfile(clean_path):
                print(f"Missing clean file for metrics: {clean_path}")
            else:
                clean_td = load_audio(clean_path, fs, n_channels).squeeze(0)
                enhanced_td = enhanced
                min_len = min(clean_td.shape[-1], enhanced_td.shape[-1])
                clean_td = clean_td[:, :min_len]
                enhanced_td = enhanced_td[:min_len]
                clean_ref = clean_td[0]
                si_sdr = compute_si_sdr(enhanced_td, clean_ref)
                si_sdr_values.append(si_sdr.item())
                stoi_score = stoi(
                    enhanced_td.unsqueeze(0), clean_ref.unsqueeze(0), fs
                ).item()
                stoi_values.append(stoi_score)
                osnr = compute_out_snr(enhanced_td, clean_ref).item()
                osnr_values.append(osnr)
                scored_results.append(
                    {
                        "path": path,
                        "si_sdr": si_sdr.item(),
                        "stoi": stoi_score,
                        "osnr": osnr,
                    }
                )
                print(
                    f"SI-SDR ({base}): {si_sdr.item():.3f} dB | STOI: {stoi_score:.3f} | OSNR: {osnr:.3f} dB"
                )

        if path in save_set:
            out_path = os.path.join(output_enh_dir, f"{base}.wav")
            sf.write(out_path, enhanced.numpy(), fs)
            noisy_out_path = os.path.join(output_noisy_dir, f"{base}.wav")
            sf.write(noisy_out_path, audio_td.squeeze(0)[0].numpy(), fs)
            if clean_td is not None:
                clean_out_path = os.path.join(output_clean_dir, f"{base}.wav")
                sf.write(clean_out_path, clean_td[0].numpy(), fs)
            print(f"Saved: {out_path}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if si_sdr_values:
        avg_si_sdr = sum(si_sdr_values) / len(si_sdr_values)
        avg_stoi = sum(stoi_values) / len(stoi_values)
        avg_osnr = sum(osnr_values) / len(osnr_values)
        print(
            f"Average SI-SDR: {avg_si_sdr:.3f} dB | Average STOI: {avg_stoi:.3f} | Average OSNR: {avg_osnr:.3f} dB"
        )

        if scored_results:
            def save_top_k(ranked, base_dir, tag):
                best_enh_dir = os.path.join(base_dir, "enhanced")
                best_noisy_dir = os.path.join(base_dir, "noisy")
                best_clean_dir = os.path.join(base_dir, "clean")
                os.makedirs(best_enh_dir, exist_ok=True)
                os.makedirs(best_noisy_dir, exist_ok=True)
                os.makedirs(best_clean_dir, exist_ok=True)

                top_k = ranked[: max(args.best_k, 0)]
                out_csv = os.path.join(base_dir, f"best_results_{tag}.csv")
                with open(out_csv, "w", encoding="ascii") as f:
                    f.write("file,si_sdr,stoi,osnr\n")
                    for r in top_k:
                        f.write(
                            f"{os.path.basename(r['path'])},{r['si_sdr']:.6f},{r['stoi']:.6f},{r['osnr']:.6f}\n"
                        )

                for r in top_k:
                    path = r["path"]
                    audio_td = load_audio(path, fs, n_channels)
                    enhanced = infer_file(exp, audio_td, args.model, args.target_dir)
                    base = os.path.splitext(os.path.basename(path))[0]
                    out_path = os.path.join(best_enh_dir, f"{base}.wav")
                    sf.write(out_path, enhanced.numpy(), fs)
                    noisy_out_path = os.path.join(best_noisy_dir, f"{base}.wav")
                    sf.write(noisy_out_path, audio_td.squeeze(0)[0].numpy(), fs)
                    if args.clean_dir is not None:
                        clean_path = os.path.join(args.clean_dir, os.path.basename(path))
                        if os.path.isfile(clean_path):
                            clean_td = load_audio(clean_path, fs, n_channels).squeeze(0)
                            clean_out_path = os.path.join(best_clean_dir, f"{base}.wav")
                            sf.write(clean_out_path, clean_td[0].numpy(), fs)

                print(f"Saved top-{len(top_k)} results to: {base_dir}")

            ranked_sisdr = sorted(
                scored_results,
                key=lambda r: r["si_sdr"],
                reverse=True,
            )
            ranked_stoi = sorted(
                scored_results,
                key=lambda r: r["stoi"],
                reverse=True,
            )

            save_top_k(ranked_sisdr, args.best_sisdr_dir, "sisdr")
            save_top_k(ranked_stoi, args.best_stoi_dir, "stoi")


if __name__ == "__main__":
    main()
