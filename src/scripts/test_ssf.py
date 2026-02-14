import os
import sys
import argparse
import glob
import yaml
import torch
import soundfile as sf
import random
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
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "ssf_flac_config.yaml"))
    parser.add_argument("--ckpt", default=r"checkpoints/Reverberant/JNF_SSF/ssf-best-sisdr.ckpt")
    parser.add_argument(
        "--input_dir",
        default=r"Dataset/Audio_Dataset/Test/Noisy",
    )
    parser.add_argument(
        "--clean_dir",
        default=r"Dataset/Audio_Dataset/Test/Clean",
        help="Optional clean folder to compute metrics",
    )
    parser.add_argument("--output_dir", default="enhanced_test")
    parser.add_argument(
        "--save_dir",
        default="enhanced_samples_250",
        help="Directory to save random sample outputs",
    )
    parser.add_argument(
        "--random_save_count",
        type=int,
        default=5,
        help="Save outputs for only N random files",
    )
    parser.add_argument(
        "--random_eval_count",
        type=int,
        default=250,
        help="If >0, evaluate metrics on only N random files",
    )
    parser.add_argument("--model", choices=["jnf", "ssf"], default="ssf")
    parser.add_argument("--target-dir", type=int, default=90, help="Required for SSF")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    fs = config["data"].get("fs", 16000)
    n_channels = config["data"]["n_channels"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp = load_experiment(config, args.ckpt, args.model, device)

    patterns = [os.path.join(args.input_dir, "*.flac"), os.path.join(args.input_dir, "*.wav")]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files.sort()

    if not files:
        raise FileNotFoundError(f"No audio files found in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    si_sdr_values = []
    stoi_values = []
    osnr_values = []
    if args.random_eval_count and args.random_eval_count > 0:
        rnd = random.Random(1)
        rnd.shuffle(files)
        files = files[: min(args.random_eval_count, len(files))]
    save_set = set()
    if args.random_save_count and args.random_save_count > 0:
        rnd = random.Random(0)
        shuffled = list(files)
        rnd.shuffle(shuffled)
        save_set = set(shuffled[: min(args.random_save_count, len(shuffled))])

    for path in files:
        audio_td = load_audio(path, fs, n_channels)
        enhanced = infer_file(exp, audio_td, args.model, args.target_dir)
        base = os.path.splitext(os.path.basename(path))[0]
        if not save_set or path in save_set:
            out_path = os.path.join(args.save_dir, f"{base}_enhanced.wav")
            sf.write(out_path, enhanced.numpy(), fs)
            noisy_out_path = os.path.join(args.save_dir, f"{base}_noisy.wav")
            sf.write(noisy_out_path, audio_td.squeeze(0)[0].numpy(), fs)
            print(f"Saved: {out_path}")
        if args.clean_dir is not None:
            clean_path = os.path.join(args.clean_dir, os.path.basename(path))
            if not os.path.isfile(clean_path):
                print(f"Missing clean file for metrics: {clean_path}")
                continue
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
            print(
                f"SI-SDR ({base}): {si_sdr.item():.3f} dB | STOI: {stoi_score:.3f} | OSNR: {osnr:.3f} dB"
            )

    if si_sdr_values:
        avg_si_sdr = sum(si_sdr_values) / len(si_sdr_values)
        avg_stoi = sum(stoi_values) / len(stoi_values)
        avg_osnr = sum(osnr_values) / len(osnr_values)
        print(
            f"Average SI-SDR: {avg_si_sdr:.3f} dB | Average STOI: {avg_stoi:.3f} | Average OSNR: {avg_osnr:.3f} dB"
        )


if __name__ == "__main__":
    main()
