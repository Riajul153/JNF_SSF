# JNF_SSF (PyTorch + ONNX)

This repository contains JNF/SSF model code, training scripts, ONNX export, and ONNX evaluation.

## What is included

- Full `src/` code needed to train and run JNF/SSF
- ONNX export script for SSF checkpoints
- ONNX evaluation script (SI-SDR, OSINR, PESQ, STOI)
- Pretrained checkpoints organized by condition

## Repository layout

```text
JNF_SSF_repo/
  src/
    config/
      jnf_flac_config.yaml
      ssf_flac_config.yaml
    data/
    models/
    scripts/
      train_jnf.py
      train_ssf.py
      train_jnf_ssf.py
      test_model.py
      test_flac_folder.py
      test_ssf.py
  checkpoints/
    Anechoic/
      JNF_SSF/
        ssf-best-sisdr.ckpt
    Reverberant/
      JNF_SSF/
        ssf-best-sisdr.ckpt
  export_jnf_ssf_onnx.py
  eval_jnf_onnx_metrics.py
  requirements.txt
```

## Environment setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For GPU ONNX inference:

```bash
pip install onnxruntime-gpu
```

## Configure dataset paths

Edit:

- `src/config/jnf_flac_config.yaml`
- `src/config/ssf_flac_config.yaml`

Set these keys to your dataset:

- `train_clean_dir`
- `train_noisy_dir`
- `val_clean_dir`
- `val_noisy_dir`
- `test_clean_dir`
- `test_noisy_dir`

## Train from scratch

### Train JNF

```bash
python src/scripts/train_jnf.py
```

### Train SSF

```bash
python src/scripts/train_ssf.py
```

### Joint pipeline script (optional)

```bash
python src/scripts/train_jnf_ssf.py
```

## PyTorch inference/evaluation on folder

```bash
python src/scripts/test_flac_folder.py \
  --model ssf \
  --ckpt checkpoints/Reverberant/JNF_SSF/ssf-best-sisdr.ckpt \
  --input_dir Dataset/Audio_Dataset/Test/Noisy \
  --clean_dir Dataset/Audio_Dataset/Test/Clean \
  --output_dir outputs/reverb_eval \
  --target-dir 90 \
  --device cuda
```

## Export checkpoint to ONNX

```bash
python export_jnf_ssf_onnx.py \
  --ckpt checkpoints/Reverberant/JNF_SSF/ssf-best-sisdr.ckpt \
  --config src/config/ssf_flac_config.yaml \
  --output models/jnf_reverb_dynamic.onnx \
  --dynamic-batch \
  --dynamic-time
```

## Evaluate ONNX model

```bash
python eval_jnf_onnx_metrics.py \
  --onnx_model models/jnf_reverb_dynamic.onnx \
  --test_dir Dataset/Audio_Dataset/Test \
  --output_dir eval_outputs/reverb \
  --prefer_cuda
```

Outputs:

- `metrics_progress.csv`
- `metrics_means.csv`
- `metrics_averages.txt`

## Notes

- JNF/SSF scripts are config-driven. Prefer editing YAML configs instead of hardcoding paths.
- For reproducible results, keep `fs`, STFT, and channel settings consistent between training and inference.
