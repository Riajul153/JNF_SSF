# JNF\_SSF (PyTorch + ONNX)

This repository contains a PyTorch implementation of **Spatially-Selective Deep Non-linear Filters (SSF)** and their **Joint Nonlinear Filter (JNF)** variant, with utilities for training, evaluation, and ONNX export for edge deployment.

The codebase was developed by **SP CUP 2026 MEMBER INFORMATION: Team Nyquist** for **IEEE Signal Processing Cup 2026 (Phase-2)**, targeting a challenging two-microphone smartphone setting where **spatial aliasing** and **reverberation** make classical steering-based beamformers brittle.

---

## Reference paper (please cite)

This implementation is based on the SSF/JNF framework proposed in:

- **K. Tesch and T. Gerkmann**, *“Multi-channel Speech Separation Using Spatially-Selective Deep Non-linear Filters”*, arXiv:2304.12023, 2023.  
  arXiv: https://arxiv.org/abs/2304.12023

If you use this repository, please cite the original paper:

```bibtex
@misc{tesch2023multichannel,
  title={Multi-channel Speech Separation Using Spatially Selective Deep Non-linear Filters},
  author={Tesch, Kristina and Gerkmann, Timo},
  year={2023},
  eprint={2304.12023},
  archivePrefix={arXiv},
  primaryClass={eess.AS}
}
```

---

## What this repo provides

- **End-to-end SSF/JNF training** for multi-channel mixtures (complex STFT input).
- **Reproducible evaluation** (PESQ / STOI / SDR-style metrics depending on the script configuration).
- **ONNX export + ONNXRuntime sanity checks** for deployment experiments.

---

## Setup

### 1) Create and activate the Conda environment
```bash
conda env create -f environment.yml
conda activate JNF_SSF
```

### 2) Install ONNX and ONNX Runtime
```bash
pip install onnx onnxruntime
```

### 3) Verify GPU availability
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Usage

### Train the model
Edit `config.yaml` as needed, then run:
```bash
python train.py
```

### Evaluate on test set
```bash
python test.py
```

---

## ONNX Export

### Export to ONNX
```bash
python export_jnf_ssf_onnx.py
```

### Validate ONNX outputs (PyTorch vs ONNX)
```bash
python eval_jnf_onnx_metrics.py
```

---

## Repository structure

```
JNF_SSF/
├── models/                  # Network architectures
│   ├── JNF_SSF.py           # Main SSF/JNF model
├── utils/
│   ├── audio_utils.py       # STFT/ISTFT, normalization
│   ├── metrics.py           # PESQ/STOI/etc. helpers
├── config.yaml              # Training/evaluation configuration
├── train.py                 # Training entry point
├── test.py                  # Evaluation entry point
├── export_jnf_ssf_onnx.py   # ONNX export utility
├── eval_jnf_onnx_metrics.py # PyTorch vs ONNX validation script
└── README.md
```

---

## Team and contributions

### Core team (SP CUP 2026 MEMBER INFORMATION: Team Nyquist)
All team members contributed to the end-to-end pipeline—data preparation, model development, experiments, ONNX export checks, and the final report/paper write-up.

- **Fariha Anjum Oshin** (ID: 2106009) — farihaoshin10@gmail.com
**Wahi Farhan Hoque** (ID: 2106100) — wahihoque@gmail.com
**Mahafuza Maisha** (ID: 2106143) — alexfriedman748@gmail.com
**Sumayea Sultana** (ID: 2106099) — sumayeasultana2003@gmail.com
**Md Abu Saleh Akib** (ID: 2106007) — asaub2019@gmail.com
**Zarifa Tabassum** (ID: 2106052) — zarifatabassum49@gmail.com
**Riajul Karim Chowdhury** (ID: 2006153) — riajulkarimchowdhury712@gmail.com
**Md. Nagib Mahfuz** (ID: 2006166) — 2006166@eee.buet.ac.bd
**Md.Symria Raihan** (ID: 2006133) — mssymriaraihan@gmail.com

### Supervision / mentorship
- **Supervisor:** Dr. Mohammad Ariful Haque — arifulhoque@eee.buet.ac.bd (https://eee.buet.ac.bd/people/faculty/dmarh)
- **Tutor:** Aye Thein Maung  — ayetheinmaung32@gmail.com

---

## Notes

- The SSF/JNF family is **not a post-filter on top of a beamformer**: it learns a joint spatial + spectral mapping directly from multi-channel features, which is particularly important when the array spacing pushes the system into aliasing regimes.
- For ONNX export, keep tensor shapes and dynamic axes consistent with the chosen runtime target (mobile/desktop), and validate numerically with `eval_jnf_onnx_metrics.py`.

