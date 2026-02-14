# Model Testing Script

This script loads a trained model from a checkpoint and tests it on the test dataset from the HDF5 file.

## Usage

### Basic Usage (Recommended)
```bash
cd src
python scripts/test_model.py
```

This will:
- Load the JNF model configuration from `config/jnf_config.yaml`
- Use the best checkpoint from `../logs/ckpts/best.ckpt` (as specified in config)
- Test the model on the test dataset from your HDF5 file
- Display test results and metrics

### Test SSF Model Instead
```bash
python scripts/test_model.py --model ssf --config config/ssf_config.yaml
```

### Quick Results Display
```bash
python scripts/show_results.py
```

This script displays the test results without running the full testing process.

## Arguments

- `--config`: Path to the configuration YAML file (default: `config/jnf_config.yaml`)
- `--ckpt`: Path to the checkpoint file (optional, uses config default if not provided)
- `--model`: Type of model to test (`jnf` or `ssf`, default: `jnf`)

## Expected Output

```
Loading configuration from config/jnf_config.yaml...
Using checkpoint: ../logs/ckpts/best.ckpt
Testing JNF model
Setting up data module...
Data module initialized successfully.
Loading model from checkpoint...
Model loaded successfully.
Setting up trainer for testing...
Starting testing on test dataset...
Testing completed!
Test results:
  test/loss: 2.967
  test/clean_td_loss: 0.129
  test/noise_td_loss: 0.025
  test/clean_mag_loss: 1.110
  test/noise_mag_loss: 0.312
  monitor_loss: 2.967
  test/si_sdr: -21.975
```

## Understanding the Results

The test script evaluates the model's speech enhancement performance using several metrics:

- **`test/loss`**: Overall loss combining time-domain and magnitude losses
- **`test/clean_td_loss`**: Time-domain loss for clean speech reconstruction
- **`test/noise_td_loss`**: Time-domain loss for noise suppression
- **`test/clean_mag_loss`**: Magnitude loss for clean speech in frequency domain
- **`test/noise_mag_loss`**: Magnitude loss for noise in frequency domain
- **`monitor_loss`**: Same as test/loss, used for model monitoring
- **`test/si_sdr`**: Scale-Invariant Signal-to-Distortion Ratio (in dB) - higher is better

**Interpretation**:
- SI-SDR values above 0 dB indicate good speech enhancement
- Values between -10 to 0 dB suggest moderate performance
- Values below -10 dB may indicate the model needs more training or has issues

## Requirements

- Trained model checkpoint file
- Configuration file matching the model type
- HDF5 data file and metadata file as specified in config
- PyTorch Lightning and other dependencies

## Example Output

```
Loading configuration from config/jnf_config.yaml...
Using checkpoint: ../logs/ckpts/best.ckpt
Testing JNF model
Setting up data module...
Data module initialized successfully.
Loading model from checkpoint...
Model loaded successfully.
Setting up trainer for testing...
Starting testing on test dataset...
Testing completed!
Test results:
  test/loss: 0.123
  test/snr: 15.67
  ...
```