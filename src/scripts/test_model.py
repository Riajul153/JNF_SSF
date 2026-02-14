import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from models.exp_jnf import JNFExp
from models.exp_ssf import SSFExp
from data.datamodule import HDF5DataModule
import yaml
import argparse
import torch

def load_model(ckpt_file: str, config, model_type: str = 'jnf'):
    """Load the trained model from checkpoint."""
    if model_type.lower() == 'jnf':
        exp_class = JNFExp
    elif model_type.lower() == 'ssf':
        exp_class = SSFExp
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: 'jnf', 'ssf'")

    init_params = exp_class.get_init_params(config)
    model = exp_class.load_from_checkpoint(ckpt_file, **init_params)
    return model

def test_model(config_path: str, ckpt_path: str = None, model_type: str = 'jnf'):
    """
    Load the best model and test it on the test dataset.

    Args:
        config_path: Path to the configuration YAML file
        ckpt_path: Path to the checkpoint file (optional, uses config default if not provided)
        model_type: Type of model ('jnf' or 'ssf')
    """
    # Set Tensor Core precision for better performance
    torch.set_float32_matmul_precision('high')
    # Load configuration
    print(f"Loading configuration from {config_path}...")
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    # Set seed for reproducibility
    pl.seed_everything(config.get('seed', 0), workers=True)

    # Determine checkpoint path
    if ckpt_path is None:
        ckpt_path = config['training'].get('resume_ckpt', None)

    if ckpt_path is None:
        raise ValueError("No checkpoint path provided. Either specify ckpt_path or set resume_ckpt in config.")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    print(f"Using checkpoint: {ckpt_path}")
    print(f"Testing {model_type.upper()} model")

    # Initialize data module
    print("Setting up data module...")
    data_config = config['data']
    dm = HDF5DataModule(**data_config)
    print("Data module initialized successfully.")

    # Load the trained model
    print("Loading model from checkpoint...")
    model = load_model(ckpt_path, config, model_type)
    print("Model loaded successfully.")

    # Create trainer for testing
    print("Setting up trainer for testing...")
    trainer = pl.Trainer(
        devices=config['training'].get('devices', 1),
        accelerator=config['training'].get('accelerator', 'gpu'),
        strategy=config['training'].get('strategy', 'auto'),
        enable_progress_bar=True,
        logger=False,  # Disable logging for testing
        enable_checkpointing=False,  # Disable checkpointing during testing
    )

    # Test the model on the test dataset
    print("Starting testing on test dataset...")
    test_results = trainer.test(model, dm, ckpt_path=ckpt_path)

    print("Testing completed!")
    print(f"Raw test_results: {test_results}")
    print(f"Type of test_results: {type(test_results)}")
    
    if test_results and len(test_results) > 0:
        print("Test results:")
        for key, value in test_results[0].items():
            print(f"  {key}: {value}")
    else:
        print("No test results returned!")
        print("This might be because no metrics were logged during testing.")

    return test_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the trained model on the test dataset')
    parser.add_argument('--config', type=str, default='config/jnf_config.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to the checkpoint file (optional, uses config default if not provided)')
    parser.add_argument('--model', type=str, default='jnf', choices=['jnf', 'ssf'],
                        help='Type of model to test (jnf or ssf)')

    args = parser.parse_args()

    try:
        test_model(args.config, args.ckpt, args.model)
    except Exception as e:
        print(f"Error during testing: {e}")
        sys.exit(1)