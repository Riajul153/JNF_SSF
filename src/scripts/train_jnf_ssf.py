import os
import sys
from typing import Optional
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
import yaml

from data.datamodule import HDF5DataModule
from data.flac_datamodule import FlacDataModule
from models.exp_jnf import JNFExp
from models.exp_ssf import SSFExp
from models.models import FTJNF, JNF_SSF


def setup_logging(exp_name: str, tb_log_dir: str, version_id: Optional[int] = None):
    if version_id is None:
        tb_logger = pl_loggers.TensorBoardLogger(
            tb_log_dir, name=exp_name, log_graph=False
        )
        version_id = int((tb_logger.log_dir).split("_")[-1])
    else:
        tb_logger = pl_loggers.TensorBoardLogger(
            tb_log_dir, name=exp_name, log_graph=False, version=version_id
        )
    return tb_logger, version_id


def build_trainer(
    training_config,
    logger,
    early_stopping_callback,
    best_checkpoint_callback,
    epoch_checkpoint_callback,
):
    trainer_args = {
        "enable_model_summary": True,
        "logger": logger,
        "log_every_n_steps": training_config.get("log_every_n_steps", 100),
        "max_epochs": training_config["max_epochs"],
        "gradient_clip_val": training_config["gradient_clip_val"],
        "gradient_clip_algorithm": training_config["gradient_clip_algorithm"],
        "callbacks": [
            early_stopping_callback,
            best_checkpoint_callback,
            epoch_checkpoint_callback,
            ModelSummary(max_depth=2),
        ],
    }
    if "devices" in training_config or "accelerator" in training_config:
        if "devices" in training_config:
            trainer_args["devices"] = training_config["devices"]
        if "accelerator" in training_config:
            trainer_args["accelerator"] = training_config["accelerator"]
    elif "gpus" in training_config:
        gpus = training_config["gpus"]
        if gpus:
            trainer_args["accelerator"] = "gpu"
            trainer_args["devices"] = gpus
        else:
            trainer_args["accelerator"] = "cpu"
            trainer_args["devices"] = 1
    if "precision" in training_config:
        trainer_args["precision"] = training_config["precision"]
    return pl.Trainer(**trainer_args)


def build_datamodule(data_config):
    data_source = data_config.get("source", "hdf5").lower()
    if data_source == "flac":
        return FlacDataModule(**data_config)
    return HDF5DataModule(**data_config)


def build_early_stopping(patience: int = 10):
    return EarlyStopping(
        monitor="val/loss",
        patience=patience,
        mode="min",
        verbose=True,
    )


class ValSiSdrPrinter(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        sisdr = trainer.callback_metrics.get("val/si_sdr")
        if sisdr is not None:
            print(f"val/si_sdr: {sisdr:.4f}")


class BestSiSdrSaver(pl.Callback):
    def __init__(self, dirpath: str, prefix: str):
        super().__init__()
        self.dirpath = dirpath
        self.prefix = prefix
        self.best = None

    def on_validation_epoch_end(self, trainer, pl_module):
        sisdr = trainer.callback_metrics.get("val/si_sdr")
        if sisdr is None:
            return
        sisdr_value = float(sisdr)
        if self.best is None or sisdr_value > self.best:
            self.best = sisdr_value
            os.makedirs(self.dirpath, exist_ok=True)
            ckpt_path = os.path.join(self.dirpath, f"{self.prefix}-best-sisdr.ckpt")
            pt_path = os.path.join(self.dirpath, f"{self.prefix}-best-sisdr.pt")
            trainer.save_checkpoint(ckpt_path)
            torch.save(pl_module.state_dict(), pt_path)


def pick_jnf_experiment_params(ssf_experiment):
    allowed = {
        "learning_rate",
        "weight_decay",
        "loss_alpha",
        "cirm_comp_K",
        "cirm_comp_C",
        "reference_channel",
    }
    return {k: v for k, v in ssf_experiment.items() if k in allowed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-jnf", action="store_true", help="Skip JNF training")
    args = parser.parse_args()

    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
    ssf_config_path = os.path.join(config_dir, "ssf_flac_config.yaml")
    jnf_config_path = os.path.join(config_dir, "jnf_flac_config.yaml")

    with open(ssf_config_path) as config_file:
        ssf_config = yaml.safe_load(config_file)
    with open(jnf_config_path) as config_file:
        jnf_config = yaml.safe_load(config_file)

    pl.seed_everything(ssf_config.get("seed", 0), workers=True)

    data_config = dict(ssf_config["data"])
    stft_length = data_config.get("stft_length_samples", 512)
    stft_shift = data_config.get("stft_shift_samples", 256)
    dm = build_datamodule(data_config)

    training_config = ssf_config["training"]
    tb_log_dir = ssf_config["logging"]["tb_log_dir"]
    ckpt_dir = ssf_config["logging"]["ckpt_dir"]

    if not args.skip_jnf:
        print("Setting up JNF...")
        jnf_logger, _ = setup_logging("JNF", tb_log_dir)
        jnf_best_checkpoint = ModelCheckpoint(
            monitor="val/loss",
            save_top_k=1,
            mode="min",
            dirpath=ckpt_dir,
            filename="jnf-best",
        )
        jnf_epoch_checkpoint = ModelCheckpoint(
            save_top_k=-1,
            dirpath=ckpt_dir,
            filename="jnf-{epoch:02d}",
        )
        jnf_trainer = build_trainer(
            training_config,
            jnf_logger,
            build_early_stopping(),
            jnf_best_checkpoint,
            jnf_epoch_checkpoint,
        )
        jnf_trainer.callbacks.append(ValSiSdrPrinter())
        jnf_trainer.callbacks.append(BestSiSdrSaver(ckpt_dir, "jnf"))
        jnf_model = FTJNF(**jnf_config["network"])
        jnf_exp = JNFExp(
            model=jnf_model,
            stft_length=stft_length,
            stft_shift=stft_shift,
            **pick_jnf_experiment_params(ssf_config["experiment"]),
        )
        jnf_trainer.fit(jnf_exp, dm, ckpt_path=None)
    else:
        print("Skipping JNF training.")

    print("Setting up SSF...")
    ssf_logger, _ = setup_logging("JNF-SSF", tb_log_dir)
    ssf_best_checkpoint = ModelCheckpoint(
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        dirpath=ckpt_dir,
        filename="ssf-best",
    )
    ssf_epoch_checkpoint = ModelCheckpoint(
        save_top_k=-1,
        dirpath=ckpt_dir,
        filename="ssf-{epoch:02d}",
    )
    ssf_trainer = build_trainer(
        training_config,
        ssf_logger,
        build_early_stopping(),
        ssf_best_checkpoint,
        ssf_epoch_checkpoint,
    )
    ssf_trainer.callbacks.append(ValSiSdrPrinter())
    ssf_trainer.callbacks.append(BestSiSdrSaver(ckpt_dir, "ssf"))
    ssf_model = JNF_SSF(**ssf_config["network"])
    ssf_exp = SSFExp(
        model=ssf_model,
        stft_length=stft_length,
        stft_shift=stft_shift,
        **ssf_config["experiment"],
    )
    ssf_trainer.fit(
        ssf_exp,
        dm,
        ckpt_path=ssf_config["training"].get("ssf_resume_ckpt", None),
    )
