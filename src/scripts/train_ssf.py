import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelSummary, EarlyStopping, ModelCheckpoint
from models.exp_ssf import SSFExp
from models.models import JNF_SSF
from data.datamodule import HDF5DataModule
from data.flac_datamodule import FlacDataModule
from typing import Optional
import yaml

EXP_NAME='JNF-SSF'

def setup_logging(tb_log_dir: str, version_id: Optional[int]= None):
    """
    Set-up a Tensorboard logger.

    :param tb_log_dir: path to the log dir
    :param version_id: the version id (integer). Consecutive numbering is used if no number is given. 
    """

    if version_id is None:
        tb_logger = pl_loggers.TensorBoardLogger(tb_log_dir, name=EXP_NAME, log_graph=False)

        # get current version id
        version_id = int((tb_logger.log_dir).split('_')[-1])
    else: 
        tb_logger = pl_loggers.TensorBoardLogger(tb_log_dir, name=EXP_NAME, log_graph=False, version=version_id)

    return tb_logger, version_id

def load_model(ckpt_file: str,
               _config):
    init_params = SSFExp.get_init_params(_config)
    model = SSFExp.load_from_checkpoint(ckpt_file, **init_params)
    model.to('cuda')
    return model

def get_trainer(devices, logger, max_epochs, gradient_clip_val, gradient_clip_algorithm, strategy, accelerator, early_stopping_callback, best_checkpoint_callback, epoch_checkpoint_callback):
    return pl.Trainer(enable_model_summary=True,
                         logger=logger,
                         devices=devices,
                         log_every_n_steps=100,
                         max_epochs=max_epochs,
                         gradient_clip_val = gradient_clip_val,
                         gradient_clip_algorithm = gradient_clip_algorithm,
                         strategy = strategy,
                         accelerator = accelerator,
                         callbacks=[
                             early_stopping_callback,
                             best_checkpoint_callback,
                             epoch_checkpoint_callback,
                             ModelSummary(max_depth=2)
                                    ],

                         )

if __name__=="__main__":

    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "ssf_flac_config.yaml")
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)




    ## REPRODUCIBILITY
    pl.seed_everything(config.get('seed', 0), workers=True)

    ## LOGGING
    tb_logger, version = setup_logging(config['logging']['tb_log_dir'])

    ## DATA
    data_config = config['data']
    stft_length = data_config.get('stft_length_samples', 512)
    stft_shift = data_config.get('stft_shift_samples', 256)
    data_source = data_config.get('source', 'hdf5').lower()
    if data_source == 'flac':
        dm = FlacDataModule(**data_config)
    else:
        dm = HDF5DataModule(**data_config)

    ## CONFIGURE EXPERIMENT
    ckpt_file = config['training'].get('resume_ckpt', None)
    model = JNF_SSF(**config['network'])
    exp = SSFExp(model=model,
                stft_length=stft_length,
                stft_shift=stft_shift,
                **config['experiment'])

    ## EARLY STOPPING
    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=10,
        mode="min",
        verbose=True
    )

    ## MODEL CHECKPOINT
    best_checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        dirpath=config['logging']['ckpt_dir'],
        filename='best'
    )
    epoch_checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,

        
        dirpath=config['logging']['ckpt_dir'],
        filename='ssf-{epoch:02d}'
    )

    ## TRAIN
    trainer = get_trainer(logger=tb_logger, early_stopping_callback=early_stopping, best_checkpoint_callback=best_checkpoint_callback, epoch_checkpoint_callback=epoch_checkpoint_callback, **config['training'])
    trainer.fit(exp, dm, ckpt_path=ckpt_file)
