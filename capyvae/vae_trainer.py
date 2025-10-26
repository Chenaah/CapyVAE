import datetime
import os
import argparse
import pdb
import random
import sys
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
from trieste.space import Box, DiscreteSearchSpace
import torch
from linear_vae import LinearVAE
from twist_controller.sim.evolution.vae.abo import AsynchronousBO
from weighted_dataset import WeightedDataset
from twist_controller.utils.logger import LoggerCallback


class DebugCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        pass


class VAE():
    def __init__(self, cfg=None, dataset=None, wandb_run=None, optimizer=None, logger=None, optimizer_kwargs={}):
         # Create arg parser
        self.hparams = cfg if cfg is not None else get_vae_cfg()
        self.hparams.dataset_path = dataset
        # log_path = create_log_dir("pretrain", hparams.dataset, hparams.max_epochs, 1, note=f"ls{hparams.latent_dim}", log_folder_name=os.path.join(CILIA2D_ROOT_DIR, "logs", "retraining"))
        self.log_path = self.hparams.log_dir 
        self.n_workers = optimizer_kwargs.get("n_workers", 2)
        self.use_result_buffer = optimizer_kwargs.get("use_result_buffer", True)
        self.load_gp = optimizer_kwargs.get("load_gp", None)
        self.likelihood_variance = optimizer_kwargs.get("likelihood_variance", 1e-3)
        self.opt_bounds = optimizer_kwargs.get("opt_bounds", (-4, 4))
        self.fitness_func = optimizer_kwargs.get("fitness_func", None)
        self.logger = logger

        if wandb_run is not None:
            # The logger should use the existed wandb run automatically
            self.wandb_logger = WandbLogger(config=OmegaConf.to_container(self.hparams), 
                                            save_dir=self.log_path)
        else:
            self.wandb_logger = None
        
        self._setup_vae()

        if optimizer is not None:
            self.optimizer_name = optimizer
            self._setup_optimizer(optimizer=optimizer)
        else:
            self.bboptimizer = None


    def _setup_vae(self):
        pl.seed_everything(self.hparams.seed)
        # Create data
        self.data = WeightedDataset(self.hparams)
        self.data.setup("init")
        self.hparams.input_size = self.data.data_shape[1]

        # Load model
        self.model = LinearVAE(self.hparams).to(device=torch.device(self.hparams.device))
        if self.hparams.load_from_checkpoint is not None:
            self.model = LinearVAE.load_from_checkpoint(self.hparams.load_from_checkpoint, map_location=torch.device(self.hparams.device))
            # utils.update_hparams(self.hparams, self.model) TODO: Fix this if further training is needed
        # Data used for investigating the VAE
        # self.model.max_idx = self.data.max_idx
        # self.model.max_length = self.data.max_length
        # self.model.sampled_original_data = self.data.sampled_original_data

        # Main trainerx
        self.trainer = pl.Trainer(
            accelerator="gpu",
            devices=[self.model.device.index] if self.model.device.type == 'cuda' else 'auto',
            default_root_dir=self.log_path,
            max_epochs=self.hparams.max_epochs,
            callbacks=[pl.callbacks.ModelCheckpoint(
                            every_n_epochs=1, monitor="loss/val", save_top_k=3,
                            save_last=True,
                            dirpath=self.log_path,
                            filename="best",
                        ), 
                        RichProgressBar(theme=RichProgressBarTheme(progress_bar="red", progress_bar_finished="green")),
                        LoggerCallback(),
                        DebugCallback()
                        ],
            logger=self.wandb_logger,
        )

    def _setup_optimizer(self, optimizer):
        assert optimizer in ["abo", "dbo", "dbo2"] # TODO: Add other optimizers
        lb, ub = self.opt_bounds
        if optimizer == "cma":
            param = ng.p.Array(shape=(self.hparams.latent_dim,), lower=lb, upper=ub)
            self.bboptimizer = ng.optimizers.CMA(parametrization=param, budget=100, num_workers=1)
        elif optimizer == "turbo":
            self.bboptimizer = TuRBO(dim=self.hparams.latent_dim,
                                      n_init=20,
                                      lb=lb,
                                      ub=ub,
                                      candidate_buffer_size=self.n_workers
                                      )
        elif optimizer == "abo":
            self.bboptimizer = AsynchronousBO(dim=self.hparams.latent_dim,
                                              n_init=self.n_workers,
                                              lb=lb,
                                              ub=ub,
                                              load_gp=self.load_gp,
                                              log_dir=os.path.join(os.path.dirname(self.log_path), "abo_models"),
                                              likelihood_variance= self.likelihood_variance
                                              )
            
        
        self.param_dict = {}  # tuple(polished_design) -> param
        self.result_buffer = {}


    def train_vae(self):
        # Fit
        print("Training started")
        self.trainer.fit(self.model, datamodule=self.data)
        print("Training finished")


    def test_vae(self):
        data = self.data.data_val
        indices = torch.randint(low=0, high=data.size(0), size=(10,))
        sampled_data = data[indices]
        print("BEFORE: ")
        for d in sampled_data:
            print(d)
        sampled_data = torch.tensor(sampled_data, device=self.model.device, dtype=self.model.dtype)
        mu, logstd = self.model.encode_to_params(sampled_data)
        designs = self.model.decoder(mu)
        print("AFTER: ")
        for d in designs:
            print(d)

    def _latent_to_design(self, latent_value):
        latent_value = torch.tensor(np.array([latent_value]), device=self.model.device, dtype=self.model.dtype)
        designs = self.model.decoder(latent_value)
        return designs[0]

    def get_init_designs(self):
        assert self.bboptimizer is not None, "Optimizer not set up"
        params = self.bboptimizer.ask_init()
        self.init_designs = [self._latent_to_design(param) for param in params]

        print("Init designs: ", self.init_designs)
        return self.init_designs
    

    def get_new_design(self):
        # Ask from optimizer -> decoded to design
        assert self.bboptimizer is not None, "Optimizer not set up"
        assert self._latent_to_design is not None, "Latent to design function not set up"

        latent_value = self.bboptimizer.ask() # Encourage exploration a bit

        polished_design = self._latent_to_design(latent_value)

        # polished_design = extent_to_5x1(polished_design)
        print("New design: ", polished_design)
        self.param_dict[tuple(polished_design)] = latent_value

        
        if self.use_result_buffer and tuple(polished_design) in self.result_buffer:
            print("Already evaluated >-<")
            # random_point = self.bboptimizer.search_space.sample(1)
            self.tell_result(polished_design, self.result_buffer[tuple(polished_design)],
                             latent_value=torch.tensor(np.array([latent_value]), device=self.model.device, dtype=self.model.dtype))
            return self.get_new_design()
        
        self.result_buffer[tuple(polished_design)] = None

        return polished_design

    def tell_init_results(self, scores, run_names=None):
        # Score: the higher the better
        losses = [-score for score in scores]
        self.bboptimizer.tell_init(losses)
        for design, score, name in zip(self.init_designs, scores, run_names):
            self.result_buffer[tuple(design)] = score

            self.logger.log(("polished_design", design), "run_name", name)
            self.logger.log(("polished_design", design), "score", score)
            self.logger.flush()

    
    def tell_result(self, design, score, run_name=None, latent_value=None):
        # Score: the higher the better
        try:
            param = self.param_dict[tuple(design)]
        except KeyError:
            pdb.set_trace()
        loss = -score
        self.bboptimizer.tell(param, loss)
        self.result_buffer[tuple(design)] = score


        # pdb.set_trace()
        self.logger.log(("polished_design", design), "run_name", run_name)
        if latent_value is None:
            self.logger.log(("polished_design", design), "score", score)
        else:
            self.logger.log(("latent", latent_value), "score", score)

        self.logger.flush()

        if run_name is not None:
            self.bboptimizer._save_gp(name=f"{run_name}.pkl")



    def optimize(self):
        # Move this part out to keep the code clean
        # As this involves communication with other programs
        raise NotImplementedError



def get_vae_cfg():
    run_token = datetime.datetime.now().strftime("%m%d%H%M%S")

    vae_cfg = {
        # General
        "seed": 10,
        "batch_size": 128,
        "lr": 1e-3,
        "val_frac": 0.05,
        "wandb": False,
        "max_epochs": 2, # 80, 
        "log_dir": os.path.join("exp", f"{run_token}"),
        "dataset_path": "designs_asym_filtered_onehot.pt",

        # Share
        "latent_dim": 8,
        "input_size": None,
        "hidden_dims": [512, 128, 64, 64],

        # Pretraining
        "beta_start": 1e-6,
        "beta_final": 1,
        "beta_step": 1.01,
        "beta_step_freq": 10,
        "beta_warmup": 10,
        "load_from_checkpoint": None,
        "property_key": "score",
        "weight_type": "rank",
        "rank_weight_k": 1e-3,
        "weight_quantile": None,
        "lso_strategy": "sample", #opt
        "query_budget": 500,
        "retraining_frequency": 5,
        "n_retrain_epochs": 2, #1 # 0.1
        "n_init_retrain_epochs": 10,
        "samples_per_model": 50,
        "opt_grid_len": 50,
        "device": "cuda:0",
        "likelihood_variance": 1e-3,

    }
    cfg = OmegaConf.create(vae_cfg)
    return cfg

if __name__ == "__main__":

    vae_trainer = VAE(optimizer="abo", dataset="designs_asym_filtered_onehot.pt")
    vae_trainer.train_vae()
    vae_trainer.test_vae()
    init_designs = vae_trainer.get_init_designs()