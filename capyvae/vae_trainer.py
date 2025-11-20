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
import torch
from capyvae.linear_vae import LinearVAE
from twist_controller.sim.evolution.vae.abo import AsynchronousBO
from capyvae.weighted_dataset import WeightedDataset
from twist_controller.utils.logger import LoggerCallback


class DebugCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        pass


class VAE():
    """
    Variational Autoencoder trainer for design optimization.
    
    Supports flexible dataset inputs - automatically handled by WeightedDataset:
    - File paths (.pt, .npz, .pkl)
    - NumPy arrays or PyTorch tensors
    - Tuples of (data, properties)
    - Dict with 'data' and 'properties' keys
    - WeightedDataset instances
    
    Examples:
        # From file path
        >>> vae = VAE(dataset="data.pt", latent_dim=8, max_epochs=50)
        
        # From arrays with properties
        >>> data = np.random.randn(1000, 128)
        >>> scores = np.random.rand(1000)
        >>> vae = VAE(dataset=(data, scores), latent_dim=16)
        
        # From array without properties (unweighted)
        >>> data = np.random.randn(1000, 128)
        >>> vae = VAE(dataset=data, latent_dim=8)
        
        # With black-box optimizer
        >>> vae = VAE(dataset="data.pt", optimizer="abo", opt_bounds=(-4, 4))
        
        # Custom configuration
        >>> vae = VAE(
        ...     dataset="data.pt",
        ...     latent_dim=16,
        ...     hidden_dims=[1024, 256, 128],
        ...     max_epochs=100,
        ...     batch_size=256,
        ...     weight_type="rank",
        ...     rank_weight_k=0.001
        ... )
    """
    
    def __init__(
        self, 
        dataset,
        # Model parameters
        latent_dim: int = 8,
        hidden_dims: list = None,
        reconstruction_loss: str = 'auto',
        # Training parameters
        max_epochs: int = 80,
        lr: float = 1e-3,
        batch_size: int = 128,
        val_frac: float = 0.05,
        # VAE-specific training
        beta_start: float = 1e-6,
        beta_final: float = 1.0,
        beta_step: float = 1.01,
        beta_step_freq: int = 10,
        beta_warmup: int = 10,
        # Weighted sampling
        property_key: str = "score",
        weight_type: str = "rank",
        rank_weight_k: float = 1e-3,
        weight_quantile: float = None,
        # Data normalization
        normalize: bool = False,
        # Device and paths
        device: str = "cuda:0",
        log_dir: str = None,
        load_from_checkpoint: str = None,
        # Optimizer for design optimization (optional)
        optimizer: str = None,
        opt_bounds: tuple = (-4, 4),
        n_workers: int = 2,
        use_result_buffer: bool = True,
        load_gp: str = None,
        likelihood_variance: float = 1e-3,
        # Logging
        wandb_run=None,
        logger=None,
        seed: int = 10,
    ):
        """
        Initialize VAE trainer.
        
        Args:
            dataset: Dataset input (see WeightedDataset for supported formats)
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer sizes (default: [512, 128, 64, 64])
            reconstruction_loss: Loss type ('auto', 'bernoulli', 'mse')
                - 'auto': Automatically detect based on data (default)
                - 'bernoulli': For binary data (0s and 1s)
                - 'mse': For continuous data
            max_epochs: Maximum training epochs
            lr: Learning rate
            batch_size: Training batch size
            val_frac: Validation set fraction
            beta_start: Initial KL weight (for beta-VAE)
            beta_final: Final KL weight
            beta_step: Beta annealing step multiplier
            beta_step_freq: Steps between beta updates
            beta_warmup: Warmup epochs before annealing
            property_key: Key name for properties in files
            weight_type: Weighting strategy ('rank', 'fb', etc.)
            rank_weight_k: Rank weighting coefficient
            weight_quantile: Quantile for filtering (fb mode)
            normalize: If True, normalizes data to zero mean and unit variance
                      Automatically denormalizes generated samples
            device: Device ('cuda:0', 'cpu', etc.)
            log_dir: Logging directory (auto-generated if None)
            load_from_checkpoint: Path to checkpoint file
            optimizer: Black-box optimizer type ('abo', etc.)
            opt_bounds: Latent space bounds (lb, ub)
            n_workers: Parallel workers for optimization
            use_result_buffer: Cache optimization results
            load_gp: Load Gaussian Process from path
            likelihood_variance: GP likelihood variance
            wandb_run: Existing W&B run instance
            logger: Custom logger instance
            seed: Random seed
        """
        # Setup configuration
        if log_dir is None:
            run_token = datetime.datetime.now().strftime("%m%d%H%M%S")
            log_dir = os.path.join("exp", f"{run_token}")
        
        if hidden_dims is None:
            hidden_dims = [512, 128, 64, 64]
        
        self.hparams = OmegaConf.create({
            "seed": seed,
            "batch_size": batch_size,
            "lr": lr,
            "val_frac": val_frac,
            "wandb": wandb_run is not None,
            "max_epochs": max_epochs,
            "log_dir": log_dir,
            "latent_dim": latent_dim,
            "input_size": None,  # Set after data loading
            "hidden_dims": hidden_dims,
            "reconstruction_loss": reconstruction_loss,
            "beta_start": beta_start,
            "beta_final": beta_final,
            "beta_step": beta_step,
            "beta_step_freq": beta_step_freq,
            "beta_warmup": beta_warmup,
            "load_from_checkpoint": load_from_checkpoint,
            "property_key": property_key,
            "weight_type": weight_type,
            "rank_weight_k": rank_weight_k,
            "weight_quantile": weight_quantile,
            "normalize": normalize,
            "device": device,
        })
        
        self.log_path = log_dir
        self.n_workers = n_workers
        self.use_result_buffer = use_result_buffer
        self.load_gp = load_gp
        self.likelihood_variance = likelihood_variance
        self.opt_bounds = opt_bounds
        self.logger = logger
        self._dataset_input = dataset

        # Setup wandb logger
        if wandb_run is not None:
            self.wandb_logger = WandbLogger(
                config=OmegaConf.to_container(self.hparams), 
                save_dir=self.log_path
            )
        else:
            self.wandb_logger = None
        
        self._setup_vae()

        if optimizer is not None:
            self.optimizer_name = optimizer
            self._setup_optimizer(optimizer=optimizer)
        else:
            self.bboptimizer = None


    def _setup_vae(self):
        """Setup VAE model, dataset, and trainer."""
        pl.seed_everything(self.hparams.seed)
        
        # Create dataset - WeightedDataset handles all input formats
        if isinstance(self._dataset_input, WeightedDataset):
            # Already a WeightedDataset instance
            self.data = self._dataset_input
        else:
            # Let WeightedDataset handle all other formats
            # (paths, arrays, tuples, dicts, etc.)
            self.data = WeightedDataset(
                data=self._dataset_input,
                batch_size=self.hparams.batch_size,
                val_frac=self.hparams.val_frac,
                property_key=self.hparams.property_key,
                weight_type=self.hparams.weight_type,
                rank_weight_k=self.hparams.rank_weight_k,
                weight_quantile=self.hparams.weight_quantile,
                normalize=self.hparams.normalize,
            )
        
        self.data.setup("init")
        self.hparams.input_size = self.data.data_shape[1]

        # Load model
        self.model = LinearVAE(self.hparams).to(device=torch.device(self.hparams.device))
        
        # Set normalization statistics in the model if available
        norm_stats = self.data.get_normalization_stats()
        if norm_stats is not None:
            self.model.set_normalization_stats(norm_stats['mean'], norm_stats['std'])
            print(f"✓ Normalization enabled in model")
        if self.hparams.load_from_checkpoint is not None:
            self.model = LinearVAE.load_from_checkpoint(
                self.hparams.load_from_checkpoint, 
                map_location=torch.device(self.hparams.device)
            )

        # Main trainer
        self.trainer = pl.Trainer(
            accelerator="gpu",
            devices=[self.model.device.index] if self.model.device.type == 'cuda' else 'auto',
            default_root_dir=self.log_path,
            max_epochs=self.hparams.max_epochs,
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    every_n_epochs=1, 
                    monitor="loss/val", 
                    save_top_k=3,
                    save_last=True,
                    dirpath=self.log_path,
                    filename="best",
                ), 
                RichProgressBar(
                    theme=RichProgressBarTheme(
                        progress_bar="red", 
                        progress_bar_finished="green"
                    )
                ),
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
        """
        Train the VAE model.
        
        Alias for train() method for better clarity.
        """
        self.train()
    
    def train(self):
        # Fit
        print("Training started")
        self.trainer.fit(self.model, datamodule=self.data)
        print("Training finished")

    def test_vae(self, n_samples=5, n_latent_samples=3, verbose=True):
        """
        Test VAE model (user-friendly alias for test()).
        
        Args:
            n_samples: Number of validation samples to test reconstruction on
            n_latent_samples: Number of new designs to generate from latent space
            verbose: Whether to print detailed output
        
        Returns:
            dict: Contains reconstruction errors and generated samples
        """
        return self.test(n_samples=n_samples, n_latent_samples=n_latent_samples, verbose=verbose)

    def test(self, n_samples=5, n_latent_samples=3, verbose=True):
        """
        Test VAE model to provide intuition about reconstruction and generation.
        
        Args:
            n_samples: Number of validation samples to test reconstruction on
            n_latent_samples: Number of new designs to generate from latent space
            verbose: Whether to print detailed output
        
        Returns:
            dict: Contains reconstruction errors and generated samples
        """
        self.model.eval()
        
        with torch.no_grad():
            # ========== Part 1: Test Reconstruction Quality ==========
            if verbose:
                print("\n" + "="*60)
                print("PART 1: RECONSTRUCTION TEST")
                print("="*60)
                print("Testing how well the VAE reconstructs existing data points\n")
            
            data = self.data.data_val
            indices = torch.randint(low=0, high=data.size(0), size=(n_samples,))
            original_data = data[indices]
            
            # Move to device
            original_data_tensor = original_data.clone().detach().to(
                device=self.model.device, 
                dtype=self.model.dtype
            )
            
            # Encode and decode
            mu, logstd = self.model.encode_to_params(original_data_tensor)
            reconstructed = self.model.decoder(mu)
            
            # Calculate reconstruction error
            mse_errors = torch.mean((original_data_tensor - reconstructed) ** 2, dim=1)
            avg_mse = torch.mean(mse_errors).item()
            
            if verbose:
                print(f"Sample\t{'Original (first 10 dims)':<30} | {'Reconstructed (first 10 dims)':<30} | MSE")
                print("-" * 95)
                for i, (orig, recon, err) in enumerate(zip(original_data, reconstructed, mse_errors)):
                    orig_str = str(orig[:10].cpu().numpy().round(3))
                    recon_str = str(recon[:10].cpu().numpy().round(3))
                    print(f"  {i+1}\t{orig_str:<30} | {recon_str:<30} | {err.item():.6f}")
                
                print(f"\nAverage Reconstruction MSE: {avg_mse:.6f}")
                print(f"Latent vectors (mu) shape: {mu.shape}")
            
            # ========== Part 2: Sample New Designs from Latent Space ==========
            if verbose:
                print("\n" + "="*60)
                print("PART 2: GENERATION FROM LATENT SPACE")
                print("="*60)
                print("Generating new designs by sampling from latent space\n")
            
            # Sample from prior (standard normal distribution)
            z_samples = self.model.sample_prior(n_latent_samples)
            generated_designs = self.model.decoder(z_samples)
            
            # Denormalize if normalization was applied
            generated_designs_denorm = self.model.denormalize(generated_designs)
            
            if verbose:
                print(f"Sampled {n_latent_samples} random latent vectors from N(0,1)")
                print(f"Generated designs shape: {generated_designs.shape}")
                if self.model.is_normalized():
                    print(f"✓ Denormalization applied to generated samples\n")
                else:
                    print(f"(No normalization was used)\n")
                
                print("Generated Designs (first 10 dimensions):")
                print("-" * 60)
                for i, design in enumerate(generated_designs_denorm):
                    design_str = str(design[:10].cpu().numpy().round(3))
                    print(f"  Design {i+1}: {design_str}")
            
            # ========== Part 3: Latent Space Statistics ==========
            if verbose:
                print("\n" + "="*60)
                print("PART 3: LATENT SPACE STATISTICS")
                print("="*60)
                
                # Encode a larger batch to see latent space distribution
                batch_size = min(100, data.size(0))
                sample_indices = torch.randint(low=0, high=data.size(0), size=(batch_size,))
                sample_batch = data[sample_indices].clone().detach().to(
                    device=self.model.device, 
                    dtype=self.model.dtype
                )
                mu_batch, logstd_batch = self.model.encode_to_params(sample_batch)
                
                print(f"\nLatent space analysis (from {batch_size} samples):")
                print(f"  Latent dimension: {self.hparams.latent_dim}")
                print(f"  Mean of latent means: {mu_batch.mean(dim=0).cpu().numpy().round(3)}")
                print(f"  Std of latent means:  {mu_batch.std(dim=0).cpu().numpy().round(3)}")
                print(f"  Mean of latent stds:  {torch.exp(logstd_batch).mean(dim=0).cpu().numpy().round(3)}")
            
            print("\n" + "="*60 + "\n")
        
        self.model.train()
        
        # Return results for programmatic access
        return {
            'reconstruction_mse': avg_mse,
            'original_samples': original_data.cpu().numpy(),
            'reconstructed_samples': reconstructed.cpu().numpy(),
            'latent_vectors': mu.cpu().numpy(),
            'generated_designs': generated_designs_denorm.cpu().numpy(),  # Return denormalized
            'latent_samples': z_samples.cpu().numpy(),
            'is_normalized': self.model.is_normalized()
        }

    def _latent_to_design(self, latent_value):
        latent_value = torch.tensor(np.array([latent_value]), device=self.model.device, dtype=self.model.dtype)
        designs = self.model.decoder(latent_value)
        return designs[0]
    
    def denormalize(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """
        Denormalize data back to original scale.
        
        This is a convenience method that calls the model's denormalize function.
        Use this to convert generated samples back to the original data scale.
        
        Args:
            normalized_data: Normalized data tensor (can be on CPU or GPU)
            
        Returns:
            Denormalized data tensor (same device as input)
            
        Example:
            >>> z = vae_trainer.model.sample_prior(10)
            >>> generated = vae_trainer.model.decoder(z)
            >>> original_scale = vae_trainer.denormalize(generated)
        """
        return self.model.denormalize(normalized_data)

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
    # Example 1: Simple usage with file path
    vae_trainer = VAE(dataset="data/designs_asym_filtered_onehot.pt", max_epochs=2)
    vae_trainer.train_vae()
    vae_trainer.test_vae()
    
    # # Example 2: With custom dataset (e.g., from external source)
    # from drone_dataset import DroneDataset
    # drone_data = DroneDataset("/home/chen/LAB/CapyVAE/drone/log.txt")
    # configs, scores = drone_data.export_to_array(include_added_mass=False)
    
    # vae_trainer = VAE(
    #     dataset=(configs, scores),  # Pass as tuple
    #     optimizer="abo",
    #     latent_dim=8,
    #     max_epochs=2
    # )
    # vae_trainer.train_vae()
    # vae_trainer.test_vae()
    # init_designs = vae_trainer.get_init_designs()