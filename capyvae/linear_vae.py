""" Contains code for the shapes model """

import itertools
import os
import pdb
import warnings
import numpy as np
import torch
from torch import nn, distributions
# from torchvision.utils import make_grid
import pytorch_lightning as pl
from torch.nn import functional as F
import wandb
from torchsummary import summary

class BaseVAE(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams_ = hparams
        self.save_hyperparameters()
        self.latent_dim = hparams.latent_dim
        self.max_idx = None # placeholder
        self.max_length = None # placeholder
        self.sampled_original_data = None # placeholder

        # Register buffers for prior
        self.register_buffer("prior_mu", torch.zeros([self.latent_dim]))
        self.register_buffer("prior_sigma", torch.ones([self.latent_dim]))
        
        # Register buffers for data normalization (for denormalization of generated samples)
        # Will be set by the trainer if normalization is enabled
        self.register_buffer("data_mean", None)
        self.register_buffer("data_std", None)

        # Create beta
        self.beta = hparams.beta_final
        self.beta_annealing = False
        if hparams.beta_start is not None:
            self.beta_annealing = True
            self.beta = hparams.beta_start
            assert (
                hparams.beta_step is not None
                and hparams.beta_step_freq is not None
                and hparams.beta_warmup is not None
            )

        self.logging_prefix = None
        self.log_progress_bar = False

    def forward(self, x):
        """ calculate the VAE ELBO """
        mu, logstd = self.encode_to_params(x)
        
        # Use deterministic encoding when beta is very low or zero
        # This prevents posterior collapse
        if self.beta < 0.01 and not self.beta_annealing:
            # Deterministic autoencoder mode - use mu directly
            z_sample = mu
            kl_loss = torch.zeros(1, device=x.device)
        else:
            # Standard VAE with sampling
            std = torch.exp(logstd)
            std = torch.clamp(std, min=1e-6)
            try:
                encoder_distribution = torch.distributions.Normal(
                    loc=mu, scale=std
                )
            except ValueError as e:
                pdb.set_trace()
            z_sample = encoder_distribution.rsample()
            
            # Manual formula for kl divergence (more numerically stable!)
            kl_div = 0.5 * (torch.exp(2 * logstd) + mu.pow(2) - 1.0 - 2 * logstd)
            kl_loss = kl_div.sum() / z_sample.shape[0]
        
        reconstruction_loss = self.decoder_loss(z_sample, x)

        # Final loss
        loss = reconstruction_loss + self.beta * kl_loss

        # Logging
        if self.logging_prefix is not None:
            self.log(
                f"rec/{self.logging_prefix}",
                reconstruction_loss,
                prog_bar=self.log_progress_bar,
            )
            self.log(
                f"kl/{self.logging_prefix}", kl_loss, prog_bar=self.log_progress_bar
            )
            self.log(f"loss/{self.logging_prefix}", loss)
        return loss

    def sample_prior(self, n_samples):
        return torch.distributions.Normal(self.prior_mu, self.prior_sigma).sample(
            torch.Size([n_samples])
        )
    
    def set_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Set normalization statistics for denormalization.
        
        Args:
            mean: Mean tensor used for normalization
            std: Standard deviation tensor used for normalization
        """
        self.data_mean = mean.to(self.device)
        self.data_std = std.to(self.device)
    
    def denormalize(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """
        Denormalize data back to original scale.
        
        Args:
            normalized_data: Normalized data tensor
            
        Returns:
            Denormalized data tensor
        """
        if self.data_mean is None or self.data_std is None:
            # No normalization was applied
            return normalized_data
        
        return normalized_data * self.data_std + self.data_mean
    
    def is_normalized(self) -> bool:
        """
        Check if the model has normalization enabled.
        
        Returns:
            True if normalization statistics are set, False otherwise
        """
        return self.data_mean is not None and self.data_std is not None

    def _increment_beta(self):

        if not self.beta_annealing:
            return

        # Check if the warmup is over and if it's the right step to increment beta
        if (
            self.global_step > self.hparams_.beta_warmup
            and self.global_step % self.hparams_.beta_step_freq == 0
        ):
            # Multiply beta to get beta proposal
            self.beta = min(self.hparams_.beta_final, self.beta * self.hparams_.beta_step)

    # Methods to overwrite (ones that differ between specific VAE implementations)
    def encode_to_params(self, x):
        """ encode a batch to it's distributional parameters """
        raise NotImplementedError

    def decoder_loss(self, z: torch.Tensor, x_orig) -> torch.Tensor:
        """ Get the loss of the decoder given a batch of z values to decode """
        raise NotImplementedError

    def decode_deterministic(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        self._increment_beta()
        self.log("beta", self.beta, prog_bar=True)

        self.logging_prefix = "train"
        loss = self(batch[0])
        self.logging_prefix = None
        return loss

    def validation_step(self, batch, batch_idx):
        self.logging_prefix = "val"
        self.log_progress_bar = True
        loss = self(batch[0])
        self.logging_prefix = None
        self.log_progress_bar = False
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams_.lr)


# class UnFlatten(torch.nn.Module):
#     """ unflattening layer """

#     def __init__(self, filters=1, size=28):
#         super().__init__()
#         self.filters = filters
#         self.size = size

#     def forward(self, x):
#         return x.view(x.size(0), self.filters, self.size, self.size)


class LinearVAE(BaseVAE):
    """ Linear VAE for encoding/decoding deisign pipelines """

    def __init__(self, hparams):
        super().__init__(hparams)

        # n_types = hparams.n_types
        # input_shape = (n_types, hparams.matrix_size, hparams.matrix_size)
        hidden_dims = hparams.hidden_dims.copy()
        self.input_size = hparams.input_size
        
        # Set reconstruction loss type: 'bernoulli' for binary data, 'mse' for continuous
        # Can be auto-detected or manually set via hparams
        self.reconstruction_loss_type = getattr(hparams, 'reconstruction_loss', 'auto')

        # Encoder
        modules = []
        input_size = self.input_size
        for h_dim in hidden_dims:
            modules.append(nn.Linear(input_size, h_dim))
            modules.append(nn.ReLU())
            input_size = h_dim  # Update input_size for the next layer
        modules.append(nn.Linear(hidden_dims[-1], 2 * self.latent_dim))
        self.encoder = nn.Sequential(*modules)

        # Decoder
        modules = [
            nn.Linear(self.latent_dim, hidden_dims[-1]),
            nn.ReLU()  # FIXED: Add ReLU after first layer
        ]
        for i in range(len(hidden_dims) - 1, 0, -1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_dims[0], self.input_size))
        self.decoder = nn.Sequential(*modules)


        # H_out ​= (H_in​−1)*stride[0] − 2×padding[0] + dilation[0]×(kernel_size[0]−1) + output_padding[0] + 1#
        summary(self.encoder.cuda(), (self.input_size,))
        summary(self.decoder.cuda(), (self.latent_dim,))

        self.last_valid_step = 0


    def encode_to_params(self, x):
        enc_output = self.encoder(x)
        mu, logstd = enc_output[:, : self.latent_dim], enc_output[:, self.latent_dim :]
        return mu, logstd


    def decoder_loss(self, z, x_orig):
        """
        Calculate reconstruction loss with automatic loss type detection.
        
        Supports:
        - 'bernoulli': For binary data (0s and 1s), uses Bernoulli log-likelihood
        - 'mse': For continuous data, uses Mean Squared Error
        - 'auto': Automatically detects based on data values
        """
        reconstruction = self.decoder(z)
        
        # Auto-detect loss type on first call if set to 'auto'
        if self.reconstruction_loss_type == 'auto':
            # Check if data is binary (all values are 0 or 1)
            unique_vals = torch.unique(x_orig)
            is_binary = len(unique_vals) <= 2 and torch.all((unique_vals == 0) | (unique_vals == 1))
            
            if is_binary:
                self.reconstruction_loss_type = 'bernoulli'
                print(f"🔍 Auto-detected binary data → Using Bernoulli loss")
            else:
                self.reconstruction_loss_type = 'mse'
                print(f"🔍 Auto-detected continuous data → Using MSE loss")
        
        # Calculate loss based on type
        if self.reconstruction_loss_type == 'bernoulli':
            # Bernoulli loss for binary data
            dist = distributions.Bernoulli(logits=reconstruction)
            return -dist.log_prob(x_orig).sum() / z.shape[0]
        
        elif self.reconstruction_loss_type == 'mse':
            # MSE loss for continuous data
            return F.mse_loss(reconstruction, x_orig, reduction='sum') / z.shape[0]
        
        else:
            raise ValueError(f"Unknown reconstruction_loss type: {self.reconstruction_loss_type}. "
                           f"Must be 'auto', 'bernoulli', or 'mse'.")
    
    # def decode_deterministic(self, z: torch.Tensor) -> torch.Tensor:
    #     logits = self.decoder(z)
    #     return torch.argmax(logits, dim=1)
    
    # def on_validation_epoch_end(self):
    #     # Visualize latent space
    #     self.visualize_latent_space(20)

    # def visualize_latent_space(self, nrow: int) -> torch.Tensor:

    #     # Currently only support 2D manifold visualization
    #     assert self.latent_dim >= 2

    #     # Create latent manifold
    #     unit_line = np.linspace(-4, 4, nrow)
    #     latent_grid = list(itertools.product(unit_line, repeat=2))
    #     latent_grid = np.array(latent_grid, dtype=np.float32)
    #     z_manifold = torch.as_tensor(latent_grid, device=self.device)

    #     if self.latent_dim > 2:
    #         z_manifold = torch.cat((z_manifold, torch.zeros((z_manifold.shape[0], self.latent_dim-2), device=self.device)), dim=1)

    #     # Decode latent manifold
    #     with torch.no_grad():
    #         img = self.decode_deterministic(z_manifold).detach().cpu()
    #     img = img.unsqueeze(1)

    #     # Make grid
    #     img = make_grid(img/4, nrow=nrow, padding=5, pad_value=0.5)

    #     # Sample robots
    #     num_sampled = 6
    #     r1, r2 = -4, 4
    #     z_manifold = (r1 - r2) * torch.rand(num_sampled, self.latent_dim, device=self.device) + r2
    #     codes = self.decode_deterministic(z_manifold).detach().cpu().numpy()
    #     sampled_imgs = [matrix2img(code) for code in codes]

    #     if not hasattr(self, "chosen_robot_code"):
    #         self.chosen_robot_code = gen_random_robot_draw(codes[0].shape[-1], np.array([1,2,3,4]))
    #     robot_code_tensor = torch.as_tensor(self.chosen_robot_code, dtype=torch.long, device=self.device).unsqueeze(0)
    #     robot_code_tensor = torch.nn.functional.one_hot(robot_code_tensor, num_classes=-1)
    #     robot_code_tensor = torch.permute(robot_code_tensor, (0,3,1,2)).float()
    #     mu, _ = self.encode_to_params(robot_code_tensor)
    #     robot_code_decoded  = self.decode_deterministic(mu).detach().cpu().numpy()[0]
    #     robot_photo_before = matrix2img(self.chosen_robot_code, "temp")
    #     robot_photo_after = matrix2img(robot_code_decoded, "temp")
    #     self.logger.experiment.log({"Latent Manifold": [wandb.Image(img, caption="latent manifold")],
    #                                 "Sampled Robots (z)": [wandb.Image(sampled_imgs[i], caption=f"sampled robots {i}") for i in range(num_sampled)],
    #                                 "Sampled Robots (x)": [wandb.Image(robot_photo_before, caption="before VAE"), wandb.Image(robot_photo_after, caption="after VAE")]
    #                                 })
        
    #     self.last_valid_step = self.global_step
