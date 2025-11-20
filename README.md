# CapyVAE

**Status:** 🚧 Under Active Development

A simple, flexible VAE (Variational Autoencoder) library for design optimization.

Originally developed for the [ModularLegs project](https://modularlegs.github.io/) by Chen Yu.

## Part of the Modular Embodied AI Ecosystem

This project is part of an ongoing development of a modular embodied AI ecosystem. Related projects include:

- **[metamachine](https://github.com/Chenaah/metamachine)** - A simulation framework for modular robots.
- **[capybarish](https://github.com/Chenaah/capybarish)** - A lightweight communication middleware.
- **[CapyRL](https://github.com/Chenaah/CapyRL)** - A lightweight JAX-based reinforcement learning library.
- **[CapyFormer](git@github.com:Chenaah/CapyFormer.git)** - A simple, flexible Transformer library for locomotion control.

---

## Installation

```bash
git clone https://github.com/Chenaah/CapyVAE.git
cd CapyVAE
pip install -e .
```

---

## Quick Start

### Basic Usage

```python
from capyvae import VAE

# Train a VAE from a data file
vae = VAE(dataset="data.pt", latent_dim=8, max_epochs=50)
vae.train()
```

### With Custom Data

```python
import numpy as np

# Your data and scores
data = np.random.randn(1000, 128).astype(np.float32)
scores = np.random.rand(1000)

# Train VAE with weighted sampling (emphasizes high scores)
vae = VAE(
    dataset=(data, scores),  # Pass as tuple
    latent_dim=16,
    max_epochs=100,
    weight_type="rank",
    rank_weight_k=1e-3
)
vae.train_vae()
```

### Design Optimization

```python
# Train with black-box optimizer for design optimization
vae = VAE(
    dataset=(data, scores),
    latent_dim=8,
    optimizer="abo",
    opt_bounds=(-4, 4)
)
vae.train_vae()

# Generate initial designs
init_designs = vae.get_init_designs()

# Get new optimized designs
new_design = vae.get_new_design()
```

---

## Dataset Support

CapyVAE supports multiple dataset formats:

- **File paths**: `.pt`, `.npz`, `.pkl` files
- **NumPy arrays**: `np.ndarray`
- **PyTorch tensors**: `torch.Tensor`
- **Tuples**: `(data, properties)` for weighted sampling
- **Dictionaries**: `{"data": array, "properties": scores}`


---

## Contact

Chen Yu - [GitHub](https://github.com/Chenaah)
