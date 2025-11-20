import argparse
import os
from pathlib import Path
import pdb
import pickle
import random
from typing import Union, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import pytorch_lightning as pl

from twist_controller import CONTROLLER_ROOT_DIR
from twist_controller.sim.evolution.encoding_wrapper import to_onehot
from twist_controller.sim.evolution.vae import utils

NUM_WORKERS = 3


class WeightedDataset(pl.LightningDataModule):
    """
    A flexible PyTorch Lightning DataModule for weighted datasets with automatic format detection.
    
    This class supports multiple input formats:
    1. File path (str or Path) - automatically detects .pt, .npz, or .pkl formats
    2. Tuple (data, properties) - data array with corresponding properties
    3. Dictionary with 'data' and optional 'properties' keys
    4. Direct tensor/array input
    5. Legacy hparams object (backward compatible)
    
    Examples:
        # From file path
        >>> dataset = WeightedDataset("data.pt", batch_size=128, val_frac=0.1)
        
        # From tuple (data, properties)
        >>> data = np.random.randn(1000, 128)
        >>> scores = np.random.rand(1000)
        >>> dataset = WeightedDataset((data, scores), batch_size=64)
        
        # From numpy array only (unweighted)
        >>> data = np.random.randn(1000, 128)
        >>> dataset = WeightedDataset(data, batch_size=64)
        
        # From dictionary with properties
        >>> dataset = WeightedDataset(
        ...     {"data": data_array, "properties": scores},
        ...     property_key="score",
        ...     rank_weight_k=1e-3
        ... )
        
        # Legacy compatibility (hparams object)
        >>> dataset = WeightedDataset(hparams)
    """

    def __init__(
        self, 
        data: Union[str, Path, tuple, np.ndarray, torch.Tensor, Dict[str, Any], Any],
        properties: Optional[np.ndarray] = None,
        batch_size: int = 128,
        val_frac: float = 0.05,
        property_key: str = "score",
        weight_type: str = "rank",
        rank_weight_k: float = 1e-3,
        weight_quantile: Optional[float] = None,
        num_workers: int = NUM_WORKERS,
        normalize: bool = False,
        # Legacy support - if cfg/hparams object is passed as first arg
        **kwargs
    ):
        """
        Initialize WeightedDataset with flexible input formats.
        
        Args:
            data: Can be one of:
                - str/Path: File path to .pt, .npz, or .pkl file
                - tuple: (data_array, properties_array) - data with properties
                - np.ndarray/torch.Tensor: Direct data array
                - dict: Dictionary with 'data' and optional 'properties' keys
                - object: Legacy hparams/cfg object (backward compatible)
            properties: Optional array of properties/scores for weighting (if data is array)
            batch_size: Batch size for dataloaders
            val_frac: Fraction of data to use for validation
            property_key: Key for properties in npz files
            weight_type: Type of weighting ('rank', 'fb', etc.)
            rank_weight_k: Rank weighting parameter
            weight_quantile: Quantile for filtering (used in fb weighting)
            num_workers: Number of dataloader workers
            normalize: If True, normalizes data to zero mean and unit variance
                      Stores statistics for denormalization
            **kwargs: Additional parameters
        """
        super().__init__()
        
        # Detect input type and extract configuration
        self._input_data = data
        self._input_properties = properties
        
        # Check if legacy hparams object (has dataset_path attribute)
        if hasattr(data, 'dataset_path'):
            # Legacy mode: extract from hparams
            self._legacy_mode = True
            self.cfg = data
            self.dataset_path = data.dataset_path
            self.batch_size = data.batch_size
            self.val_frac = data.val_frac
            self.property_key = data.property_key
            self.weight_type = getattr(data, 'weight_type', weight_type)
            self.rank_weight_k = getattr(data, 'rank_weight_k', rank_weight_k)
            self.weight_quantile = getattr(data, 'weight_quantile', weight_quantile)
            self.num_workers = num_workers
            self.normalize = getattr(data, 'normalize', normalize)
        else:
            # New flexible API mode
            self._legacy_mode = False
            self.dataset_path = data if isinstance(data, (str, Path)) else None
            self.batch_size = batch_size
            self.val_frac = val_frac
            self.property_key = property_key
            self.weight_type = weight_type
            self.rank_weight_k = rank_weight_k
            self.weight_quantile = weight_quantile
            self.num_workers = num_workers
            self.normalize = normalize
            
            # Create a minimal cfg object for compatibility with utils.DataWeighter
            self.cfg = type('Config', (), {
                'weight_type': self.weight_type,
                'rank_weight_k': self.rank_weight_k,
                'weight_quantile': self.weight_quantile,
            })()
        
        # Normalization statistics (computed during setup)
        self.data_mean = None
        self.data_std = None
        

    def prepare_data(self):
        pass

    def _load_from_path(self, path: Union[str, Path]) -> tuple[torch.Tensor, np.ndarray]:
        """
        Load data from file path with automatic format detection.
        
        Args:
            path: File path to load (.pt, .npz, or .pkl)
            
        Returns:
            Tuple of (data, properties) as (torch.Tensor, np.ndarray)
        """
        path_str = str(path)
        
        if path_str.endswith(".pt"):
            all_data = torch.load(path_str, map_location='cpu')
            all_properties = np.ones(all_data.shape[0])
            
        elif path_str.endswith(".npz"):
            with np.load(path_str) as npz:
                all_data = npz["data"]
                all_data = torch.from_numpy(all_data)
                
                if self.property_key in npz:
                    all_properties = npz[self.property_key]
                else:
                    all_properties = np.ones(all_data.shape[0])
                    self.cfg.rank_weight_k = np.inf
                    print("No properties found. Using unweighted pre-training.")
                    
        elif path_str.endswith(".pkl"):
            with open(path_str, "rb") as f:
                all_data = pickle.load(f)
                self.sampled_original_data = random.sample(all_data, min(10, len(all_data)))
                
                assert isinstance(all_data, list), "Pickle data must be a list."
                self.max_idx = max(max(sublist) for sublist in all_data)
                self.max_length = max(len(sublist) for sublist in all_data)
                print(f"Loaded pickle data with max_idx: {self.max_idx}, max_length: {self.max_length}")
                
                # Convert to one-hot encoding
                all_data = to_onehot(all_data, self.max_idx, self.max_length)
                all_data = torch.from_numpy(all_data) if isinstance(all_data, np.ndarray) else all_data
                all_properties = np.ones(all_data.shape[0])
                self.cfg.rank_weight_k = np.inf
                print("Using unweighted pre-training for pickle data.")
        else:
            raise ValueError(f"Unsupported file format: {path_str}. Supported: .pt, .npz, .pkl")
            
        return all_data, all_properties
    
    def _load_from_array(
        self, 
        data: Union[np.ndarray, torch.Tensor], 
        properties: Optional[np.ndarray] = None
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Load data from numpy array or torch tensor.
        
        Args:
            data: Input data array
            properties: Optional properties array
            
        Returns:
            Tuple of (data, properties) as (torch.Tensor, np.ndarray)
        """
        # Convert to torch tensor if needed
        if isinstance(data, np.ndarray):
            all_data = torch.from_numpy(data)
        else:
            all_data = data
            
        # Handle properties
        if properties is not None:
            all_properties = properties if isinstance(properties, np.ndarray) else np.array(properties)
        else:
            all_properties = np.ones(all_data.shape[0])
            self.cfg.rank_weight_k = np.inf
            print("No properties provided. Using unweighted training.")
            
        return all_data, all_properties
    
    def _load_from_dict(self, data_dict: Dict[str, Any]) -> tuple[torch.Tensor, np.ndarray]:
        """
        Load data from dictionary format.
        
        Args:
            data_dict: Dictionary with 'data' key and optional 'properties' key
            
        Returns:
            Tuple of (data, properties) as (torch.Tensor, np.ndarray)
        """
        if 'data' not in data_dict:
            raise ValueError("Dictionary must contain 'data' key")
            
        data = data_dict['data']
        properties = data_dict.get('properties', None)
        
        return self._load_from_array(data, properties)

    def setup(self, stage):
        """Load and prepare data based on input type."""
        # Determine how to load the data
        if self._legacy_mode:
            # Legacy mode: use dataset_path
            all_data, all_properties = self._load_from_path(self.dataset_path)
        elif self.dataset_path is not None:
            # New API with path
            all_data, all_properties = self._load_from_path(self.dataset_path)
        elif isinstance(self._input_data, tuple) and len(self._input_data) == 2:
            # Tuple input: (data, properties)
            data_array, properties = self._input_data
            all_data, all_properties = self._load_from_array(data_array, properties)
        elif isinstance(self._input_data, dict):
            # Dictionary input
            all_data, all_properties = self._load_from_dict(self._input_data)
        elif isinstance(self._input_data, (np.ndarray, torch.Tensor)):
            # Array input
            all_data, all_properties = self._load_from_array(self._input_data, self._input_properties)
        else:
            raise ValueError(f"Unsupported data input type: {type(self._input_data)}")

        assert all_properties.shape[0] == all_data.shape[0]
        self.data_shape = all_data.shape

        # Compute normalization statistics from training data BEFORE splitting
        # This ensures we normalize based on training distribution
        N_val = int(all_data.shape[0] * self.val_frac)
        train_data_for_stats = all_data[N_val:]
        
        if self.normalize:
            # Compute statistics on training data only
            self.data_mean = train_data_for_stats.mean(dim=0, keepdim=True)
            self.data_std = train_data_for_stats.std(dim=0, keepdim=True)
            
            # Avoid division by zero (for constant features)
            self.data_std = torch.clamp(self.data_std, min=1e-8)
            
            # Normalize all data
            all_data = (all_data - self.data_mean) / self.data_std
            print(f"✓ Data normalized: mean={self.data_mean.mean().item():.4f}, std={self.data_std.mean().item():.4f}")
        else:
            # No normalization - store None
            self.data_mean = None
            self.data_std = None

        # Split into train/val after normalization
        self.data_val = all_data[:N_val]
        self.prop_val = all_properties[:N_val]
        self.data_train = all_data[N_val:]
        self.prop_train = all_properties[N_val:]

        # Make into tensor datasets
        self.train_dataset = TensorDataset(self.data_train)
        self.val_dataset = TensorDataset(self.data_val)


        self.data_weighter = utils.DataWeighter(self.cfg)
        self.set_weights()

    def set_weights(self):
        """ sets the weights from the weighted dataset """

        # Make train/val weights
        self.train_weights = self.data_weighter.weighting_function(self.prop_train)
        self.val_weights = self.data_weighter.weighting_function(self.prop_val)

        # Create samplers
        self.train_sampler = WeightedRandomSampler(
            self.train_weights, num_samples=len(self.train_weights), replacement=True
        )
        self.val_sampler = WeightedRandomSampler(
            self.val_weights, num_samples=len(self.val_weights), replacement=True
        )

    def append_train_data(self, x_new, prop_new):

        # Special adjustment for fb-vae: only add the best points
        if self.data_weighter.weight_type == "fb":

            # Find top quantile
            cutoff = np.quantile(prop_new, self.data_weighter.weight_quantile)
            indices_to_add = prop_new >= cutoff

            # Filter all but top quantile
            x_new = x_new[indices_to_add]
            prop_new = prop_new[indices_to_add]
            assert len(x_new) == len(prop_new)

            # Replace data (assuming that number of samples taken is less than the dataset size)
            x_new_tensor = torch.from_numpy(x_new) if isinstance(x_new, np.ndarray) else x_new
            self.data_train = torch.cat(
                [self.data_train[len(x_new):], x_new_tensor], axis=0
            )
            self.prop_train = np.concatenate(
                [self.prop_train[len(x_new):], prop_new], axis=0
            )
        else:

            # Normal treatment: just concatenate the points
            self.data_train = torch.cat([self.data_train, torch.from_numpy(x_new)], axis=0)
            self.prop_train = np.concatenate([self.prop_train, prop_new], axis=0)
        self.train_dataset = TensorDataset(self.data_train)
        self.set_weights()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.train_sampler,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.val_sampler,
            drop_last=True,
        )
    
    def get_normalization_stats(self):
        """
        Get normalization statistics for denormalization.
        
        Returns:
            dict: Dictionary with 'mean' and 'std' tensors, or None if not normalized
        """
        if self.normalize and self.data_mean is not None and self.data_std is not None:
            return {
                'mean': self.data_mean,
                'std': self.data_std
            }
        return None
