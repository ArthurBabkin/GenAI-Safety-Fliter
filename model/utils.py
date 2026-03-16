"""
Utility functions for safety filter model training and evaluation.
"""

import random
import numpy as np


def seed_everything(seed: int = 42):
    """
    Set random seeds for full reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
