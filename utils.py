
import os
import yaml
import logging
import random
import numpy as np
import torch
import sys
from datetime import datetime

def load_config(config_path):
    """Load YAML config file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(config):
    """
    Setup logging to file and console.
    Structure: logs/{experiment_name}/{timestamp}.log
    """
    exp_name = config.get("experiment_name", "experiment")
    log_root = config.get("paths", {}).get("log_dir", "logs")
    
    # Create experiment directory
    exp_dir = os.path.join(log_root, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(exp_dir, f"{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging configured. Saving logs to: {log_file}")
    
    # Return path to save checkpoints nearby if needed
    ckpt_root = config.get("paths", {}).get("checkpoint_dir", "checkpoints")
    ckpt_dir = os.path.join(ckpt_root, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    return ckpt_dir

def seed_everything(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logging.info(f"Seeded everything with seed: {seed}")
