
import logging
from .ravdess import get_ravdess_dataloaders
from .iemocap_hf import get_hf_dataloaders
# from .iemocap_local import get_iemocap_dataloaders # ready if needed

def get_dataloaders(config):
    """
    Factory method to get dataloaders based on config.
    """
    ds_config = config.get('dataset', {})
    name = ds_config.get('name', '').lower()
    
    batch_size = ds_config.get('args', {}).get('batch_size', 16)
    num_workers = ds_config.get('args', {}).get('num_workers', 4)
    hf_id = ds_config.get('args', {}).get('hf_id', '')
    root_dir = ds_config.get('args', {}).get('root_dir', '') # For local

    logging.info(f"Factory initializing dataset: {name}")

    if name == 'ravdess':
        # RAVDESS defaults
        if not hf_id: hf_id = "TwinkStart/RAVDESS"
        return get_ravdess_dataloaders(hf_id=hf_id, batch_size=batch_size, num_workers=num_workers)
    
    elif name == 'iemocap':
        # IEMOCAP HF defaults
        if not hf_id: hf_id = "AbstractTTS/IEMOCAP"
        return get_hf_dataloaders(hf_id=hf_id, batch_size=batch_size, num_workers=num_workers)
    
    else:
        raise ValueError(f"Unknown dataset name: {name}. Supported: ravdess, iemocap")
