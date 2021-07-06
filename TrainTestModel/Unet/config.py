
import argparse
import sys

import torch
import yaml


# logger = utils.get_logger(__name__)

def load_config(yaml_conf_path):
    
    config = yaml.safe_load(open(yaml_conf_path, 'r'))

    # Get a device to train on
    device_str = config.get('device', None)
    if device_str is not None:
        # logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            sys.exit('CUDA not available')
            # logger.warn('CUDA not available, using CPU')
            # device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        # logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device
    return config
