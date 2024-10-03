import torch
import numpy as np
import random
from config import Config
config = Config()

def fix_random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

import time
def cleanup_hardware(p):
    print("Performing clean-up...")
    p.signal.set()
    time.sleep(3)
    print("Clean-up finished.")

import libemg
def setup_live_processing():
    p, smi = config.streamer_handle(**config.streamer_arguments)
    odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=smi)
    if config.log_to_file:
        odh.log_to_file()
    
    return p, odh, smi

