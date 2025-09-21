import numpy as np
import torch
import random

def set_seed(seed, cuda=False):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def str2bool(v):
    return v.lower() in ('true', '1', 'yes')
