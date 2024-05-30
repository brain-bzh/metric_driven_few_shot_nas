import os
import sys
import PIL
import logging
import torch
import pathlib
import argparse
import yaml
import pickle
from functools import wraps
import time
import numpy as np
from .models.cell_searchs.search_cells import node_str_dict, NAS201SearchCell as SearchCell
import torch.distributed as dist

from ..utils.paths import *

logger = logging.getLogger(__name__)
        
def path_exist(*paths):
    for path in paths:
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print(f"The new directory is created: {path}")
  

def count_parameters(model):
  """
  Count parameters in a model
  """
  total_params = 0
  for name, parameter in model.named_parameters():
      if not parameter.requires_grad: continue
      param = parameter.numel()
      total_params += param
  return total_params


def create_logger(savepath, pathtime):
  if not is_main_process():
    logger = Dummylogger()
  else:
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(savepath, f'{pathtime}.log'), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-15s: %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logger.addHandler(console)

    logger.info("Main Function with logger : {:}".format(logger))
    logger.info("Python  Version  : {:}".format(sys.version.replace("\n", " ")))
    logger.info("Pillow  Version  : {:}".format(PIL.__version__))
    logger.info("PyTorch Version  : {:}".format(torch.__version__))

    logger.info("cuDNN   Version  : {:}".format(torch.backends.cudnn.version()))
    logger.info("CUDA available   : {:}".format(torch.cuda.is_available()))
    logger.info("CUDA GPU numbers : {:}".format(torch.cuda.device_count()))
    logger.info(
        "CUDA_VISIBLE_DEVICES : {:}".format(
            os.environ["CUDA_VISIBLE_DEVICES"]
            if "CUDA_VISIBLE_DEVICES" in os.environ
            else "None"
        )
    )

def str2bool(v):
    """Argparse sees "True" or "False" as both bool(True) since they are strings.
    This is to avoid this behavior when using a config file in vscode.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def flat_concat(xs):
    return torch.cat([x.reshape(-1) for x in xs])
    
def load_yaml(file_name):
    try:
        with open(os.path.join(SEARCH_CONFIGS_PATH, file_name), 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')
    
    return config

def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield (key, value)
            yield from recursive_items(value)
        else:
            yield (key, value)

def save_yaml(args, path):
    with open(os.path.join(path, f"config.yml"), 'w') as file:
        yaml.dump(
                args,
                file,
                default_flow_style=False,
            )

def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def timer(f):
    @wraps(f)
    def aux(*args, **kwargs):
        start_time = time.time()
        output = f(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f'Execution time {f.__name__}: {execution_time:.4f} seconds')
        return output
    return aux

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6

# NASBENCH UTILS

# get string of operations from NasBench format
def get_operations(arch):
    edges = arch.split('+')
    ops = []
    for e in edges:
        ops_sub = e[1:-1].split('|')
        for op in ops_sub:
            ops.append(op[:-2])
    return ops

# transform architectures from Nasbench format to one hot encoded representations
I = np.eye(5)
matching = {'none': 0, 'skip_connect': 1, 'nor_conv_1x1': 2, 'nor_conv_3x3': 3, 'avg_pool_3x3': 4}

def transform(arch):
    op_list = get_operations(arch)
    matrix = np.zeros((6, 5))
    for i, op in enumerate(op_list):
        representation = I[matching[op]]
        matrix[i,:] = representation
    return matrix

# NTK
def ntk(network, optimizer, architecture, data):
    optimizer.zero_grad()
    architecture.optimizer.zero_grad()
    full_grad = []
    for (inputs, targets) in data:
        inputs = inputs.cuda()
        logits = network(inputs)
        logits.backward(torch.ones_like(logits))
        param_collector = []
        for _, cell in enumerate(network._cells):
            if isinstance(cell, SearchCell):
                for e in range(len(cell.edges)):
                    param_collector += list(cell.edges[node_str_dict[e]].parameters())
            else:
                param_collector += list(cell.parameters())
        grads = [param.grad for param in param_collector if param.grad != None]
        grads_detached = [g.clone().detach() for g in grads] 
        full_grad.append(flat_concat(grads_detached))
    jacob = torch.stack(full_grad)
    kernel = torch.matmul(jacob, jacob.T)
    return kernel

class Dummylogger(logging.RootLogger):
    def __init__(self, level=30):
        super(Dummylogger, self).__init__(level=30)
    def info(self, msg, *args, **kwargs):
        return None

class SecondaryWriter:
    def add_scalar(self, *args):
        return None
    def flush(self):
        return None
    
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):

    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():

    return get_rank() == 0