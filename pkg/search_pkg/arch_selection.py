
import torch
import logging
import numpy as np
from copy import deepcopy
from .procedures.eval_funcs import obtain_accuracy
from .procedures.metric_utils import AverageMeter
from .utils import load_pickle, NASBENCH_PATH
from .training import supernet_training, supernet_validation_step
import functools
import itertools

logger = logging.getLogger(__name__)

def distill(result, dataset):
    result = result.split('\n')
    if dataset == "cifar10":
        dat = result[5].replace(' ', '').split(':')
        return float(dat[2][-7:-2].strip('='))
    elif dataset =="cifar100":
        dat = result[7].replace(' ', '').split(':')
    elif dataset == "ImageNet16-120":
        dat = result[9].replace(' ', '').split(':')
    else:
        raise ValueError('invalid dataset') 
    return  float(dat[3][-7:-2].strip('='))


def optimal_path(network, api, dataset, mask):
    genotype, arch_path, arch_parameter_form = network.genotype()
    performance = api.query_by_arch(genotype, hp="200")
    results = distill(performance, dataset)
    return arch_path, arch_parameter_form, results

def stats(df, index):
        r = df['mean'].rank(axis=0)[index]
        n = df.shape[0]
        p =  r / (n + 1)
        eff = 1/(1-p)
        return {"rank": r, "size":n, "efficiency": eff}

def convert_masked_to_space(arr):
    convert = np.tile(np.arange(arr.shape[1]), (arr.shape[0], 1)) + 1
    masked = arr*convert
    masked = masked.astype(int)
    ragged_masked = [masked[i][masked[i] != 0]-1 for i in range(masked.shape[0])]

    return list(itertools.product(*ragged_masked))