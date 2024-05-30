import numpy as np
import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *

import random
from copy import deepcopy

from .init_utils import init_model

from ..lib.models import get_cell_based_tiny_net

INF = 1000

def make_subsupernet(model_config, arch_parameters = None, initial_params_path=None, load_arch_from_params=False, load_weights_from_params=False, cuda=True):

    if load_weights_from_params and initial_params_path is not None:
        if cuda:
            device = torch.device(torch.cuda.current_device())
        else:
            device = torch.device('cpu')
        model = torch.load(initial_params_path, map_location=device)
    else:
        model = get_cell_based_tiny_net(model_config)
        if cuda:
            model.cuda()
        init_model(model)
    if not load_arch_from_params or initial_params_path is None:
        if cuda:
            model.arch_parameters = nn.Parameter(torch.Tensor(deepcopy(arch_parameters)).cuda().detach().clone().cpu())
        else:
            model.arch_parameters = nn.Parameter(torch.Tensor(deepcopy(arch_parameters)).detach().clone())
    return(model)

def sample_arch(arch_parameters, seed=0):

    random.seed(seed)
    c_ops = []
    for e in range(arch_parameters.shape[0]):

        open_ops = []
        for o in range(arch_parameters.shape[1]):
            if arch_parameters[e,o] >= 0:
                open_ops.append(o)
        c_ops.append(random.choice(open_ops))

    arch_params_ = nn.Parameter(torch.full_like(arch_parameters, -INF)).requires_grad_(False)
    for i, op in enumerate(c_ops):
        arch_params_[i,op] = 0

    del c_ops, open_ops

    return(arch_params_)

###################################################### Optim Wrappers ##############################################################

class OptimWrapper():

    def __init__(self):
        pass

    def make_scheduler(self):
        pass

class AdamOptim(OptimWrapper):

    def __init__(self, lr=0.001, betas=(0.9, 0.999), weight_decay=0):

        self.optimizer = torch.optim.Adam
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

    def make_optimizer(self, params):

        return(self.optimizer(params, lr=self.lr, betas=self.betas, weight_decay=self.weight_decay))

class SGDOptim(OptimWrapper):

    def __init__(self, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):

        self.optimizer = torch.optim.SGD
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def make_optimizer(self, params):

        return(self.optimizer(params, lr=self.lr, momentum=self.momentum, dampening=self.dampening, weight_decay=self.weight_decay, nesterov=self.nesterov))


###################################################### Schedule Wrappers ###########################################################

class ScheduleWrapper():

    def __init__(self):
        pass

    def make_scheduler(self):
        pass

class StepScheduler(ScheduleWrapper):

    def __init__(self, step_size, gamma=0.1, last_epoch=-1):

        self.scheduler = StepLR
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch

    def make_scheduler(self, optimizer):

        return(self.scheduler(optimizer, step_size=self.step_size, gamma=self.gamma, last_epoch=self.last_epoch))

class LinearScheduler(ScheduleWrapper):

    def __init__(self, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1):

        self.scheduler = LinearLR
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        self.last_epoch = last_epoch

    def make_scheduler(self, optimizer):

        return(self.scheduler(optimizer, start_factor=self.start_factor, end_factor=self.end_factor, total_iters=self.total_iters, last_epoch=self.last_epoch))

class ConstantScheduler(ScheduleWrapper):

    def __init__(self, factor=1.0 / 3, total_iters=5, last_epoch=-1):

        self.scheduler = ConstantLR
        self.factor = factor
        self.total_iters = total_iters
        self.last_epoch = last_epoch

    def make_scheduler(self, optimizer):
        return(self.scheduler(optimizer, factor=self.factor, total_iters=self.total_iters, last_epoch=self.last_epoch))

class CosineAnnealingScheduler(ScheduleWrapper):

    def __init__(self, T_0, T_mult=1, eta_min=0, last_epoch=-1):

        self.scheduler = CosineAnnealingWarmRestarts
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.last_epoch = last_epoch

    def make_scheduler(self, optimizer):
        return(self.scheduler(optimizer, T_0=self.T_0, T_mult=self.T_mult, eta_min=self.eta_min, last_epoch=self.last_epoch))