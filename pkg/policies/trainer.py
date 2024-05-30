import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy

from ..lib.models import get_cell_based_tiny_net
from ..utils.init_utils import init_model

class Trainer:

    def __init__(self, model_config, cal_mode, init, training_scheme, **train_args):

        self.model_config = model_config
        self.cal_mode = cal_mode
        self.init = init

        self.training_scheme = training_scheme
        self.train_args = train_args

    def instantiate_network(self, arch_parameters, loaded_weights=None):

        self.network = get_cell_based_tiny_net(self.model_config)
        self.network.set_cal_mode(self.cal_mode)

        if loaded_weights==None:
            init_model(self.network, self.init)

        else:
            for param, _param in zip(loaded_weights, self.network.parameters()):
                _param.data.copy_(param.data)

        self.network.arch_parameters = nn.Parameter(torch.Tensor(deepcopy(arch_parameters)).cuda().detach().clone().cpu())
        self.network.cuda()
        
    def train_network(self, n_epochs):

        loaded_weights = self.training_scheme(network=self.network, n_epochs=n_epochs, **self.train_args)

        return(loaded_weights)
