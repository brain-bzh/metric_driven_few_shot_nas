import abc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import typing as typ
from copy import deepcopy
from easydict import EasyDict as EDict

from ..lib.models import get_cell_based_tiny_net
from ..utils.init_utils import init_model


class Collector:
    """
    Collector is an object tasked with passing data through models to collect a certain value related to weights or gradients.
    Typically used in Zero-Shot NAS algorithms.
    """

    def __init__(self, data_loader, n_batches, model_config = EDict(), arch_parameters=[], init='normal', seed=0):
        """
        Args:
            data_loader (_type_): typically a torch iterable data loader. Note that batch size is defined at the level of the data loader.
            n_batches (_type_): number of batches to pass through during collector forward passes.
            model_config (_type_, optional): how to instantiate model (search space, classes, block_type, etc)
            arch_parameters (list, optional): Architecture mask. Defaults to [].
            init (str, optional): weight initialization of network. Defaults to 'normal'.
            seed (int, optional): torch seed Defaults to 0.
        """
        # TODO why deep copy?
        self.data_loader = deepcopy(data_loader)
        self.train_loader = iter(data_loader)

        self.model_config = model_config
        self.arch_parameters = arch_parameters
        self.init = init

        self.batch_size = data_loader.batch_size
        self.n_networks = len(self.arch_parameters)

        self.n_batches = n_batches

        self.networks = []

        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        torch.cuda.empty_cache()

    def set_arch_parameters(self, arch_parameters: list):
        """Setter for architecture parameters

        Args:
            arch_parameters (list): list of architecture parameters for each network

        Note:
            Also used to re-initialize network attribute to an empty list
        """
        assert isinstance(arch_parameters, list)
        del self.arch_parameters
        del self.networks

        self.arch_parameters = arch_parameters
        self.n_networks = len(arch_parameters)
        self.networks= []

    def set_model_config(self, model_config: typ.Union[EDict, dict]):
        """Setter for model config

        Args:
            model_config (typ.Union[EDict, dict]): single config to instantiate all models from
        """
        assert isinstance(model_config, EDict) or isinstance(model_config, dict)
        del self.model_config
        if isinstance(model_config, dict):
            self.model_config = EDict(model_config)
        else:
            self.model_config = model_config

    def set_data_loader(self, data_loader: DataLoader):
        del self.data_loader
        del self.train_loader
        self.data_loader = deepcopy(data_loader)
        self.train_loader = iter(self.data_loader)
        self.batch_size = data_loader.batch_size

    def set_n_batches(self, n_batches):
        self.n_batches = n_batches

    def set_seed(self, seed):
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def clear(self):
        """
        Some collectors may use a clear function to reset internal values and free up memory space.
        Call after running forward passes.

        Note:
        recommended not to use `torch.cuda.empty_cache()_<https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/5>`
        """

        del self.networks
        self.networks = []
        torch.cuda.empty_cache()

    def instantiate_networks(self, loaded_weights=None):
        """
            Create n_networks number of networks, initialize weights, and initialize architecture parameters. 
            First network initialize weights from init, the other networks copy the same initialization. 

            Note:
            net.set_cal_mode('joint') is important for NasBench and sets the forward pass strategy.
            arch_parameters is used in the forward pass 
        """
        
        self.networks = [get_cell_based_tiny_net(self.model_config).cuda() for net in range(self.n_networks)]
        #All model parameters are initialized to be the same except for architecture parameters (this makes comparisons less susceptible to random initialization)
        for k, net in enumerate(self.networks):
            net.set_cal_mode('joint')
            if loaded_weights==None:
                if k==0:
                    init_model(net, self.init)
                else:
                    for param, _param in zip(self.networks[0].parameters(), net.parameters()):
                        _param.data.copy_(param.data)
            else:
                for param, _param in zip(loaded_weights, net.parameters()):
                    _param.data.copy_(param.data)

            # TODO why? take architecture params, convert to parameter, and detach and put on cpu.
            # detatch because no gradients. Its more of a mask.
            net.arch_parameters = nn.Parameter(torch.Tensor(deepcopy(self.arch_parameters[k])).cuda().detach().clone().cpu())
            net.cuda()

    def get_inputs(self, return_targets: bool = False):
        """
        Generator that returns 1 batch from the dataloader. 
        Automatically regenerates and loops back to the beginning of the loader once all batches have been passed.

        Args:
            return_targets (bool, optional): wether or not to return targets along with data. Defaults to False.
        """

        try:
            inputs, targets = next(self.train_loader)
        except Exception:
            del self.train_loader
            self.train_loader = iter(self.data_loader)
            inputs, targets = next(self.train_loader)

        if return_targets:
            return(inputs, targets)
        else:
            return(inputs)

    @abc.abstractmethod
    def forward_batches(self):
        raise NotImplementedError