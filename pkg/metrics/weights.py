import numpy as np
import torch

from copy import deepcopy
from easydict import EasyDict as EDict

from .collector import Collector

class WeightsCollector(Collector):

    def __init__(self, data_loader, n_batches, model_config=EDict(), arch_parameters=[], init='normal', seed=0, train_mode=False):

        super().__init__(data_loader, n_batches, model_config=model_config, arch_parameters=arch_parameters, init=init, seed=seed)
        self.train_mode = train_mode

        self.weights = [[] for _ in range(self.n_networks)]

        torch.cuda.empty_cache()

    #Setters
    def set_arch_parameters(self, arch_parameters):

        super().set_arch_parameters(arch_parameters)

        del self.weights
        self.weights = [[] for _ in range(self.n_networks)]

    def set_train_mode(self, train_mode=True):
        self.train_mode = train_mode

    def clear(self):

        super().clear()

        del self.weights
        self.weights = [[] for _ in range(self.n_networks)]

    def instantiate_networks(self, loaded_weights=None):

        super().instantiate_networks(loaded_weights=loaded_weights)

        for net in self.networks:
            net.cuda().train(self.train_mode)

    #Batches are passed forward n_batches times across all models
    def forward_batches(self):

        device = torch.cuda.current_device()

        for _ in range(self.n_batches):

            inputs, targets = self.get_inputs()

            _inputs = inputs.clone().cuda(device=device, non_blocking=True)

            for net, w in zip(self.networks, self.weights):
                self.forward(net, w, _inputs)
            
            del inputs 
            del _inputs

        return [w for w in self.weights]

    #For a single model, weights are queried
    def forward(self, net, w, input):
        net.zero_grad()
        for layer in net.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                weights = layer.weight
                if weights.data is not None:
                    w.append(weights.data.view(-1))
                else:
                    w.append(torch.zeros_like(weights).view(-1))
        torch.cuda.empty_cache()