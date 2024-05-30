import numpy as np
import torch

from copy import deepcopy
from easydict import EasyDict as EDict

from .collector import Collector

## Re-implementation of original code by W. Chen

class GradientCollector(Collector):
    """
        Passes data through models and collects gradients. 
    """

    def __init__(self, data_loader, n_batches, model_config = EDict(), arch_parameters=[], init='normal', seed=0, train_mode=False):
        super().__init__(data_loader, n_batches, model_config=model_config, arch_parameters=arch_parameters, init=init, seed=seed)
        self.train_mode = train_mode

        self.grads = [[] for _ in range(self.n_networks)]

        torch.cuda.empty_cache()

    #Setters
    def set_arch_parameters(self, arch_parameters):

        super().set_arch_parameters(arch_parameters)

        del self.grads
        self.grads = [[] for _ in range(self.n_networks)]

    def set_train_mode(self, train_mode: bool = True):
        """Setter for train_mode

        Args:
            train_mode (bool, optional): network in train or test mode. Defaults to True.
        """
        self.train_mode = train_mode

    def clear(self):

        super().clear()

        del self.grads
        self.grads = [[] for _ in range(self.n_networks)]

    def instantiate_networks(self, loaded_weights=None):

        super().instantiate_networks(loaded_weights=loaded_weights)

        for net in self.networks:
            net.cuda().train(self.train_mode)

    def forward_batches(self):
        """Batches are passed forward n_batches times across all models.
        Same batch of data goes to each model.

        Returns:
            list[list[torch.Tensor]: list of gradients for each network
        """

        device = torch.cuda.current_device()

        for _ in range(self.n_batches):
            
            #get one batch of data
            inputs = self.get_inputs()

            _inputs = inputs.clone().cuda(device=device, non_blocking=True)

            # for each network, run one forward with same batch of data
            for net, grads in zip(self.networks, self.grads):
                self.forward(net, grads, _inputs)
            
            del inputs
            del _inputs

        return [grads for grads in self.grads]


    def forward(self, net: torch.nn.Module, grads: list, input: torch.Tensor):
        """Forward pass over one batch and one model and collect the gradients into a flattened vector.

        Args:
            net (torch.nn.Module): network
            grads (list): list to collect gradients
            input (torch.nn.Tensor): one batch of data
        """
        net.zero_grad()
        logit = net.forward(input)[1]
        '''
        #Legacy function (literally does not work ???)
        for j in range(len(input)):
            logit[j:j+1].backward(torch.ones_like(logit[j:j+1]), retain_graph=True)
            grad = []
            for name, w in net.named_parameters():
                if 'weight' in name and w.grad is not None:
                    grad.append(w.grad.view(-1).detach())
            grads.append(torch.cat(grad, -1))

        for grad in grads:
            for grad_ in grad:
                if grad_.grad is not None:
                    grad_.backward()
        '''
        logit.backward(torch.ones_like(logit), retain_graph=True)
        grad = []
        for layer in net.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                w = layer.weight
                if w.grad is not None:
                    grad.append(w.grad)
                else:
                    grad.append(torch.zeros_like(w))
        _grad = torch.cat([_g.view(net._C, -1).detach().cpu() for _g in grad], -1)
        grads.append(_grad)
        del logit
        net.zero_grad()
        torch.cuda.empty_cache()

class LossGradientCollector(Collector):

    def __init__(self, data_loader, n_batches, loss_f, model_config=EDict(), arch_parameters=[], init='normal', seed=0, train_mode=False):

        super().__init__(data_loader, n_batches, model_config=model_config, arch_parameters=arch_parameters, init=init, seed=seed)
        self.train_mode = train_mode

        self.loss_f = loss_f

        self.grads = [[] for _ in range(self.n_networks)]

        torch.cuda.empty_cache()

    #Setters
    def set_arch_parameters(self, arch_parameters):

        super().set_arch_parameters(arch_parameters)

        del self.grads
        self.grads = [[] for _ in range(self.n_networks)]

    def set_train_mode(self, train_mode=True):
        self.train_mode = train_mode

    def clear(self):

        super().clear()

        del self.grads
        self.grads = [[] for _ in range(self.n_networks)]

    def instantiate_networks(self, loaded_weights=None):

        super().instantiate_networks(loaded_weights=loaded_weights)

        for net in self.networks:
            net.cuda().train(self.train_mode)

    #Batches are passed forward n_batches times across all models
    def forward_batches(self):

        device = torch.cuda.current_device()

        for _ in range(self.n_batches):

            inputs, targets = self.get_inputs(return_targets=True)

            _inputs = inputs.clone().cuda(device=device, non_blocking=True)

            for net, grads in zip(self.networks, self.grads):
                self.forward(net, grads, _inputs, targets)
            
            del inputs 
            del _inputs

        return [grads for grads in self.grads]

    #For a single model, gradients are queried
    def forward(self, net, grads, input, targets):
        net.zero_grad()
        logits = net.forward(input)[1].clone().cpu()
        loss = self.loss_f(logits, targets)
        loss.backward(retain_graph=True)
        for layer in net.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                w = layer.weight
                if w.grad is not None:
                    grads.append((w.grad * w.data).view(-1).detach())
                else:
                    grads.append(torch.zeros_like(w).view(-1).detach())
        net.zero_grad()
        torch.cuda.empty_cache()
