import numpy as np
import torch

from copy import deepcopy
from easydict import EasyDict as EDict

from .collector import Collector
from .metric import Metric

class HessianCollector(Collector):

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

        weights = []
        for layer in net.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                weights.append(layer.weight)
                layer.weight.requires_grad_(True)

        net.zero_grad()
        logits = net.forward(input)[1].clone().cpu()
        loss = self.loss_f(logits, targets)
        grad_1 = torch.autograd.grad(loss, weights, allow_unused=True)

        net.zero_grad()
        logits = net.forward(input)[1].clone().cpu()
        loss = self.loss_f(logits, targets)
        grad_2 = torch.autograd.grad(loss, weights, create_graph=True, allow_unused=True)

        h = 0
        l = 0
        for layer in net.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                if grad_1[l] is not None and grad_2[l] is not None:
                    h += (grad_1[l].data * grad_2[l]).sum()
                l += 1
        h.backward()

        for layer in net.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                if layer.weight.grad is not None:
                    grads.append((layer.weight.grad * layer.weight.data).view(-1).detach())

        del weights, logits, loss, grad_1, grad_2, h
        net.zero_grad()
        torch.cuda.empty_cache()

class GRASP(Metric):

    def __init__(self, collectors=[], repeat=1, seed=0):

        super().__init__(collectors, repeat=repeat, use_reference=True, seed=seed)

        if len(self.collectors)>0:
            for collector in self.collectors:
                assert isinstance(collector, HessianCollector), 'This metric requires the use of hessian collectors'

    def set_collectors(self, collectors=[]):

        super().set_collectors(collectors)

        if len(self.collectors)>0:
            for collector in self.collectors:
                assert isinstance(collector, HessianCollector), 'This metric requires the use of hessian collectors'

    def calc(self):

        self.metrics = []

        for n in range(self.n_networks + int(self.use_reference)):
            metric = 0
            for r in range(self.repeat):
                for c in range(len(self.collectors)):
                    
                    h = self.collector_outputs[r][c][n]
                    grasp = [h_.sum() for h_ in h]
                    metric += -sum(grasp)
            
            self.metrics.append(metric/(self.repeat*len(self.collectors)))
            torch.cuda.empty_cache()

        del metric

    def compare(self):
        
        for m in range(len(self.metrics)):
            for m_ in range(m+1, len(self.metrics)):
                self.dist_matrix[m, m_] = - abs(self.metrics[m]-self.metrics[m_])                               #Similarity is higher when distance is near 0. Thus invert sign to transform min cut problem into max cut problem.

        self.dist_matrix = self.dist_matrix + self.dist_matrix.T