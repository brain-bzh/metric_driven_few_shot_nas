import numpy as np
import torch

from copy import deepcopy
from easydict import EasyDict as EDict

from .gradient import LossGradientCollector
from .collector import Collector
from .metric import Metric

class SynflowCollector(Collector):

    def __init__(self, data_loader, n_batches, model_config=EDict(), arch_parameters=[], init='normal', seed=0, train_mode=False):

        super().__init__(data_loader, n_batches, model_config=model_config, arch_parameters=arch_parameters, init=init, seed=seed)
        self.train_mode = train_mode

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

            inputs = self.get_inputs()
            
            inputs_dim = list(inputs[0,:].shape)
            inputs_ = torch.ones([1]+inputs_dim).float().to(device)

            for net, grads in zip(self.networks, self.grads):
                self.forward(net, grads, inputs_)
            
            del inputs 
            del inputs_

        return [grads for grads in self.grads]

    #For a single model, gradients are queried
    def forward(self, net, grads, input):

        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()

        net.zero_grad()
        logits = net.forward(input)[1]
        torch.sum(logits).backward(retain_graph=True)

        for layer in net.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                w = layer.weight
                if w.grad is not None:
                    grads.append((w.grad * w.data).view(-1).detach())
                else:
                    grads.append(torch.zeros_like(w).view(-1).detach())

        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.mul_(signs[name])
        
        del logits, signs
        net.zero_grad()
        torch.cuda.empty_cache()

class Synaptic(Metric):

    def __init__(self, collectors=[], repeat=1, seed=0, mode='snip'):

        super().__init__(collectors, repeat=repeat, use_reference=True, seed=seed)
        self.mode = mode

        if len(self.collectors)>0:
            for collector in self.collectors:
                if self.mode=='snip':
                    assert isinstance(collector, LossGradientCollector), 'This metric requires the use of loss gradient collectors'
                elif self.mode=='synflow':
                    assert isinstance(collector, SynflowCollector), 'This metric requires the use of synflow collectors'

    def set_collectors(self, collectors=[]):

        super().set_collectors(collectors)
        if len(self.collectors)>0:
            for collector in self.collectors:
                if self.mode=='snip':
                    assert isinstance(collector, LossGradientCollector), 'This metric requires the use of loss gradient collectors'
                elif self.mode=='synflow':
                    assert isinstance(collector, SynflowCollector), 'This metric requires the use of synflow collectors'

    def calc(self):

        self.metrics = []

        for n in range(self.n_networks + int(self.use_reference)):
            metric = 0
            for r in range(self.repeat):
                for c in range(len(self.collectors)):

                    grads = self.collector_outputs[r][c][n]
                    snip = [torch.abs(grad).sum() for grad in grads]
                    metric += sum(snip)
            
            self.metrics.append(metric/(self.repeat*len(self.collectors)))
            torch.cuda.empty_cache()

        del grads, snip, metric

    def compare(self):
        
        for m in range(len(self.metrics)):
            for m_ in range(m+1, len(self.metrics)):
                self.dist_matrix[m, m_] = - abs(self.metrics[m]-self.metrics[m_])                               #Similarity is higher when distance is near 0. Thus invert sign to transform min cut problem into max cut problem.

        self.dist_matrix = self.dist_matrix + self.dist_matrix.T