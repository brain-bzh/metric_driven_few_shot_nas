import numpy as np
import torch
from torch import nn
import torch.utils.data
import torchvision.transforms as transforms

from copy import deepcopy
from easydict import EasyDict as EDict

from ..utils.vision_utils import RandChannel

from .collector import Collector
from .metric import Metric

## Re-implementation of original code from W. Chen

class PostActivationsCollector(Collector):

    def __init__(self, data_loader, n_batches, crop_size, model_config = EDict(), arch_parameters=[], init='normal', seed=0):

        super().__init__(data_loader, n_batches, model_config=model_config, arch_parameters=arch_parameters, init=init, seed=seed)
        self.crop_size = crop_size

        self.transforms = [transforms.ToPILImage(), transforms.RandomResizedCrop(self.crop_size), transforms.ToTensor(), RandChannel(1)]

        self.postactivations = []
        self.postact_data = [[] for _ in self.arch_parameters]

        torch.cuda.empty_cache()

    def set_arch_parameters(self, arch_parameters):
        
        super().set_arch_parameters(arch_parameters)
        
        del self.postactivations
        del self.postact_data
        self.postactivations = []
        self.postact_data = [[] for _ in self.arch_parameters]

    def instantiate_networks(self, loaded_weights=None):
        
        super().instantiate_networks(loaded_weights=loaded_weights)

        for net in self.networks:
            net.cuda()
            self.register_hook(net)

    def set_crop_size(self, crop_size):
        self.crop_size = crop_size
        del self.transforms
        self.transforms = [transforms.ToPILImage(), transforms.RandomResizedCrop(self.crop_size), transforms.ToTensor(), RandChannel(1)]
    
    def clear(self):

        super().clear()

        del self.postactivations
        del self.postact_data
        self.postactivations = []
        self.postact_data = [[] for _ in self.arch_parameters]
    
    def register_hook(self, model):
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(hook=self.hook_in_forward)

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.postactivations.append(output.detach())                                          #Obtain the output of ReLU nodes at each forward pass

    def forward_batches(self):
        for _ in range(self.n_batches):

            inputs = self.get_inputs()

            _inputs = torch.zeros((self.batch_size, 3, 3, 3))

            for i in range(inputs.size(0)):
                input = inputs[i,:,:,:]
                for transform in self.transforms:
                    input = transform(input)
                _inputs[i,:,:,:] = input

            del inputs
            
            for n, net in enumerate(self.networks):
                self.forward(net, n, _inputs)

            del _inputs

        return self.postact_data
    
    def forward(self, net, n, input):
        self.postactivations = []
        with torch.no_grad():
            net.forward(input.cuda())
            if len(self.postactivations) != 0:
                self.postact_data[n].append(torch.cat([f.view(input.size(0), -1).detach() for f in self.postactivations], 1))
            else:
                self.postact_data[n].append(None)
        del self.postactivations
        self.postactivations = []

#Computes and compares numbers of linear regions
class LinearRegions(Metric):

    def __init__(self, collectors=[], repeat=1, seed=0):

        super().__init__(collectors, repeat=repeat, use_reference=True, seed=seed)
        if len(self.collectors)>0:
            for collector in self.collectors:
                assert isinstance(collector, PostActivationsCollector), 'This metric requires the use of linear regions collectors'

        self.metrics = []

    def set_collectors(self, collectors=[]):

        super().set_collectors(collectors)
        if len(self.collectors)>0:
            for collector in self.collectors:
                assert isinstance(collector, PostActivationsCollector), 'This metric requires the use of linear regions collectors'

        del self.metrics
        self.metrics = []

    def calc(self):

        self.metrics = []

        for n in range(self.n_networks + int(self.use_reference)):
            metric = 0

            for r in range(self.repeat):
                for c, collector in enumerate(self.collectors):
                    
                    n_samples = collector.batch_size * collector.n_batches
                    b_ = 0

                    with torch.no_grad():
                        
                        for batch_postact in self.collector_outputs[r][c][n]:
                            if b_ == 0:
                                n_batches_local, n_neurons = batch_postact.size()
                                postact_signs = torch.zeros(n_samples, n_neurons)
                            postact_signs[b_:b_+n_batches_local] = torch.sign(batch_postact)                    #Basic activation pattern : post-activation sign
                            b_ += n_batches_local

                        postact_signs = postact_signs.cuda()
                        lr_mat = torch.matmul(postact_signs.half(), (1-postact_signs).T.half())                 #Each element in res: A * (1 - B)
                        lr_mat += torch.clone(lr_mat).T                                                         #Make symmetric, a non-zero element indicates a pair of two different linear regions
                        lr_mat = 1 - torch.sign(torch.clone(lr_mat))                                            #A non-zero element now indicates two linear regions are identical
                        lr_mat = torch.clone(lr_mat).sum(1)                                                     #For each sample's linear region: how many identical regions from other samples
                        lr_mat = 1. / torch.clone(lr_mat).float()                                               #Contribution of each redundant (repeated) linear region
                        n_lr = lr_mat.sum().item()                                                              #Sum of unique regions (by aggregating contribution of all regions)

                        del postact_signs, lr_mat
                        torch.cuda.empty_cache()
                    
                    metric += n_lr
            
            self.metrics.append(metric/(self.repeat*len(self.collectors)))
        
        del metric

    def compare(self):

        for m in range(len(self.metrics)):
            for m_ in range(m+1, len(self.metrics)):
                self.dist_matrix[m, m_] = - abs(self.metrics[m]-self.metrics[m_])                               #Similarity is higher when distance is near 0. Thus invert sign to transform min cut problem into max cut problem.

        self.dist_matrix = self.dist_matrix + self.dist_matrix.T