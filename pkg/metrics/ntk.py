import numpy as np
import torch

from copy import deepcopy

from .collector import Collector
from .gradient import GradientCollector
from .metric import Metric

## Re-implementation of original code by W. Chen
#Computes and compares NTK condition numbers.
class NTKCond(Metric):

    def __init__(self, collectors=[], repeat=1, seed=0):

        super().__init__(collectors, repeat=repeat, use_reference=True, seed=seed)
        if len(self.collectors)>0:
            for collector in self.collectors:
                assert isinstance(collector, GradientCollector), 'This metric requires the use of gradient collectors'

    def set_collectors(self, collectors=[]):

        super().set_collectors(collectors)
        if len(self.collectors)>0:
            for collector in self.collectors:
                assert isinstance(collector, GradientCollector), 'This metric requires the use of gradient collectors'

    def calc(self):

        self.metrics = []

        for n in range(self.n_networks + int(self.use_reference)):
            metric = 0
            for r in range(self.repeat):
                for c in range(len(self.collectors)):
                
                    grads = self.collector_outputs[r][c][n]
                    _grads = torch.cat(grads, dim=0)
                    _grads.to('cpu')
                    #_grads = _grads.view(_grads.shape[0]*_grads.shape[1], -1)                                  #Stack gradients from all batches
                    #_ntk = torch.einsum('nc,mc->nm', [_grads, _grads])                                         #Apply matrix product
                    _ntk = torch.matmul(_grads, _grads.T)
                    eigenvalues = torch.linalg.eigvalsh(_ntk, UPLO='U')                                         #Get matrix eigenvalues
                    metric += np.nan_to_num((eigenvalues[-1]/eigenvalues[0]).item(), copy=True, nan=1000000.0)  #Condition number : lambda_n/lambda_0

                    del grads, _grads, _ntk, eigenvalues
                    torch.cuda.empty_cache()

            self.metrics.append(metric/(self.repeat*len(self.collectors)))                                      #Average over all repetitions
        
        del metric

    def compare(self):
        
        for m in range(len(self.metrics)):
            for m_ in range(m+1, len(self.metrics)):
                self.dist_matrix[m, m_] = - abs(self.metrics[m]-self.metrics[m_])                               #Similarity is higher when distance is near 0. Thus invert sign to transform min cut problem into max cut problem.

        self.dist_matrix = self.dist_matrix + self.dist_matrix.T
