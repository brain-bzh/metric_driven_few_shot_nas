import torch
import numpy as np

from .metric import Metric

from .gradient import GradientCollector

class JacobCov(Metric):

    def __init__(self, collectors=[], repeat=1, seed=0, eps=1e-8):

        super().__init__(collectors, repeat=repeat, use_reference=True, seed=seed)
        if len(self.collectors)>0:
            for collector in self.collectors:
                assert isinstance(collector, GradientCollector), 'This metric requires the use of gradient collectors'
        
        self.eps = eps

    def set_collectors(self, collectors=[]):

        super().set_collectors(collectors)
        if len(self.collectors)>0:
            for collector in self.collectors:
                assert isinstance(collector, GradientCollector), 'This metric requires the use of gradient collectors'

    def set_eps(self, eps):

        self.eps = eps

    def calc(self):

        self.metrics = []

        for n in range(self.n_networks + int(self.use_reference)):
            array = None
            metric = 0
            for r in range(self.repeat):
                for c in range(len(self.collectors)):

                    grads = self.collector_outputs[r][c][n]
                    _grads = torch.cat(grads, dim=0)
                    _grads.to('cpu')
                    if array is None:
                        array = torch.zeros((self.repeat*len(self.collectors)*_grads.size(0), _grads.size(1)))
                    array[(c*self.repeat+r)*_grads.size(0):(c*self.repeat+r+1)*_grads.size(0),:] = _grads
                    
            corr = torch.corrcoef(array)
            eigval, _ = torch.linalg.eig(corr)
            metric = -torch.sum(np.log(eigval + self.eps) + 1/(eigval + self.eps))
            self.metrics.append(metric)

        del array, corr, eigval, _, metric

    def compare(self):
        
        for m in range(len(self.metrics)):
            for m_ in range(m+1, len(self.metrics)):
                self.dist_matrix[m, m_] = - abs(self.metrics[m]-self.metrics[m_])                               #Similarity is higher when distance is near 0. Thus invert sign to transform min cut problem into max cut problem.

        self.dist_matrix = self.dist_matrix + self.dist_matrix.T
