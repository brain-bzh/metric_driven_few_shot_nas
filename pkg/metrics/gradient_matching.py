import numpy as np
import torch
from torch.nn.functional import cosine_similarity

from .metric import Metric
from .gradient import GradientCollector

class GradientMatching(Metric):

    def __init__(self, collectors=[], repeat=1, seed=0):

        super().__init__(collectors, repeat=repeat, use_reference=False, seed=seed)
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
            metric = None
            for r in range(self.repeat):
                for c in range(len(self.collectors)):

                    grads = self.collector_outputs[r][c][n]
                    _grads = torch.stack(grads, dim=0)
                    _grads = _grads.view(-1)
                    if metric is None:
                        metric = torch.zeros((self.repeat*len(self.collectors), _grads.size(0)))
                    metric[c*self.repeat+r,:] = _grads
                
            self.metrics.append(metric.view(1, metric.shape[0]*metric.shape[1]))

        del metric

    def compare(self):

        for m in range(len(self.metrics)):
            for m_ in range(m+1, len(self.metrics)):
                self.dist_matrix[m, m_] = torch.mean(cosine_similarity(self.metrics[m], self.metrics[m_]))

        self.dist_matrix = self.dist_matrix + self.dist_matrix.T
        print(self.dist_matrix)