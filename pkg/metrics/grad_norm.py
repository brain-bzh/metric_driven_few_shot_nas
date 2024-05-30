import torch

from .metric import Metric

from .gradient import GradientCollector

class GradNorm(Metric):

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
            metric = 0
            for r in range(self.repeat):
                for c in range(len(self.collectors)):

                    grads = self.collector_outputs[r][c][n]
                    _grads = torch.stack(grads, 0)
                    metric += torch.linalg.norm(_grads)

            self.metrics.append(metric/(self.repeat*len(self.collectors)))

        del metric

    def compare(self):
        
        for m in range(len(self.metrics)):
            for m_ in range(m+1, len(self.metrics)):
                self.dist_matrix[m, m_] = - abs(self.metrics[m]-self.metrics[m_])                               #Similarity is higher when distance is near 0. Thus invert sign to transform min cut problem into max cut problem.

        self.dist_matrix = self.dist_matrix + self.dist_matrix.T
