import numpy as np

from ..metrics import Metric

class HybridMetric:

    def __init__(self, metrics, weights=None):

        self.metrics = metrics
        self.n_metrics = len(self.metrics)
        if weights==None:
            self.weights = [1/len(metrics) for i in range(self.n_metrics)]
        else:
            self.weights = weights

        self.metrics_dict = {i : (self.metrics[i], self.weights[i]) for i in range(self.n_metrics)}

    def __call__(self, arch_parameters, arch_parameters_reference):

        self.n_networks = len(arch_parameters)
        self.dist_matrix = np.zeros((self.n_networks, self.n_networks))

        self.metrics_val = np.zeros((self.n_metrics, self.n_networks))
        for i, (metric, weight) in self.metrics_dict.items():
            self.metrics_val[i,:] = [weight*m for m in metric(arch_parameters, arch_parameters_reference=arch_parameters_reference)[0]]

        self.metrics_val = np.sum(self.metrics_val, axis=0)
        print(self.metrics_val)

        self.compare()

        return(self.metrics_val, self.dist_matrix)


    def compare(self):
        
        for m in range(self.n_networks):
            for m_ in range(m+1, self.n_networks):
                self.dist_matrix[m, m_] = - abs(self.metrics_val[m]-self.metrics_val[m_])                               #Similarity is higher when distance is near 0. Thus invert sign to transform min cut problem into max cut problem.

        self.dist_matrix = self.dist_matrix + self.dist_matrix.T