import abc
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np

from ..lib.models import get_cell_based_tiny_net
from ..utils.init_utils import init_model
from .collector import Collector


class Metric:
    """
    Metrics will query values from one or multiple collectors, then compute a specific metric for all passed architectures.
    If several architectures are passed, a distance matrix of the defined metric between all architectures is returned.
    """

    def __init__(self, collectors = [], repeat: int = 1, use_reference: bool = True, seed: int = 0):
        """
        Args:
            collectors (list, optional): one or several collector instances. Must be of a collector type defined by the metric. Defaults to [].
            repeat (int, optional): number of times to average collector outputs over when computing metrics. Defaults to 1.
            use_reference (bool, optional): If use_reference, arch_parameters_reference will be used as the reference. Defaults to True.
            seed (int, optional): torch seed. Defaults to 0.
        
        Note:
            use_reference is true if the metric needs to calculate a difference between the reference amm's and the reference amm. 
        """
        
        self.collectors = collectors
        for collector in self.collectors:
            assert collector.model_config['super_type']=='basic' and collector.model_config['name']=='generic', 'Incompatible config'
        self.repeat = repeat
        self.use_reference = use_reference

        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def __call__(self, arch_parameters, arch_parameters_reference: np.array = None, weights_reference=None):
        """Compute metrics and distance matrix for the given list of architectures.

        Args:
            arch_parameters (list[np.array]): list of a disjoint subset of amm's with 1 operation fixed on a specific edge per amm.   
            arch_parameters_reference (np.array, optional): amm. Defaults to None.
        """

        if not self.use_reference:
            arch_parameters_reference = None

        torch.cuda.empty_cache()

        self.n_networks = len(arch_parameters)
        
        self.metrics = []
        self.dist_matrix = np.zeros((self.n_networks, self.n_networks))
        self.collector_outputs = [[] for r in range(self.repeat)]
        network_reference = None

        #Instantiate all models
        if arch_parameters_reference is not None:
            arch_parameters_expanded = arch_parameters + [arch_parameters_reference]
        else:
            arch_parameters_expanded = arch_parameters
        for collector in self.collectors:
            collector.set_arch_parameters(arch_parameters_expanded)

        #Collect values for all models with all collectors, repeated R times
        for r in range(self.repeat):

            for collector in self.collectors:

                collector.instantiate_networks(loaded_weights=weights_reference)
                self.collector_outputs[r].append(collector.forward_batches())
                collector.clear()

        #Compute metrics for every model using the defined formula
        self.calc()

        del self.collector_outputs

        #Normalize metrics using the reference
        if arch_parameters_reference is not None:
            self.metrics = [(self.metrics[i]-self.metrics[-1])/self.metrics[-1] for i in range(self.n_networks)]

        #Make a distance matrix using the defined comparison formula
        self.compare()

        return(self.metrics, self.dist_matrix)

    #Setters
    def set_collectors(self, collectors):

        del self.collectors
        self.collectors = deepcopy(collectors)

    def set_model_config(self, model_config):

        self.model_config = model_config

    def set_use_reference(self, use_reference):

        self.use_reference = use_reference

    def set_seed(self, seed):

        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @abc.abstractmethod
    def calc(self):
        """
        Generic metric computation : average of collector outputs sums across all repetitions
        Metric objects should define their own calc functions to compute the values of the metric for a list of models
        """
        self.metrics = [sum([sum([self.collector_outputs[r][c][n] for c in range(len(self.collectors))]) for r in range(self.repeat)])/self.repeat for n in range(self.n_networks +1)]

    @abc.abstractmethod
    def compare(self):
        """
        Metric objects should define their own compare functions to compute a distance matrix.
        This can be adapted to accomodate whether the splitting problem that needs to be solved is min-cut or max-cut, and in other cases as well (see GM-NAS approach)
        Generic metric comparison : L1 norm
        """

        for m in range(len(self.metrics)):
            for m_ in range(m+1, len(self.metrics)):
                self.dist_matrix[m, m_] = abs(self.metrics[m]-self.metrics[m_])

        self.dist_matrix = self.dist_matrix + self.dist_matrix.T