import numpy as np
import torch
from torch import nn
import torch.utils.data
import torchvision.transforms as transforms

from copy import deepcopy
from easydict import EasyDict as EDict

from ..lib.models.cell_searchs.generic_model import ResNetBasicblock, SearchCell
from ..lib.models.cell_operations import OPS
from .collector import Collector
from .metric import Metric

class NParametersCollector(Collector):

    def __init__(self, data_loader, n_batches, model_config = EDict(), arch_parameters=[], init='normal', seed=0):

        self.model_config = model_config
        self.arch_parameters = arch_parameters
        self.n_params = []
        self.networks = []

    def set_arch_parameters(self, arch_parameters):
        
        super().set_arch_parameters(arch_parameters)

    def instantiate_networks(self, loaded_weights=None):
        
        self.networks = [
            DummyNBNetwork(
                arch_params_,
                self.model_config.C,
                self.model_config.N,
                self.model_config.max_nodes,
                self.model_config.num_classes,
                self.model_config.space,
                self.model_config.affine,
                self.model_config.track_running_stats
            ) for arch_params_ in self.arch_parameters
        ]

    def clear(self):

        super().clear()
        self.n_params = []

    def forward_batches(self):

        for n, net in enumerate(self.networks):
            self.n_params.append(np.sum([m.numel() for m in net.parameters()]))

        return(self.n_params)

class NParameters(Metric):

    def __init__(self, collectors = [], repeat: int = 1, use_reference: bool = True, seed: int = 0):

        super().__init__(collectors, repeat=repeat, use_reference=True, seed=seed)
        if len(self.collectors)>0:
            for collector in self.collectors:
                assert isinstance(collector, NParametersCollector), 'This metric requires the use of number of parameters collectors.'

        self.metrics = []

    def set_collectors(self, collectors=[]):

        super().set_collectors(collectors)
        if len(self.collectors)>0:
            for collector in self.collectors:
                assert isinstance(collector, NParametersCollector), 'This metric requires the use of number of parameters collectors.'

        del self.metrics
        self.metrics = []

    def calc(self):

        self.metrics = []

        for n in range(self.n_networks + int(self.use_reference)):
            metric = 0

            for r in range(self.repeat):
                for c in range(len(self.collectors)):
                    
                    nparams = self.collector_outputs[r][c][n]
                    metric += nparams
            
            self.metrics.append(metric/(self.repeat*len(self.collectors)))
            torch.cuda.empty_cache()
        
        del metric

    def compare(self):

        for m in range(len(self.metrics)):
            for m_ in range(m+1, len(self.metrics)):
                self.dist_matrix[m, m_] = - abs(self.metrics[m]-self.metrics[m_])                               #Similarity is higher when distance is near 0. Thus invert sign to transform min cut problem into max cut problem.

        self.dist_matrix = self.dist_matrix + self.dist_matrix.T

class DummySearchCell(nn.Module):

    def __init__(
        self,
        arch_parameters,
        C_in,
        C_out,
        stride,
        max_nodes,
        op_names,
        affine=False,
        track_running_stats=True,
    ):
        super(DummySearchCell, self).__init__()

        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out

        k = 0
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                if j == 0:
                    xlists = []
                    for o, op_name in enumerate(op_names):
                        if arch_parameters[k, o] > -1000:
                            xlists.append(
                                OPS[op_name](C_in, C_out, stride, affine, track_running_stats)
                            )
                else:
                    xlists = []
                    for o, op_name in enumerate(op_names):
                        if arch_parameters[k, o] > -1000:
                            xlists.append(
                                OPS[op_name](C_in, C_out, 1, affine, track_running_stats)
                            )
                self.edges[node_str] = nn.ModuleList(xlists)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

class DummyNBNetwork(nn.Module):
    #Only used for parameter counting

    def __init__(
            self, arch_parameters, C, N, max_nodes, num_classes, search_space, affine, track_running_stats, depth=-1, use_stem=True
        ):

        super(DummyNBNetwork, self).__init__()
        self._C = C
        self._layerN = N
        self.use_stem = use_stem #
        self._max_nodes = max_nodes
        self._num_class = num_classes #
        self._search_space = search_space #
        self._affine = affine #
        self._trs = track_running_stats #
        self._stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )
        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
        C_prev, num_edge, edge2index = C, None, None
        self._cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if depth <= 0 or index < depth: #
                if reduction:
                    cell = ResNetBasicblock(C_prev, C_curr, 2)
                else:
                    cell = DummySearchCell(
                        arch_parameters,
                        C_prev,
                        C_curr,
                        1,
                        max_nodes,
                        search_space,
                        affine,
                        track_running_stats,
                    )
                    if num_edge is None:
                        num_edge, edge2index = cell.num_edges, cell.edge2index
                    else:
                        assert (
                            num_edge == cell.num_edges and edge2index == cell.edge2index
                        ), "invalid {:} vs. {:}.".format(num_edge, cell.num_edges)
                self._cells.append(cell)
                C_prev = cell.out_dim
        self._op_names = deepcopy(search_space)
        self._Layer = len(self._cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(
            nn.BatchNorm2d(
                C_prev, affine=affine, track_running_stats=track_running_stats
            ),
            nn.ReLU(inplace=True),
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
