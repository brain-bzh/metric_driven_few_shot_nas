import torch, random
import torch.nn as nn
from copy import deepcopy
from typing import Text
from torch.distributions.categorical import Categorical

from ..lib.models.cell_operations import ResNetBasicblock, drop_path
from ..lib.models.cell_searchs.search_cells import NAS201SearchCell as SearchCell
from ..lib.models.cell_searchs.genotypes import Structure

class LambdaDartsModel(nn.Module):
    def __init__(
        self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats, depth=-1, use_stem=True #
    ):
        super(LambdaDartsModel, self).__init__()
        self._C = C
        self._layerN = N
        self.use_stem = use_stem #
        self._max_nodes = max_nodes

        self.lambda_ = 0
        self.epsilon_0 = 0
        self.corr_alphas = []

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
                    cell = SearchCell(
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
        self._num_edge = num_edge
        # algorithm related
        self.arch_parameters = nn.Parameter(
            1e-3 * torch.randn(num_edge, len(search_space))
        )
        self._mode = None
        self.dynamic_cell = None
        self._tau = None
        self._algo = None
        self._drop_path = None
        self.verbose = False

    def set_cal_mode(self, mode, dynamic_cell=None):
        assert mode in ["gdas", "enas", "urs", "joint", "select", "dynamic"]
        self._mode = mode
        if mode == "dynamic":
            self.dynamic_cell = deepcopy(dynamic_cell)
        else:
            self.dynamic_cell = None

    def set_drop_path(self, progress, drop_path_rate):
        if drop_path_rate is None:
            self._drop_path = None
        elif progress is None:
            self._drop_path = drop_path_rate
        else:
            self._drop_path = progress * drop_path_rate

    @property
    def mode(self):
        return self._mode

    @property
    def drop_path(self):
        return self._drop_path

    @property
    def weights(self):
        xlist = list(self._stem.parameters())
        xlist += list(self._cells.parameters())
        xlist += list(self.lastact.parameters())
        xlist += list(self.global_pooling.parameters())
        xlist += list(self.classifier.parameters())
        return xlist

    def set_tau(self, tau):
        self._tau = tau

    @property
    def tau(self):
        return self._tau

    @property
    def alphas(self):
        if self._algo == "enas":
            return list(self.controller.parameters())
        else:
            return [self.arch_parameters]

    @property
    def message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self._cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self._cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_C}, Max-Nodes={_max_nodes}, N={_layerN}, L={_Layer}, alg={_algo})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    @property
    def genotype(self):
        genotypes = []
        for i in range(1, self._max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                with torch.no_grad():
                    weights = self.arch_parameters[self.edge2index[node_str]]
                    op_name = self._op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def dync_genotype(self, use_random=False):
        genotypes = []
        with torch.no_grad():
            alphas_cpu = nn.functional.softmax(self.arch_parameters, dim=-1)
        for i in range(1, self._max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                if use_random:
                    op_name = random.choice(self._op_names)
                else:
                    weights = alphas_cpu[self.edge2index[node_str]]
                    op_index = torch.multinomial(weights, 1).item()
                    op_name = self._op_names[op_index]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def get_log_prob(self, arch):
        with torch.no_grad():
            logits = nn.functional.log_softmax(self.arch_parameters, dim=-1)
        select_logits = []
        for i, node_info in enumerate(arch.nodes):
            for op, xin in node_info:
                node_str = "{:}<-{:}".format(i + 1, xin)
                op_index = self._op_names.index(op)
                select_logits.append(logits[self.edge2index[node_str], op_index])
        return sum(select_logits).item()

    def return_topK(self, K, use_random=False):
        archs = Structure.gen_all(self._op_names, self._max_nodes, False)
        pairs = [(self.get_log_prob(arch), arch) for arch in archs]
        if K < 0 or K >= len(archs):
            K = len(archs)
        if use_random:
            return random.sample(archs, K)
        else:
            sorted_pairs = sorted(pairs, key=lambda x: -x[0])
            return_pairs = [sorted_pairs[_][1] for _ in range(K)]
            return return_pairs
    
    def normalize_archp(self):
        self._save_arch_parameters()
        if self.mode == "gdas":
            while True:
                gumbels = -torch.empty_like(self.arch_parameters).exponential_().log()
                logits = (self.arch_parameters.log_softmax(dim=1) + gumbels) / self.tau
                probs = nn.functional.softmax(logits, dim=1)
                index = probs.max(-1, keepdim=True)[1]
                one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                hardwts = one_h - probs.detach() + probs
                if (
                    (torch.isinf(gumbels).any())
                    or (torch.isinf(probs).any())
                    or (torch.isnan(probs).any())
                ):
                    continue
                else:
                    break
            with torch.no_grad():
                hardwts_cpu = hardwts.detach().cpu()
            return hardwts, hardwts_cpu, index, "GUMBEL"
        else:
            alphas = nn.functional.softmax(self.arch_parameters, dim=-1)
            alphas = alphas.requires_grad_(True)
            index = alphas.max(-1, keepdim=True)[1]
            with torch.no_grad():
                alphas_cpu = alphas.detach().cpu()
                alphad_cpu = alphas_cpu.requires_grad_(True)
            return alphas, alphas_cpu, index, "SOFTMAX"
  
    def _save_arch_parameters(self):
        self._saved_arch_parameters = [p.clone() for p in self.arch_parameters]
  
    def softmax_arch_parameters(self):
        self._save_arch_parameters()
        for p in self.arch_parameters:
            p.data.copy_(nn.functional.softmax(p, dim=-1))
            
    def restore_arch_parameters(self):
        for i, p in enumerate(self.arch_parameters):
            p.data.copy_(self._saved_arch_parameters[i])
        del self._saved_arch_parameters
  
    def clip(self):
        for p in self.arch_parameters:
            for line in p:
                max_index = line.argmax()
                line.data.clamp_(0, 1)
                if line.sum() == 0.0:
                    line.data[max_index] = 1.0
                line.data.div_(line.sum())

    def forward(self, inputs, pert=None):
        alphas, alphas_cpu, index, verbose_str = self.normalize_archp()
        if self.use_stem: #
            feature = self._stem(inputs)
        else: #
            feature = inputs
        self.corr_alphas = []
        for i, cell in enumerate(self._cells):
            if isinstance(cell, SearchCell):
                alphas_snapshot = alphas.clone()
                alphas_snapshot.retain_grad()
                self.corr_alphas.append(alphas_snapshot)
                if pert:
                    alphas_snapshot = alphas_snapshot - pert[i]
                feature = cell.forward_joint(feature, alphas_snapshot)
            else:
                feature = cell(feature)
            if self.drop_path is not None:
                feature = drop_path(feature, self.drop_path)
        if self.verbose and random.random() < 0.001:
            print(verbose_str)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits
    
    def get_arch_grads(self):
        grads = [w.grad.data.clone().detach().reshape(-1) for w in self.corr_alphas]
        return grads

    def get_perturbations(self):
        grads = self.get_arch_grads()

        def get_perturbation_for_cell(layer_gradients):
            with torch.no_grad():
                weight = 1 / ((len(layer_gradients) * (len(layer_gradients) - 1)) / 2)
                if self.corr_type == 'corr':
                    u = [g / g.norm(p=2.0) for g in layer_gradients]
                    sum_u = sum(u)
                    I = torch.eye(sum_u.shape[0]).cuda()
                    P = [(1 / g.norm(p=2.0)) * (I - torch.ger(u_l, u_l)) for g, u_l in zip(layer_gradients, u)]
                    perturbations = [weight * (P_l @ sum_u).reshape(self.alphas_normal.shape) for P_l in P]
                elif self.corr_type == 'signcorr':
                    perturbations = []
                    for i in range(len(layer_gradients)):
                        g = layer_gradients[i]
                        dir = torch.zeros_like(g)
                        for j in range(len(layer_gradients)):
                            if i==j: continue
                            g_ = layer_gradients[j]
                            dot, abs_dot = torch.dot(g, g_), torch.dot(torch.abs(g), torch.abs(g_))
                            dir += (torch.ones_like(g_) - (dot / abs_dot) * torch.sign(g) * torch.sign(g_)) * g_ / abs_dot
                        perturbations.append(weight * dir.reshape(self.arch_parameters.shape))
            return perturbations

        pert = get_perturbation_for_cell(grads)
        self.epsilon = self.epsilon_0 / torch.cat(pert, dim=0).norm(p=2.0).item()

        pert_ = []
        i = 0
        for cell in self._cells:
            if isinstance(cell, SearchCell):
                pert_.append(pert[i] * self.epsilon)
                i += 1
            else:
                pert_.append(None)
        return pert_

    def get_reg_grads(self, forward_grads, backward_grads):
        for idx, param in enumerate(self.parameters()):
            f = forward_grads[idx]
            b = backward_grads[idx]
            if f is not None and b is not None and param is not None:
                reg_grad = (f - b).div_(2 * self.epsilon)
                param.grad.data.add_(self.lambda_ * reg_grad)

        #reg_grad = [(f - b).div_(2 * self.epsilon) for f, b in zip(forward_grads, backward_grads)]
        #for idx, param in enumerate(self.parameters()):
        #   param.grad.data.add_(self.lambda_ * reg_grad[idx])

    def get_corr(self):
        grads_normal, grads_reduce = self.get_arch_grads()

        def corr(x):
            res = []
            norms = [x_.norm() for x_ in x]
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    res.append(
                        (torch.dot(x[i], x[j]) / (norms[i] * norms[j])).item())
            return sum(res) / len(res)
        return corr(grads_normal)
