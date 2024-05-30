import math
import numpy as np
import torch
from torch import nn

from copy import deepcopy

from ..lib.natsbench import create
from .paths import NASBENCH_PATH

INF = 1000

## Taken from TENAS (Credits W. Chen)

def round_to(number, precision, eps=1e-8):
    #round number to significant figure
    dtype = type(number)
    if number == 0:
        return number
    sign = number / abs(number)
    number = abs(number) + eps
    power = math.floor(math.log(number, 10)) + 1
    if dtype == int:
        return int(sign * round(number*10**(-power), precision) * 10**(power))
    else:
        return sign * round(number*10**(-power), precision) * 10**(power)

def is_single_path(network):
    #Check if network is single path
    arch_parameters = network.arch_parameters
    edge_active = (nn.functional.softmax(arch_parameters, 1) > 0.01).float().sum(1)
    for edge in edge_active:
        assert edge > 0
        if edge > 1:
            return False
    return True

## Custom

def arch2alphas(arch, search_space, max_nodes, return_supernet=False):
    #Input : architecture string (NAS Bench query format)
    #Output : alpha matrix (0 = activated operation, -INF = deactivated operation)
    n_ops = len(search_space)
    op_dict = {search_space[i] : i for i in range(n_ops)} #column index for ops
    n_edges = sum(range(1, max_nodes))
    alphas = np.full((n_edges,n_ops), -INF)
    if return_supernet:
        alphas_ori = deepcopy(alphas)
    arch_ = arch.split('|') #'','op0','+','op1','op2','+','op3','op4','op5',''
    list_op = []
    for op in arch_:
        if op!='' and op!='+':
            list_op.append(op)
    for e in range(n_edges): #all edges
        for op in op_dict.keys():
            if list_op[e].startswith(op):
                if op!='none':
                    alphas[e, op_dict[op]] = 0. #activate op
    if return_supernet:
        return(nn.Parameter(torch.Tensor(alphas)), nn.Parameter(torch.Tensor(alphas_ori)))
    else:
        return(nn.Parameter(torch.Tensor(alphas)))

def make_rankings(sorted_struct):  
    #sorted_struct is of the type list([sorted_value, identifier])
    #returns dict(identifier:rank), with ties
    rankings = {}
    for idx in range(len(sorted_struct)):
        if idx==0:
            rankings[sorted_struct[idx][1]] = 1
        else:
            if sorted_struct[idx][0] == sorted_struct[idx-1][0]:
                rankings[sorted_struct[idx][1]] = rankings[sorted_struct[idx-1][1]]
            else:
                rankings[sorted_struct[idx][1]] = rankings[sorted_struct[idx-1][1]] + 1
    return(rankings)

def merge_rankings(list_rankings):
    #merges all rankings in list_rankings on the basis of similar identifiers
    #identifier that does not have a ranking in all rankings is dropped
    merged_ranking = {}
    ref = list_rankings[0]
    for id in ref.keys():
        is_in_all_rankings = True
        for ranking in list_rankings[1:]:
            if id not in ranking:
                is_in_all_rankings = False
        if is_in_all_rankings==True:
            merged_ranking[id] = [ranking[id] for ranking in list_rankings]
    return(merged_ranking)

def split_distance(split1, split2):
    #a custom distance function to caracterize the dissimilarity between two splits of the same subset

    def make_set(split):
        set = []
        for group in split:
            for k in group:
                set.append(k)
        return(set)

    set1 = sorted(make_set(split1))
    set2 = sorted(make_set(split2))
    assert set1 == set2, 'Subsets are not the same'

    dist = 0
    for k in set1:

        for group in split1:
            if k in group:
                k_neighbors1 = deepcopy(group)
                k_neighbors1.remove(k)
        for group in split2:
            if k in group:
                k_neighbors2 = deepcopy(group)
                k_neighbors2.remove(k)

        for n in k_neighbors1:
            if n not in k_neighbors2:
                dist += 1
        for n in k_neighbors2:
            if n not in k_neighbors1:
                dist += 1

    return(dist/(2*(len(set1)-1)))

def n_nets_from_arch_params(arch_parameters):
    n_nets = 1
    for l in range(arch_parameters.shape[0]):
        open_ops = 0
        for c in range(arch_parameters.shape[1]):
            if deepcopy(arch_parameters)[l, c].cpu().numpy() > -INF:
                open_ops += 1
        n_nets = n_nets * open_ops
    return(n_nets)

def list_all_archs(arch_parameters, edge=0):
    n_edges = arch_parameters.shape[1]

    if edge == n_edges:
        list_archs = []
        open_ops = []
        for o in range(arch_parameters.shape[1]):
            if deepcopy(arch_parameters)[edge, o].cpu().numpy() > -INF:
                open_ops.append(o)

        for op in open_ops:
            sub_arch_parameters = deepcopy(arch_parameters)
            sub_arch_parameters[edge, :] = -INF
            sub_arch_parameters[edge, op] = 0

            list_archs.append(sub_arch_parameters.cpu())

        return(list_archs)

    else:
        list_archs = []
        open_ops = []
        for o in range(arch_parameters.shape[1]):
            if deepcopy(arch_parameters)[edge, o].cpu().numpy() > -INF:
                open_ops.append(o)

        for op in open_ops:
            sub_arch_parameters = deepcopy(arch_parameters)
            sub_arch_parameters[edge, :] = -INF
            sub_arch_parameters[edge, op] = 0

            list_archs += list_all_archs(sub_arch_parameters, edge+1)
        
        return(list_archs)

def make_nasbench_rankings(list_archs, model, dataset='cifar10'):
    max_nets = len(list_archs)
    api = create(NASBENCH_PATH, 'tss', fast_mode=True, verbose=False)
    arch_list = []
    arch_parameters_backup = deepcopy(model.arch_parameters.detach().clone())
    for arch in list_archs:
        model.arch_parameters = nn.Parameter(arch)
        genotype = model.genotype
        idx = api.query_index_by_arch(genotype)
        performance = api.get_more_info(idx, dataset, hp='200', is_random=False)['test-accuracy']
        arch_list.append([performance, arch])
    return(sorted(arch_list, key=lambda i: i[0], reverse=True))