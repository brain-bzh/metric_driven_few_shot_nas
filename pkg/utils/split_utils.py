import math
import numpy as np
import random
from copy import deepcopy

from itertools import combinations_with_replacement, permutations

INF = 1000

########################################################################### Splits initializations ###########################################################################

def single_split(m, b, drop_leaves=False, seed=0):
    #Create a branching tree with m leaves, given a single branching factor that will be used at all stages of the tree
    #At the last layer, some random branches will be assigned branching factor 1 to make m leaves attainable
    #If drop leaves is True, the last layer is dropped, thus the number of leaves will be strictly inferior to m

    random.seed(seed)

    splits = {}
    n_iterations = max(0, math.ceil(math.log(m, b))-1)                    #In this case, the depth of the tree is easily derived from m and b
    n_nodes = [1]
    for i in range(0, n_iterations):
        splits[i] = {j : b for j in range(n_nodes[-1])}
        n_nodes.append(b**(i+1))

    if not drop_leaves:
        combis = list(combinations_with_replacement([1, b], n_nodes[-1]))   #All combinations of 1 and b of length the number of nodes
        r_bf = None
        for c in combis:
            if sum(c) == m:                                                 #There is a single combination of sum m
                r_bf = list(c)

        assert r_bf!=None, 'Incompatible parameters'                        #If no suitable combination was found, it is impossible to make m leaves from branching factor b

        random.shuffle(r_bf)                                                #Shuffle the combination to avoid bias
        splits[n_iterations] = {j : r_bf[j] for j in range(n_nodes[-1])}

    return(splits, n_iterations, n_nodes)

def ascending_split(m: int, b, drop_leaves: bool = False, seed: int = 0):
    """Create a branching tree with m leaves, given a list b of branching factors.

    Args:
        m (int): total number of leaves
        b (list[int]): list of branching factors
        drop_leaves (bool, optional): if m is not a power of b, then drop the final layer in the tree just return the closest lower power multiple of b. Defaults to False.
        seed (int, optional): random seed. Defaults to 0.

    Returns:
        splits (dict[int: dict[int: int]]): {level: {name of node at level: # of branches from node}}
        m_iterations (int): number of levels in tree
        n_nodes (list[int]): number of nodes at each level in the tree
    
    Note:
        Branching factors will be used in the tree in ascending order.
        At the last layer, branching factors will be assigned based on the combination of branching factors of sum m and of least variance
        In order to make m leaves attainable in most cases, 1 is always a possible branching factor
    """

    random.seed(seed)
    b_ = deepcopy(b)

    splits = {}
    q = m
    i = 0
    n_nodes = [1]
    min_bf = [1, min(b_)]

    while q/min_bf[-1] > 1:                                                 #Loop stops when reaching the leaves layer
        q = q/min_bf[-1]
        splits[i] = {j: min_bf[-1] for j in range(n_nodes[-1])}
        n_nodes.append(math.prod(min_bf))                                   #Number of nodes on this layer is the product of branching factors in upper layers
        if len(b_)>1:                                                       #Update the branching factor to be used in the next layer (if some are still unused, otherwise continue using the highest)
            b_.remove(min_bf[-1])
        min_bf.append(min(b_))
        i+=1

    assert n_nodes[-1] <= 32, "combinations with replacement becomes unfeasible with large number of max_networks. Try again with a smaller number of networks."
        
    if not drop_leaves:
        combis = list(combinations_with_replacement(min_bf, n_nodes[-1]))   #All combinations of branching factors of length the number of nodes
        r_bf = None
        for c in combis:
            if sum(c) == m:                                                 #Keep only combinations of sum m
                if r_bf == None:
                    r_bf = list(c)
                    r_bf_vals = (min(c), max(c))
                #Update chosen combination when a combination of lesser variance is found
                elif min(c) >= r_bf_vals[0] and max(c) <= r_bf_vals[1] and np.std(list(c))<=np.std(r_bf):
                    r_bf = list(c)
                    r_bf_vals = (min(c), max(c))
        
        assert r_bf!=None, 'Incompatible parameters'                        #If no suitable combination was found, it is impossible to make m leaves from branching factor b

        random.shuffle(r_bf)                                                #Shuffle the combination to avoid bias
        splits[i] = {j : r_bf[j] for j in range(n_nodes[-1])}

    n_iterations = len(splits)
    return splits, n_iterations, n_nodes

############################################################################## Architecture splitting ######################################################################

def clusterize(n_clusters, dist_matrix, ops):
    #Make n_clusters on a 1D axis from elements of ops based on the distance matrix, by listing exhaustively all combinations of varying lengths.

    assert n_clusters <= len(ops), 'Too many splits, change policy branching factor'

    if n_clusters == 1:
        return([[op for op in ops]])                                        #Case n_clusters = 1, all ops are in the single cluster

    elif n_clusters == len(ops):
        return([[op] for op in ops])                                        #Case n_clusters = n_ops, ops form a cluster each

    else:
        
        graphmin_dict = graph_cuts(ops, dist_matrix, n_clusters-1)          #Else, solve a graph min cut problem by cutting the graph recursively and choosing the lowest sum from all permmutations
        k = min(graphmin_dict, key= lambda j: graphmin_dict[j][1])          
        return(graphmin_dict[k][0])

def graph_cuts(ops, dist_matrix, remaining):
    #Recursively obtain all posible cuts of the ops list and the corresponding distance sums

    if remaining == 1:                                                      #Recursion break case : last iteration

        local_dict = {}
        for p_size in range(1, max(1, int(len(ops)/2) + 1)):                #Cuts are symmetrical so only check up to permutation size of half the number of ops
            p_ = enumerate(list(permutations(ops, p_size)))                 #List all permutations of given size
            for idx, p in p_:
                not_p = [op for op in ops if op not in p]                   #Additional space for a given permutation
                flow_sum = 0
                for op1 in p:
                    for op2 in not_p:
                        flow_sum += dist_matrix[op1, op2]                   #Add up distances between ops of the two additional spaces
                local_dict[(p_size, idx)] = ([list(p), list(not_p)], flow_sum) 
        return(local_dict)

    else:                                                                   #Main recursion case

        local_dict = {}
        for p_size in range(1, max(1, int(len(ops)/2))):                    #Cuts are symmetrical so only check up to permutation size of half the number of ops
            p_ = enumerate(list(permutations(ops, p_size)))                 #List all permutations of given size
            for idx, p in p_:
                not_p = [op for op in ops if op not in p]                   #Additional space for a given permutation
                child_dict = graph_cuts(not_p, dist_matrix, remaining-1)    #Recursively obtain cuts and sums from deeper levels
                flow_sum = 0
                for op1 in p:
                    for op2 in not_p:
                        flow_sum += dist_matrix[op1, op2]                   #Add up distances between ops of the two additional spaces
                for k, v in child_dict.items():
                    part_, flow = v
                    local_dict[(p_size, idx, k)] = ([list(p)] + part_, flow_sum + flow)             #Join cuts and add up distances with children values
        return(local_dict)

def detach_ops(arch_parameters: np.array, ops, edge: int):
    """Creates a disjoint set of architecture adjacency matrices (amm), each with a different op on the same edge deleted.

    Args:
        arch_parameters (np.array): amm
        ops (_type_): ops that are available at this stage
        edge (_type_): which edge to create the disjoint set over
    """

    sep_arch_params = []
    for op in ops:
        sep_ = deepcopy(arch_parameters)
        sep_[edge, op] = -INF
        sep_arch_params.append(sep_)

    return sep_arch_params

def create_disjoint_architectures(arch_parameters: np.array, ops, edge: int):
    """Creates a disjoint set of architecture adjacency matrices (amm), each with a different op on the same edge fixed
    and the other ops on that edge deleted. Will return the same number of disjoint architectures as ops passed in.

    Args:
        arch_parameters (np.array): amm
        ops (_type_): ops that are available at this stage
        edge (_type_): which edge to create the disjoint set over

    Returns:
        list[np.array]: list of disjoint architectures over an edge.
    """
    assert len(arch_parameters.shape) == 2, "Expect the amm to be a matrix"
    assert len(ops) <= arch_parameters.shape[1], "Can't have more ops than there are operations in the amm"
    assert 0 <= edge < arch_parameters.shape[0], "Chosen edge is not valid."

    disjoint_arch_params = []
    for op in ops:
        dj_arch = deepcopy(arch_parameters)
        dj_arch[edge, :] = -INF
        dj_arch[edge, op] = 0
        disjoint_arch_params.append(dj_arch)

    return disjoint_arch_params