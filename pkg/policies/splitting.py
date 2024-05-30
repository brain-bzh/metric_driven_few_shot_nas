import typing as typ
import random
from copy import deepcopy
import numpy as np

from itertools import chain, permutations

from..metrics.metric import Metric
from ..utils.split_utils import single_split, ascending_split, clusterize, create_disjoint_architectures
from ..utils.misc_utils import round_to

INF = 1000

class SplittingPolicy():
    """
        Handles the orchestration of multi-level supernet splitting based on a Metric.
    """

    def __init__(self,
        max_nets: int,
        branching_factors,
        edges = None,
        metric: Metric = None,
        trainer=None,
        warmup_epochs=[],
        branch_mode: str = 'single',
        drop_leaves: bool = False,
        seed: int = 0
    ):
        """
        Args:
            max_nets (int): maximum leaf-sub-supernets to finish with
            branching_factors (list[int]): number of branches to split each node in the tree
            edges (typ.Union[list[int], None], optional): List of edges to split in order mode. Defaults to None.
            metric (_type_, optional): Metric which determines how to split graph. Defaults to None.
            branch_mode (str, optional): _description_. Defaults to 'single'.
            drop_leaves (bool, optional): whether to drop leaves of the branching tree. Defaults to False.
            seed (int, optional): seed to pass to metric. Defaults to 0.
        """

        self.seed = seed

        self.max_nets = max_nets
        self.branching_factors = branching_factors
        self.edges = edges
        self.metric = metric
        self.trainer = trainer
        self.warmup_epochs = warmup_epochs
        self.branch_mode = branch_mode
        self.drop_leaves = drop_leaves

        self.init_splits()

    def init_splits(self):
        """
            Called at initialization and in many setters. 
        """
        # Note: 'single' is never activated or used because self.branching_factors is always a list.
        # while simpler 'ascending' actually covers the same case and we can use it in generality
        if self.branch_mode == 'single':
            assert isinstance(self.branching_factors, int), 'Tried using branch mode single with more than 1 branching factor'
            assert self.branching_factors <= self.max_nets, 'Incompatible parameters' #Can't branch more than the maximum number of leaves
            self.splits, self.n_iterations, self.n_nodes = single_split(self.max_nets, self.branching_factors, self.drop_leaves, self.seed)
            if self.edges!=None:
                assert isinstance(self.edges, list) and len(self.edges) >= self.n_iterations, 'Not enough edges' #Case where not enough edges were passed to sustain the number of iterations

        elif self.branch_mode == 'ascending':
            assert isinstance(self.branching_factors, list), 'Use a list with branch mode ascending'
            assert min(self.branching_factors) <= self.max_nets, 'Incompatible parameters' #Make sure all branching factors are under the maximum number of leaves
            self.splits, self.n_iterations, self.n_nodes = ascending_split(self.max_nets, self.branching_factors, self.drop_leaves, self.seed)
            if self.edges!=None:
                assert isinstance(self.edges, list) and len(self.edges) >= self.n_iterations, 'Not enough edges' #Case where not enough edges were passed to sustain the number of iterations

    def split(self, arch_parameters: np.array, return_splits: bool = False):
        """Given an architecture adjacency matrix, prepares for splits, sends off to a recursive splitting function and returns the leaf sub-super networks
        Orchestration and initialization function. 

        Args:
            arch_parameters (np.array): architecture adjacency matrix denoting active edges of an architecture.
            return_splits (bool, optional): TODO _description_. Defaults to False.

        Note:
            It is possible to pass an already trimmed architecture, in that case trimmed operations will never be regenerated.
            If metric is None, then operation groupings at each edge are selected at random.
        """

        self.groupings = {}    
        n_edges = arch_parameters.shape[0]
        n_ops = arch_parameters.shape[1]
        if self.edges!=None:
            assert max(self.edges) <= n_edges-1, 'Edges to split do not match the search space'
        assert max([max(v.values()) for k, v in self.splits.items()]) <= n_ops, 'Branching factor does not match the search space'

        if self.metric == None:
            archs = self.random_splits(arch_parameters, 0, self.edges, seed=self.seed)
        else:
            archs = self.metric_splits(arch_parameters, 0, self.edges, trainer=self.trainer, warmup_epochs=self.warmup_epochs)

        for it in range(self.n_iterations - 1):
            archs = list(chain.from_iterable(archs))    #Un-nest architectures
        if return_splits:
            return(archs, self.groupings)
        return(archs)

    #Construct trimmed architecture adjacency matrices from the previous step matrix and a list of operation groupings
    def construct_archs(self, arch_parameters, splits, edge):

            n_splits = len(splits)
            archs = [deepcopy(arch_parameters) for k in range(n_splits)]
            for i in range(len(archs)):
                arch = archs[i]
                arch[edge, :] = -INF
                for op in splits[i]:
                    archs[i][edge, op] = 0.

            return(archs)

    #Construct n_splits operation groupings with random selection from a set of operations, such that groups are balanced 
    def construct_random_op_splits(self, ops, n_splits, seed):

        random.seed(seed)
        random.shuffle(ops)

        splits = [[] for k in range(n_splits)]
        for o in range(len(ops)-len(splits)):
            ch = random.choice(range(len(splits)))
            splits[ch].append(ops[o])

        print(splits, o)

        empty_splits = [s for s in splits if len(s)==0]
        print(empty_splits)
        for s, split in enumerate(empty_splits):
            split.append(ops[o+s+1])

        print(splits)

        for p in range(o+len(empty_splits)+1, len(ops)):
            ch = random.choice(range(len(splits)))
            splits[ch].append(ops[p])

        print(splits)

        return(splits)

        """
        splits = [[] for k in range(n_splits)]
        n = 0
        while len(ops) > 0:
            op = random.choice(ops)
            splits[n].append(op)
            n+=1
            if n == n_splits:
                n =  0
            ops.remove(op)
        return(splits)
        """

    #Recursively split an adjacency matrix until desired number of leaves, with random selection of the groupings
    def random_splits(self, arch_parameters, index, edges = None, it: int = 0, seed=0):
        """When no metric is provided, default to random splitting of architecture adjacency matrix (aam) along provided edges. 

        Args:
            arch_parameters (np.array): aam  
            index (_type_): _description_
            edges (_type_, optional): list of edges to split over. Defaults to None.
            it (int, optional): recursion parameter, denotes depth of recursion. Defaults to 0.
        """

        n_edges = arch_parameters.shape[0]
        n_ops = arch_parameters.shape[1]
        random.seed(seed)
        print(seed)

        if it+1 not in self.splits.keys():              #Break case : arrived at leaf

            seed_it = random.choice(list(range(10000)))
            random.seed(seed_it)

            if self.edges == None:                      #Draw the edge randomly

                if edges == None:                       #Case where edges is not yet created
                    edges = list(range(n_edges))
                c_edge = random.choice(edges)

            else:                                       #Draw the next edge in order
                c_edge = self.edges[it]

            #Operations that have already been trimmed are not included in the operation subset
            choice_ops = [idx for idx in range(n_ops) if arch_parameters[c_edge, idx]>-INF]
            n_splits = self.splits[it][index]

            splits = self.construct_random_op_splits(choice_ops, n_splits, seed)
            if it not in self.groupings.keys():
                self.groupings[it] = [splits]
            else:
                self.groupings[it].append(splits)

            #End architectures, no further trimming is done
            return(self.construct_archs(arch_parameters, splits, c_edge))

        else:                                           #Recursion step

            seed_it = random.choice(list(range(10000)))
            random.seed(seed_it)

            if self.edges == None:                      #Draw the edge randomly

                if edges == None:                       #Case where edges is not yet created
                    edges = list(range(n_edges))
                c_edge = random.choice(edges)
                edges.remove(c_edge)

            else:                                       #Draw the next edge in order
                c_edge = self.edges[it]

            #Operations that have already been trimmed are not included in the operation subset
            choice_ops = [idx for idx in range(n_ops) if arch_parameters[c_edge, idx]>-INF]
            n_splits = self.splits[it][index]

            splits = self.construct_random_op_splits(choice_ops, n_splits, seed)
            if it not in self.groupings.keys():
                self.groupings[it] = [splits]
            else:
                self.groupings[it].append(splits)

            #Intermediate architectures, are passed to the next recursion step
            #index : index of the architecture in the splits dictionary (this is for getting the corresponding branching factor)
            #edges : the subset of remaining edges to draw from at the next step
            #iteration : current iteration number, used as a stopping criterion
            archs = self.construct_archs(arch_parameters, splits, c_edge)
            return([self.random_splits(archs[idx], idx, deepcopy(edges), it+1, seed=random.choice(list(range(10000*(idx+1))))) for idx in range(n_splits)])

    def metric_splits(self, arch_parameters: np.array, index: int, edges = None, it: int = 0, trainer=None, warmup_epochs=[], parent_weights=None):
        """Recursively split an adjacency matrix until desired number of leaves, by clustering edges together based on a metric

        Args:
            arch_parameters (np.array): aam
            index (int): _description_
            edges (list[int], optional): list of edges to split over. Defaults to None.
            it (int, optional): recursion parameter, denotes depth of recursion. Defaults to 0.
        """

        n_edges = arch_parameters.shape[0]
        n_ops = arch_parameters.shape[1]
        random.seed(self.seed)

        if it+1 not in self.splits.keys():              #Break case : arrived at leaf

            if self.edges == None:                      #Draw the edge randomly

                if edges == None:                       #Case where edges is not yet created
                    edges = list(range(n_edges))
                c_edge = random.choice(edges)

            else:                                       #Draw the next edge in order
                c_edge = self.edges[it]

            #Operations that have already been trimmed are not included in the operation subset
            choice_ops = [idx for idx in range(n_ops) if arch_parameters[c_edge, idx]>-INF]
            n_splits = self.splits[it][index]

            if trainer!=None and len(warmup_epochs)>=1:
                trainer.instantiate_network(arch_parameters, parent_weights)
                weights = trainer.train_network(warmup_epochs[0])
            else:
                weights = None

            sep_arch_params = create_disjoint_architectures(arch_parameters, choice_ops, c_edge)                          #Create temp matrices with a single value pruned

            _, dist_matrix = self.metric(sep_arch_params, arch_parameters_reference=arch_parameters, weights_reference=weights)   #Create distance matrix based on a metric

            splits = clusterize(n_splits, dist_matrix, deepcopy(choice_ops))                           #Create splits by clustering edges together based on the distance matrix
            if it not in self.groupings.keys():
                self.groupings[it] = [splits]
            else:
                self.groupings[it].append(splits)
            print(splits)
            print(c_edge)

            #End architectures, no further trimming is done
            return(self.construct_archs(arch_parameters, splits, c_edge))

        else:                                           #Recursion step

            if self.edges == None:                      #Draw the edge randomly

                if edges == None:                       #Case where edges is not yet created
                    edges = list(range(n_edges))
                c_edge = random.choice(edges)
                edges.remove(c_edge)

            else:                                       #Draw the next edge in order
                c_edge = self.edges[it]

            #Operations that have already been trimmed are not included in the operation subset
            choice_ops = [idx for idx in range(n_ops) if arch_parameters[c_edge, idx]>-INF]
            n_splits = self.splits[it][index]

            if trainer!=None and len(warmup_epochs)>=1:
                trainer.instantiate_network(arch_parameters, parent_weights)
                weights = trainer.train_network(warmup_epochs[0])
            else:
                weights = None

            sep_arch_params = create_disjoint_architectures(arch_parameters, choice_ops, c_edge)                           #Create temp matrices with a single value prunned

            _, dist_matrix = self.metric(sep_arch_params, arch_parameters_reference=arch_parameters, weights_reference=weights)    #Create distance matrix based on a metric

            splits = clusterize(n_splits, dist_matrix, deepcopy(choice_ops))                            #Create splits by clustering edges together based on the distance matrix
            if it not in self.groupings.keys():
                self.groupings[it] = [splits]
            else:
                self.groupings[it].append(splits)
            print(splits)
            print(c_edge)

            #Create intermediate architectures and trim further
            archs = self.construct_archs(arch_parameters, splits, c_edge)
            if len(warmup_epochs)==1:
                warmup_epochs = []
            else:
                warmup_epochs = warmup_epochs[1:]
            return([self.metric_splits(archs[idx], idx, deepcopy(edges), it+1, trainer=trainer, warmup_epochs=warmup_epochs, parent_weights=weights) for idx in range(n_splits)])