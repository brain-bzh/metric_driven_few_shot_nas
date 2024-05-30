import pytest
from .split_utils import ascending_split, create_disjoint_architectures
import numpy as np

class TestAscendingSplit:

    MAX_LEAVES = 32
    save_attainable = {}

    @pytest.fixture(
        params=range(2,6)
    )
    def branch_factor(self, request):
        return request.param

    @pytest.fixture(
        params=range(2, MAX_LEAVES)
    )
    def total_leaves(self, request):
        return request.param 

    def test_ascending_split(self, total_leaves, branch_factor):
        """Many of these tests should get the AssertionError: Incompatible Parameters. This is expected behavior. 

        This is based on the fact that only 1 or branch_factor number of branches is allowed to spawn a new child.
        If the total_leaves is not something that can be acheived by spawning new children with this limitation then 
        the assertion is thrown.

        We account for it by checking if total_leaves `is_attainable()`.

        Args:
            total_leaves (_type_): _description_
            branch_factor (_type_): _description_
        """
        if (total_leaves >= branch_factor) and (total_leaves in self.is_attainable(branch_factor)):
            base = branch_factor
            _n_iterations = np.ceil(np.log(total_leaves) / np.log(base))

            bf = [branch_factor]
            splits, n_iterations, n_nodes = ascending_split(total_leaves, bf)

            _total_leaves = 0
            for key, value in splits[len(n_nodes)-1].items():
                _total_leaves+=value

            assert n_iterations == _n_iterations, "n_iterations doesn't match tree level"
            assert all([node == branch_factor**i for i, node in enumerate(n_nodes)]), "n_nodes doesn't match branch_factor"
            assert total_leaves == _total_leaves, "Number of leaves in splits doesn't match total leaves"

    def is_attainable(self, bf):
        if not self.save_attainable.get(bf):
            self.save_attainable[bf] = set([bf + val for val in range(0, self.MAX_LEAVES, bf-1)])
        return self.save_attainable.get(bf)

class TestCreateDisjointArchitectures:
    INF = -1000

    @pytest.mark.parametrize(
        ["arch_parameters", "ops", "edge"],
        [
            [np.zeros((6,5)), np.arange(5), 0], #vary edge
            [np.zeros((6,5)), np.arange(5), 3], #vary edge
            [np.zeros((6,5)), np.arange(5), 5], #vary edge
            [np.zeros((6,5)), np.arange(0, 5, 2), 0], #vary num ops
            [np.zeros((6,5)), np.arange(0, 5, 3), 0], #vary num ops
            [np.zeros((6,5)), np.arange(0), 0], #vary num ops
            [np.zeros((3,5)), np.arange(5), 0], # vary size arch_params
            [np.zeros((6,3)), np.arange(3), 0], # vary size arch_params
        ]
    )
    def test_all_ops(self, arch_parameters, ops, edge):
        n_edge, n_ops = arch_parameters.shape
        disjoint_archs = create_disjoint_architectures(arch_parameters, ops, edge)

        assert len(disjoint_archs) == len(ops) # created correct number of disjoint architectures
        assert all([arch.shape == arch_parameters.shape for arch in disjoint_archs]) # same size as arch_parametrs
        assert all([arch[edge, :].sum() == (n_ops-1)*self.INF for arch in disjoint_archs]) # all ops except one zero'd out
        assert all([arch.sum() == (n_ops-1)*self.INF for arch in disjoint_archs]) # only 1 edge was manipulated
        assert all([arch[:, op].sum() == 0 for op, arch in zip(ops, disjoint_archs)]) # the correct op is kept fixed

