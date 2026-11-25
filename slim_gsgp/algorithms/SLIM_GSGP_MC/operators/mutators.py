# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Inflate and deflate mutations for multiclass SLIM-GSGP.

Inflate: samples a class mask m in {0,1}^K and appends a MultiBlock with one SLIM delta
(built with the ORIGINAL two_trees_delta / one_tree_delta variators) per active class.
Deflate: removes one random non-base block (from all classes), as in the original SLIM.
"""
import random

import torch
from slim_gsgp.algorithms.GSGP.representations.tree import Tree
from slim_gsgp.algorithms.SLIM_GSGP.operators.mutators import one_tree_delta, two_trees_delta
from slim_gsgp.algorithms.SLIM_GSGP_MC.representations.block import MultiBlock
from slim_gsgp.algorithms.SLIM_GSGP_MC.representations.individual import MCIndividual
from slim_gsgp.utils.utils import get_random_tree

MASK_MODES = ("bernoulli", "one_hot", "all")


def sample_mask(n_classes, mask_mode="bernoulli", p_mask=0.5):
    """
    Sample the class mask of a new block.

    bernoulli : each bit ~ Bernoulli(p_mask), resampled until at least one bit is 1
    one_hot   : exactly one class, chosen uniformly
    all       : every class
    """
    if mask_mode == "all":
        return [1] * n_classes
    if mask_mode == "one_hot":
        mask = [0] * n_classes
        mask[random.randrange(n_classes)] = 1
        return mask
    if mask_mode == "bernoulli":
        while True:
            mask = [int(random.random() < p_mask) for _ in range(n_classes)]
            if any(mask):
                return mask
    raise ValueError(f"mask_mode must be one of {MASK_MODES}")


def mc_inflate_mutation(FUNCTIONS, TERMINALS, CONSTANTS, n_classes, two_trees=True, operator="sum",
                        sig=False, mask_mode="bernoulli", p_mask=0.5):
    """
    Generate a multiclass inflate mutation.

    Parameters mirror slim_gsgp.algorithms.SLIM_GSGP.operators.mutators.inflate_mutation,
    plus n_classes, mask_mode and p_mask.
    """
    variator = two_trees_delta(operator=operator) if two_trees else one_tree_delta(operator=operator, sig=sig)
    logistic = True if two_trees else sig
    n_random_trees = 2 if two_trees else 1

    def inflate(individual, ms, X, max_depth=8, p_c=0.1, X_test=None, grow_probability=1, reconstruct=True):
        mask = sample_mask(n_classes, mask_mode, p_mask)
        components = []
        for k in range(n_classes):
            if not mask[k]:
                components.append(None)
                continue
            random_trees = [
                get_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, inputs=X, p_c=p_c,
                                grow_probability=grow_probability, logistic=logistic)
                for _ in range(n_random_trees)
            ]
            if X_test is not None:
                for rt in random_trees:
                    rt.calculate_semantics(X_test, testing=True, logistic=logistic)
            components.append(Tree(
                structure=[variator, *random_trees, ms],
                train_semantics=variator(*random_trees, ms, testing=False),
                test_semantics=variator(*random_trees, ms, testing=True) if X_test is not None else None,
                reconstruct=True,
            ))

        block = MultiBlock(components, mask, operator)
        block.calculate_semantics(X)
        if X_test is not None:
            block.calculate_semantics(X_test, testing=True)

        train = torch.cat([individual.train_semantics, block.train_semantics.unsqueeze(0)])
        test = (torch.cat([individual.test_semantics, block.test_semantics.unsqueeze(0)])
                if individual.test_semantics is not None and block.test_semantics is not None else None)
        return MCIndividual([*individual.collection, block], train, test, operator)

    return inflate


def mc_deflate_mutation(individual, reconstruct=True):
    """Remove one random delta block (the base block, index 0, is never removed)."""
    p = random.randint(1, individual.size - 1)
    collection = [*individual.collection[:p], *individual.collection[p + 1:]]
    train = torch.cat([individual.train_semantics[:p], individual.train_semantics[p + 1:]])
    test = (torch.cat([individual.test_semantics[:p], individual.test_semantics[p + 1:]])
            if individual.test_semantics is not None else None)
    return MCIndividual(collection, train, test, individual.operator)
