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
MultiBlock: one block of a multiclass SLIM-GSGP individual.

A block holds K per-class components (one per class) and a binary mask m in {0,1}^K.
    - base block (index 0): K GP trees, mask = [1]*K
    - delta block (created by inflate): for each active class k, a SLIM delta
      (same structure as the original SLIM block, i.e. Tree([variator, *random_trees, ms]));
      inactive classes hold None and contribute the neutral element of the operator
      (0 for "sum", 1 for "mul").

Semantics of a block are stored with shape (K, n).
"""
from types import SimpleNamespace

import torch
from slim_gsgp.algorithms.GSGP.representations.tree_utils import apply_tree

_FUNCTION_SYMBOLS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}


def neutral_element(operator):
    """Neutral element of the block aggregation operator: 0 for 'sum', 1 for 'mul'."""
    return 0.0 if operator == "sum" else 1.0


def _as_vector(sem, n):
    """Constant trees return a scalar; repeat it so every component has shape (n,)."""
    return sem.repeat(n) if sem.shape == torch.Size([]) else sem


def _is_logistic(variator):
    """Whether the random trees of a delta go through a sigmoid (mirrors inflate_mutation)."""
    # tt_delta_* (two trees) -> always sigmoid; ot_delta_<op>_<sig> -> sigmoid iff sig is True
    return not variator.__name__.endswith("_False")


def infix(structure, feature_names=None):
    """Readable infix string of a GP tree stored as a nested tuple."""
    if isinstance(structure, tuple):
        name = structure[0]
        if len(structure) == 3:
            sym = _FUNCTION_SYMBOLS.get(name)
            left, right = infix(structure[1], feature_names), infix(structure[2], feature_names)
            return f"({left} {sym} {right})" if sym else f"{name}({left}, {right})"
        return f"{name}({infix(structure[1], feature_names)})"
    if isinstance(structure, str) and structure.startswith("constant_"):
        return structure.replace("constant_", "").replace("_", "-")
    if feature_names is not None and isinstance(structure, str) and structure.startswith("x"):
        return str(feature_names[int(structure[1:])])
    return str(structure)


class MultiBlock:
    """
    One block of a multiclass SLIM individual.

    Parameters
    ----------
    components : list
        List of length K. Each element is a GSGP Tree (tuple structure for the base block,
        [variator, *random_trees, ms] for a delta) or None for an inactive class.
    mask : list of int
        Binary vector of length K; mask[k] == 1 iff the block affects class k.
    operator : {"sum", "mul"}
        Aggregation operator of the SLIM version.
    """

    def __init__(self, components, mask, operator="sum"):
        assert len(components) == len(mask), "components and mask must have length K"
        assert all((c is None) == (m == 0) for c, m in zip(components, mask)), \
            "a component must be None exactly when its mask bit is 0"
        self.components = components
        self.mask = [int(m) for m in mask]
        self.operator = operator
        self.n_classes = len(components)
        self.is_base = all(c is not None and isinstance(c.structure, tuple) for c in components)
        self.nodes = [c.nodes if c is not None else 0 for c in components]
        self.depths = [c.depth if c is not None else 0 for c in components]
        self.train_semantics = None  # (K, n)
        self.test_semantics = None   # (K, n)

    def _stack(self, testing, n):
        rows, ref = [], None
        for c in self.components:
            if c is not None:
                ref = _as_vector(c.test_semantics if testing else c.train_semantics, n)
                break
        for c in self.components:
            if c is None:
                rows.append(torch.full((n,), neutral_element(self.operator), dtype=ref.dtype))
            else:
                rows.append(_as_vector(c.test_semantics if testing else c.train_semantics, n))
        return torch.stack(rows)

    def calculate_semantics(self, inputs, testing=False):
        """Compute (and cache) the (K, n) semantics of the block."""
        if testing and self.test_semantics is None:
            for c in self.components:
                if c is not None:
                    c.calculate_semantics(inputs, testing=True)
            self.test_semantics = self._stack(True, len(inputs))
        elif not testing and self.train_semantics is None:
            for c in self.components:
                if c is not None:
                    c.calculate_semantics(inputs, testing=False)
            self.train_semantics = self._stack(False, len(inputs))

    def predict(self, data):
        """Recompute the block output on new data, without touching cached semantics. Shape (K, n)."""
        n, rows = len(data), []
        for c in self.components:
            if c is None:
                rows.append(torch.full((n,), neutral_element(self.operator)))
            elif isinstance(c.structure, tuple):
                rows.append(_as_vector(apply_tree(c, data), n))
            else:
                variator, random_trees, ms = c.structure[0], c.structure[1:-1], c.structure[-1]
                logistic = _is_logistic(variator)
                proxies = []
                for rt in random_trees:
                    out = apply_tree(rt, data)
                    proxies.append(SimpleNamespace(train_semantics=torch.sigmoid(out) if logistic else out))
                rows.append(_as_vector(variator(*proxies, ms, testing=False), n))
        return torch.stack([r.float() for r in rows])

    def component_repr(self, k, feature_names=None):
        """Readable expression of the component of class k (None if inactive)."""
        c = self.components[k]
        if c is None:
            return None
        if isinstance(c.structure, tuple):
            return infix(c.structure, feature_names)
        variator, rts, ms = c.structure[0], c.structure[1:-1], c.structure[-1]
        t = [infix(rt.structure, feature_names) for rt in rts]
        ms = float(ms)
        if len(rts) == 2:
            core = f"{ms:.4f}*(sig({t[0]}) - sig({t[1]}))"
        elif _is_logistic(variator):
            core = f"{ms:.4f}*(2*sig({t[0]}) - 1)"
        else:
            core = f"{ms:.4f}*(1 - 2/(1 + |{t[0]}|))"
        return core if self.operator == "sum" else f"(1 + {core})"
