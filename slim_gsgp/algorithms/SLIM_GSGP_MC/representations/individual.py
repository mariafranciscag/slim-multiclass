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
MCIndividual: multiclass SLIM-GSGP individual.

The individual is a list of MultiBlocks. Stacked semantics have shape (B, K, n);
aggregating over blocks (sum or prod over dim 0) gives (K, n), transposed to the
(n, K) logit matrix used by the fitness function, softmax and argmax.

    additive:        z_k(x) = T_{0,k}(x) + sum_b m_{b,k} * Delta_{b,k}(x)
    multiplicative:  z_k(x) = T_{0,k}(x) * prod_b (1 + m_{b,k} * Delta_{b,k}(x))
"""
import math

import dill
import torch

LOGIT_CLAMP = 1e12


def _class_depth(depths):
    """Depth of one class expression; same formula as the original SLIM Individual."""
    return max(d - (i - 1) if i != 0 else d for i, d in enumerate(depths)) + (len(depths) - 1)


class MCIndividual:
    """
    Multiclass SLIM-GSGP individual.

    Parameters
    ----------
    collection : list of MultiBlock
        collection[0] is the base block; the remaining ones are delta blocks.
    train_semantics, test_semantics : torch.Tensor or None
        Stacked block semantics of shape (B, K, n), or None (computed lazily).
    operator : {"sum", "mul"}
        Aggregation operator of the SLIM version.
    """

    def __init__(self, collection, train_semantics=None, test_semantics=None, operator="sum"):
        self.collection = collection
        self.operator = operator
        self.n_classes = collection[0].n_classes
        self.size = len(collection)
        self.train_semantics = train_semantics
        self.test_semantics = test_semantics
        self.fitness = None
        self.test_fitness = None
        self.version = None
        self._compute_size_attributes()

    def _compute_size_attributes(self):
        """Per-class and total size/depth (interpretability measures)."""
        self.nodes_per_class, self.depth_per_class, self.blocks_per_class = [], [], []
        for k in range(self.n_classes):
            active = [b for b in self.collection if b.mask[k]]
            self.blocks_per_class.append(len(active))
            # nodes of each active block + one joining operator between consecutive blocks
            self.nodes_per_class.append(sum(b.nodes[k] for b in active) + len(active) - 1)
            self.depth_per_class.append(_class_depth([b.depths[k] for b in active]))
        self.nodes_count = sum(self.nodes_per_class)
        self.depth = max(self.depth_per_class)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.collection[item]

    def calculate_semantics(self, inputs, testing=False):
        """Compute (and cache) the stacked (B, K, n) semantics."""
        if testing and self.test_semantics is None:
            for b in self.collection:
                b.calculate_semantics(inputs, testing=True)
            self.test_semantics = torch.stack([b.test_semantics for b in self.collection])
        elif not testing and self.train_semantics is None:
            for b in self.collection:
                b.calculate_semantics(inputs, testing=False)
            self.train_semantics = torch.stack([b.train_semantics for b in self.collection])

    def _aggregate(self, stacked):
        agg = torch.sum(stacked, dim=0) if self.operator == "sum" else torch.prod(stacked, dim=0)
        return torch.clamp(agg, -LOGIT_CLAMP, LOGIT_CLAMP).T  # (n, K)

    def logits(self, testing=False):
        """Logit matrix (n, K) from the cached semantics."""
        return self._aggregate(self.test_semantics if testing else self.train_semantics)

    def evaluate(self, ffunction, y, testing=False):
        """Evaluate with a classification fitness ffunction(y_true, logits). Non-finite -> +inf."""
        value = float(ffunction(y, self.logits(testing)))
        if not math.isfinite(value):
            value = float("inf")
        if testing:
            self.test_fitness = value
        else:
            self.fitness = value
        return value

    # prediction
    def predict_logits(self, data):
        """Logit matrix (n, K) for new data, recomputed from the stored structure."""
        return self._aggregate(torch.stack([b.predict(data) for b in self.collection]))

    def predict_proba(self, data):
        """Class probabilities (n, K) via softmax."""
        return torch.softmax(self.predict_logits(data), dim=1)

    def predict(self, data):
        """Predicted class labels (n,) via argmax."""
        return torch.argmax(self.predict_logits(data), dim=1)

    # representation
    def get_tree_representation(self, feature_names=None, class_names=None):
        """One readable expression per class."""
        op = " + " if self.operator == "sum" else " * "
        lines = []
        for k in range(self.n_classes):
            terms = [b.component_repr(k, feature_names) for b in self.collection if b.mask[k]]
            name = class_names[k] if class_names is not None else k
            lines.append(f"z[{name}] = " + op.join(terms))
        return "\n".join(lines)

    def print_tree_representation(self, feature_names=None, class_names=None):
        print(self.get_tree_representation(feature_names, class_names))

    def save_to_file(self, file_path):
        with open(file_path, "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def load_from_file(file_path):
        with open(file_path, "rb") as f:
            return dill.load(f)
