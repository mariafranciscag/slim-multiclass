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
"""MCPopulation: population of multiclass SLIM-GSGP individuals."""
from joblib import Parallel, delayed


def _evaluate_mc_individual(individual, ffunction, y, testing=False):
    return individual.evaluate(ffunction, y, testing=testing)


class MCPopulation:
    def __init__(self, population):
        self.population = population
        self.size = len(population)
        self.nodes_count = sum(ind.nodes_count for ind in population)
        self.fit = None

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.population[item]

    def calculate_semantics(self, inputs, testing=False):
        for ind in self.population:
            ind.calculate_semantics(inputs, testing)

    def evaluate(self, ffunction, y, n_jobs=1):
        if n_jobs == 1:
            self.fit = [ind.evaluate(ffunction, y) for ind in self.population]
        else:
            self.fit = Parallel(n_jobs=n_jobs)(
                delayed(_evaluate_mc_individual)(ind, ffunction, y) for ind in self.population
            )
            for ind, f in zip(self.population, self.fit):
                ind.fitness = f
