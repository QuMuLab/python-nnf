"""NNF example based on https://arxiv.org/pdf/1807.06397.pdf"""

import collections
import functools
import itertools

import nnf

from nnf import amc, And, Or, Var

memoize = functools.lru_cache(None)  # huge speedup


def powerset(iterable):
    # https://docs.python.org/3/library/itertools.html#recipes
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def lin(candidates):
    # candidates must be hashable, have total ordering
    builder = nnf.Builder()

    candidates = frozenset(candidates)
    n = len(candidates)
    T = frozenset(powerset(candidates))
    assert len(T) == 2**n

    @memoize
    def defeats(i, j):
        assert i != j
        if i < j:
            return builder.Var((i, j))
        else:
            return builder.Var((j, i), False)

    @memoize
    def C(S):
        if S == candidates:
            return builder.true

        return builder.Or(C_child(i, S)
                          for i in candidates - S)

    @memoize
    def C_child(i, S):
        children = {C(S | {i})}
        children.update(defeats(i, j)
                        for j in candidates - S
                        if i != j)
        return builder.And(children)

    return C(frozenset())


def model_to_order(model):
    @functools.cmp_to_key
    def cmp(a, b):
        if (a, b) in model:
            return -1 if model[(a, b)] else 1
        return 1 if model[(b, a)] else -1

    return tuple(sorted({candidate
                         for pair in model.keys()
                         for candidate in pair}, key=cmp))


def order_to_model(order):
    model = {}
    for i, candidate in enumerate(order):
        for other in order[i + 1:]:
            if candidate < other:
                model[(candidate, other)] = True
            else:
                model[(other, candidate)] = False
    return model


def kemeny(votes):
    labels = collections.Counter()
    for vote in votes:
        for name, true in order_to_model(vote).items():
            labels[nnf.Var(name, true)] += 1
    return {model_to_order(model)
            for model in amc.maxplus_reduce(lin(votes[0]), labels).models()}


def slater(votes):
    totals = collections.Counter()
    for vote in votes:
        for name, true in order_to_model(vote).items():
            totals[nnf.Var(name, true)] += 1
    labels = {}
    for var in totals:
        labels[var] = 1 if totals[var] > totals[~var] else 0
    return {model_to_order(model)
            for model in amc.maxplus_reduce(lin(votes[0]), labels).models()}


def lin_top(candidates, k):
    candidates = frozenset(candidates)
    n = len(candidates)
    T = frozenset(subset for subset in powerset(candidates)
                  if len(subset) <= k)

    f_i = itertools.chain.from_iterable

    @memoize
    def C(S):
        if len(S) < k:
            return Or(C_child(i, S)
                      for i in candidates - S)
        return And(~Var((i, j))
                   for i in candidates - S
                   for j in candidates - S
                   if i != j)

    @memoize
    def C_child(i, S):
        return And(f_i((Var((i, j)), ~Var((j, i)), C(S | {i}))
                   for j in candidates - S
                   if i != j))

    return C(frozenset())


def test():
    s = lin(range(4))

    assert len(list(s.models())) == 24

    # unambiguous ordering
    s_1 = s.condition({(0, 1): True, (1, 2): True, (2, 3): True})
    assert list(s_1.models()) == [{(0, 2): True, (0, 3): True, (1, 3): True}]

    # ambiguous ordering
    s_2 = s.condition({(0, 1): True, (1, 2): True, (2, 3): False})
    assert len(list(s_2.models())) == 3

    # circular ordering
    s_3 = s.condition({(0, 1): True, (1, 2): True, (0, 2): False})
    assert len(list(s_3.models())) == 0
    assert not s_3.satisfiable()
    assert s_3.simplify() == nnf.false

    # strings as candidates
    named = lin({"Alice", "Bob", "Carol"})
    assert len(list(named.models())) == 6

    # AMC
    assert s.decomposable()
    assert s.deterministic()
    assert s.smooth()

    assert amc.NUM_SAT(s) == 24
    assert amc.SAT(s)

    assert amc.SAT(s_2)
    assert amc.NUM_SAT(s_2) == 3

    assert not amc.SAT(s_3)
    assert amc.NUM_SAT(s_3) == 0

    example_votes = [
        ('a', 'b', 'c', 'd'),
        ('a', 'b', 'c', 'd'),
        ('b', 'a', 'c', 'd'),
        ('b', 'a', 'c', 'd'),
        ('b', 'c', 'a', 'd'),
        ('b', 'c', 'a', 'd'),
        ('d', 'c', 'a', 'b'),
        ('d', 'c', 'a', 'b'),
        ('d', 'c', 'a', 'b'),
    ]

    assert kemeny(example_votes) == {
        ('b', 'c', 'a', 'd'),
        ('a', 'b', 'c', 'd'),
    }
    assert slater(example_votes) == {
        ('b', 'c', 'a', 'd'),
        ('a', 'b', 'c', 'd'),
        ('c', 'a', 'b', 'd')
    }

    s_top = lin_top(range(4), 2)
    assert s_top.decomposable()


if __name__ == '__main__':
    test()
