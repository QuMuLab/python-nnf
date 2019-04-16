"""NNF example based on https://arxiv.org/pdf/1807.06397.pdf"""

import functools
import itertools

from nnf import And, Or, Var, true, false

memoize = functools.lru_cache(None)  # huge speedup


def powerset(iterable):
    # https://docs.python.org/3/library/itertools.html#recipes
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def lin(candidates):
    # candidates must be hashable, have total ordering
    candidates = frozenset(candidates)
    n = len(candidates)
    T = frozenset(powerset(candidates))
    assert len(T) == 2**n

    @memoize
    def defeats(i, j):
        assert i != j
        if i < j:
            return Var((i, j))
        else:
            return Var((j, i), False)

    @memoize
    def C(S):
        if S == candidates:
            return true

        return Or(C_child(i, S)
                  for i in candidates - S)

    @memoize
    def C_child(i, S):
        return And({C(S | {i}),
                    *(defeats(i, j)
                      for j in candidates - S
                      if i != j)})

    return C(frozenset())


s = lin(range(4))

assert len(list(s.models())) == 24

# unambiguous ordering
s_1 = s.instantiate({(0, 1): True, (1, 2): True, (2, 3): True})
assert list(s_1.models()) == [{(0, 2): True, (0, 3): True, (1, 3): True}]

# ambiguous ordering
s_2 = s.instantiate({(0, 1): True, (1, 2): True, (2, 3): False})
assert len(list(s_2.models())) == 3

# circular ordering
s_3 = s.instantiate({(0, 1): True, (1, 2): True, (0, 2): False})
assert len(list(s_3.models())) == 0
assert not s_3.satisfiable()
assert s_3.simplify() == false

# strings as candidates
named = lin({"Alice", "Bob", "Carol"})
assert len(list(named.models())) == 6
