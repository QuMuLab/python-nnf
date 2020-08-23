import typing as t

import nnf

from nnf.util import T_NNF, Name


class Builder:
    """Automatically deduplicates NNF nodes as you make them, to save memory.

    Usage:

    >>> builder = Builder()
    >>> var = builder.Var('A')
    >>> var2 = builder.Var('A')
    >>> var is var2
    True

    As long as the Builder object exists, the nodes it made will be kept in
    memory. Make sure not to keep it around longer than you need.

    It's often a better idea to avoid creating nodes multiple times in the
    first place. That will save processing time as well as memory.

    If you use a Builder, avoid using operators. Even negating variables
    should be done with ``builder.Var(name, False)`` or they won't be
    deduplicated.
    """

    def __init__(self, seed: t.Iterable[nnf.NNF] = ()) -> None:
        """:param seed: Nodes to store for reuse in advance."""
        self.stored = {
            nnf.true: nnf.true,
            nnf.false: nnf.false,
        }  # type: t.Dict[nnf.NNF, nnf.NNF]
        for node in seed:
            self.stored[node] = node
        self.true = nnf.true
        self.false = nnf.false

    def Var(self, name: Name, true: bool = True) -> nnf.Var:
        ret = nnf.Var(name, true)
        return self.stored.setdefault(ret, ret)  # type: ignore

    def And(self, children: t.Iterable[T_NNF] = ()) -> nnf.And[T_NNF]:
        ret = nnf.And(children)
        return self.stored.setdefault(ret, ret)  # type: ignore

    def Or(self, children: t.Iterable[T_NNF] = ()) -> nnf.Or[T_NNF]:
        ret = nnf.Or(children)
        return self.stored.setdefault(ret, ret)  # type: ignore

    true = nnf.true
    false = nnf.false
