"""Transformations using the well-known `Tseitin encoding
<https://en.wikipedia.org/wiki/Tseytin_transformation>`_.

The Tseitin transformation converts any arbitrary circuit to one in CNF in
polynomial time/space. It does so at the cost of introducing new variables
(one for each logical connective in the formula).
"""

from nnf import NNF, Var, And, Or, Internal
from nnf.util import memoize


def to_CNF(theory: NNF) -> And[Or[Var]]:
    """Convert an NNF into CNF using the Tseitin Encoding.

    :param theory: Theory to convert.
    """

    clauses = []

    @memoize
    def process_node(node: NNF) -> Var:

        if isinstance(node, Var):
            return node

        assert isinstance(node, Internal)

        children = {process_node(c) for c in node.children}

        if len(children) == 1:
            [child] = children
            return child

        aux = Var.aux()

        if any(~var in children for var in children):
            if isinstance(node, And):
                clauses.append(Or({~aux}))
            else:
                clauses.append(Or({aux}))

        elif isinstance(node, And):
            clauses.append(Or({~c for c in children} | {aux}))
            for c in children:
                clauses.append(Or({~aux, c}))

        elif isinstance(node, Or):
            clauses.append(Or(children | {~aux}))
            for c in children:
                clauses.append(Or({~c, aux}))

        else:
            raise TypeError(node)

        return aux

    @memoize
    def process_required(node: NNF) -> None:
        """For nodes that have to be satisfied.

        This lets us perform some optimizations.
        """
        if isinstance(node, Var):
            clauses.append(Or({node}))
            return

        assert isinstance(node, Internal)

        if len(node.children) == 1:
            [child] = node.children
            process_required(child)

        elif isinstance(node, Or):
            children = {process_node(c) for c in node.children}
            if any(~var in children for var in children):
                return
            clauses.append(Or(children))

        elif isinstance(node, And):
            for child in node.children:
                process_required(child)

        else:
            raise TypeError(node)

    process_required(theory)
    ret = And(clauses)
    NNF._is_CNF_loose.set(ret, True)
    NNF._is_CNF_strict.set(ret, True)
    return ret
