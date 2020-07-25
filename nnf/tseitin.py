"""Transformations using the well-known Tseitin encoding
<https://en.wikipedia.org/wiki/Tseytin_transformation>

The Tseitin transformation converts any arbitrary circuit to one in CNF in
polynomial time/space. It does so at the cost of introducing new variables
(one for each logical connective in the formula).
"""

import typing as t

from nnf import NNF, Var, And, Or, memoize, Internal


aux_count = 0


def Aux():
    global aux_count
    aux_count += 1
    return Var("aux_%d" % aux_count)


def to_CNF(theory: NNF) -> And[Or[Var]]:
    """Convert an NNF into CNF using the Tseitin Encoding.

    Assumes that the theory is simplified.

    :param theory: Theory to convert.
    """

    clauses: t.List[Or[Var]] = []

    @memoize
    def process_node(node: NNF) -> Var:

        if isinstance(node, Var):
            return node

        assert isinstance(node, Internal)

        aux = Aux()
        children = {process_node(c) for c in node.children}

        if isinstance(node, And):
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

    root = process_node(theory)
    clauses.append(Or({root}))

    return And(clauses)
