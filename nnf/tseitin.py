"""Transformations using the well-known Tseitin encoding
<https://en.wikipedia.org/wiki/Tseytin_transformation>

The Tseitin transformation converts any arbitrary circuit to one in CNF in
polynomial time/space. It does so at the cost of introducing new variables
(one for each logical connective in the formula).
"""

from nnf import NNF, Var, And, Or, memoize, Internal


aux_count = 0


def Aux() -> Var:
    global aux_count
    aux_count += 1
    return Var("aux_%d" % aux_count)


def to_CNF(theory: NNF, skip_simplification: bool = False) -> And[Or[Var]]:
    """Convert an NNF into CNF using the Tseitin Encoding.

    :param theory: Theory to convert.
    :param skip_simplification: If true, the final CNF will not be simplified
    """

    clauses = []

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
    theory = And(clauses)

    if skip_simplification:
        return theory
    else:
        return theory.simplify()
