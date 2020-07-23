"""Transformations using the well-known Tseitin encoding <https://en.wikipedia.org/wiki/Tseytin_transformation>

The Tseitin transformation converts any arbitrary circuit to one in CNF in
polynomial time/space. It does so at the cost of introducing new variables
(one for each logical connective in the formula).
"""

from nnf import NNF, Var, And, Or, Name, true, false

def to_cnf(theory: NNF) -> NNF:

    node_to_var = {}
    clauses = []

    def process_node(node: NNF) -> NNF:

        if node in node_to_var:
            return node

        if isinstance(node, Var):
            node_to_var[node] = node
            return node

        aux = Var("aux_%d" % len(node_to_var))
        node_to_var[node] = aux
        children = [process_node(c) for c in node.children]

        if isinstance(node, And):
            clauses.append(Or([~c for c in children] + [aux]))
            for c in children:
                clauses.append(Or([~aux, c]))

        elif isinstance(node, Or):
            clauses.append(Or(list(children)+[~aux]))
            for c in children:
                clauses.append(Or([~c, aux]))

        else:
            raise TypeError(node)

        return aux

    root = process_node(theory)
    clauses.append(root)

    return And(clauses)

