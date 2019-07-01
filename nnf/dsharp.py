"""A parser for `DSHARP <https://bitbucket.org/haz/dsharp>`_'s output format.

Derived by closely studying its output and source code. This format might
be some sort of established standard, in which case this parser might
reject or misinterpret some valid files in the format.
"""

import io
import typing as t

from nnf import NNF, And, Or, Var

__all__ = ('load', 'loads')


def load(fp: t.TextIO) -> NNF:
    """Load a sentence from an open file."""
    fmt, nodecount, edges, varcount = fp.readline().split()
    node_specs = dict(enumerate(line.split() for line in fp))
    assert fmt == 'nnf'
    nodes = {}  # type: t.Dict[int, NNF]
    for num, spec in node_specs.items():
        if spec[0] == 'L':
            if spec[1].startswith('-'):
                nodes[num] = Var(int(spec[1][1:]), False)
            else:
                nodes[num] = Var(int(spec[1]))
        elif spec[0] == 'A':
            nodes[num] = And(nodes[int(n)] for n in spec[2:])
        elif spec[0] == 'O':
            nodes[num] = Or(nodes[int(n)] for n in spec[3:])
        else:
            raise ValueError("Can't parse line {}: {}".format(num, spec))
    return nodes[int(nodecount) - 1]


def loads(s: str) -> NNF:
    """Load a sentence from a string."""
    return load(io.StringIO(s))
