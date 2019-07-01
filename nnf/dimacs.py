"""A parser and serializer for the DIMACS
`CNF and SAT formats <https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html>`_."""

import collections
import io
import typing as t
import warnings

from nnf import NNF, Var, And, Or, Name, true, false

__all__ = ('dump', 'load', 'dumps', 'loads')


def dump(
        obj: NNF,
        fp: t.TextIO,
        *,
        num_variables: t.Optional[int] = None,
        var_labels: t.Optional[t.Dict[Name, int]] = None,
        comment_header: t.Optional[str] = None,
        mode: str = 'sat'
) -> None:
    """Dump a sentence into an open file in a DIMACS format.

    Variable names have to be integers. If the variables in the sentence you
    want to dump are not integers, you can pass a ``var_labels`` dictionary
    to map names to integers.

    :param obj: The sentence to dump.
    :param fp: The open file.
    :param num_variables: Override the number of variables, in case there
                          are variables that don't appear in the sentence.
    :param var_labels: A dictionary mapping variable names to integers,
                       to rename non-integer variables.
    :param comment_header: A comment to include at the top of the file. May
                           include newlines.
    :param mode: Either ``'sat'`` to dump in the general SAT format,
                 or ``'cnf'`` to dump in the specialized CNF format.
    """
    if num_variables is None:
        if var_labels is None:
            if t.TYPE_CHECKING:
                names = frozenset()  # type: t.FrozenSet[int]
            else:
                names = obj.vars()
            for name in names:
                if not isinstance(name, int) or name <= 0:
                    raise TypeError(
                        "{!r} is not an integer > 0. Try supplying a "
                        "var_labels dictionary.".format(name)
                    )
            num_vars = max(names, default=0)  # type: int
        else:
            num_vars = max(var_labels.values(), default=0)
    else:
        num_vars = num_variables

    if mode == 'sat':
        _dump_sat(obj, fp, num_variables=num_vars, var_labels=var_labels,
                  comment_header=comment_header)
    elif mode == 'cnf':
        _dump_cnf(obj, fp, num_variables=num_vars, var_labels=var_labels,
                  comment_header=comment_header)


def _write_comments(comment_header: str, fp: t.TextIO) -> None:
    for line in comment_header.split('\n'):
        fp.write('c ')
        fp.write(line)
        fp.write('\n')


def _format_var(
        node: Var,
        num_variables: int,
        var_labels: t.Optional[t.Dict[Name, int]] = None
) -> str:
    if var_labels is not None:
        name = var_labels[node.name]
    else:
        name = node.name  # type: ignore
    if not isinstance(name, int) or name <= 0:
        raise TypeError("{!r} is not an integer > 0".format(name))
    if name > num_variables:
        raise ValueError(
            "{!r} is more than num_variables".format(name)
        )
    if not node.true:
        return "-{}".format(name)
    return str(name)


def _dump_sat(
        obj: NNF,
        fp: t.TextIO,
        *,
        num_variables: int,
        var_labels: t.Optional[t.Dict[Name, int]] = None,
        comment_header: t.Optional[str] = None
) -> None:
    if comment_header is not None:
        _write_comments(comment_header, fp)

    fp.write("p sat {}\n".format(num_variables))

    def serialize(node: NNF) -> None:
        if isinstance(node, Var):
            fp.write(_format_var(node, num_variables, var_labels))
        elif isinstance(node, (Or, And)):
            fp.write('+(' if isinstance(node, Or) else '*(')
            first = True
            for child in node.children:
                if first:
                    first = False
                else:
                    fp.write(' ')
                serialize(child)
            fp.write(')')
        else:
            raise TypeError("Can't serialize type {}".format(type(node)))

    fp.write('(')
    serialize(obj)
    fp.write(')')


def _dump_cnf(
        obj: NNF,
        fp: t.TextIO,
        *,
        num_variables: int,
        var_labels: t.Optional[t.Dict[Name, int]] = None,
        comment_header: t.Optional[str] = None
) -> None:
    if not isinstance(obj, And):
        raise TypeError("CNF sentences must be conjunctions")

    if comment_header is not None:
        _write_comments(comment_header, fp)

    fp.write("p cnf {} {}\n".format(num_variables, len(obj.children)))

    first = True
    for clause in obj.children:
        if not isinstance(clause, Or):
            raise TypeError("CNF sentences must be conjunctions of "
                            "disjunctions")
        if not len(clause.children) > 0:
            raise TypeError("CNF sentences shouldn't have empty clauses")
        if not first:
            fp.write('0')
        else:
            first = False
        for child in clause.children:
            if not isinstance(child, Var):
                raise TypeError("CNF sentences must be conjunctions of "
                                "disjunctions of variables")
            fp.write(' ')
            fp.write(_format_var(child, num_variables, var_labels))
        fp.write('\n')


def dumps(
        obj: NNF,
        *,
        num_variables: t.Optional[int] = None,
        var_labels: t.Optional[t.Dict[Name, int]] = None,
        comment_header: t.Optional[str] = None,
        mode: str = 'sat'
) -> str:
    """Like :func:`dump`, but to a string instead of to a file."""
    buffer = io.StringIO()
    dump(obj, buffer, num_variables=num_variables, var_labels=var_labels,
         comment_header=comment_header, mode=mode)
    return buffer.getvalue()


def load(fp: t.TextIO) -> NNF:
    """Load a sentence from an open file.

    The format is automatically detected.
    """
    for line in fp:
        if line.startswith('c'):
            continue
        if line.startswith('p '):
            problem = line.split()
            if len(line) < 2:
                raise ValueError("Malformed problem line")
            fmt = problem[1]
            if 'sat' in fmt or 'SAT' in fmt:
                # problem[2] contains the number of variables
                # but that's currently not explicitly represented
                return _load_sat(fp)
            elif 'cnf' in fmt or 'CNF' in fmt:
                # problem[2] has the number of variables
                # problem[3] has the number of clauses
                return _load_cnf(fp)
            else:
                raise ValueError("Unknown format '{}'".format(fmt))
        else:
            print(repr(line))
            raise ValueError(
                "Couldn't find a problem line before an unknown kind of line"
            )
    else:
        raise ValueError(
            "Couldn't find a problem line before the end of the file"
        )


def loads(s: str) -> NNF:
    """Like :func:`load`, but from a string instead of from a file."""
    return load(io.StringIO(s))


def _load_sat(fp: t.TextIO) -> NNF:
    tokens = collections.deque()  # type: t.Deque[str]
    for line in fp:
        if line.startswith('c'):
            continue
        tokens.extend(
            line.replace('(', '( ')
                .replace(')', ' ) ')
                .replace('+(', ' +(')
                .replace('*(', ' *(')
                .replace('-', ' - ')
                .split()
        )
    result = _parse_sat(tokens)
    if tokens:
        warnings.warn("Found extra tokens past the end of the sentence")
    return result


def _parse_sat(tokens: 't.Deque[str]') -> NNF:
    cur = tokens.popleft()
    if cur == '(':
        content = _parse_sat(tokens)
        close = tokens.popleft()
        if close != ')':
            raise ValueError("Expected closing paren, found {!r}"
                             .format(close))
        return content
    elif cur == '-':
        content = _parse_sat(tokens)
        if not isinstance(content, Var):
            raise ValueError("Only variables can be negated, not {!r}"
                             .format(content))
        return ~content
    elif cur == '*(':
        children = []
        while tokens[0] != ')':
            children.append(_parse_sat(tokens))
        tokens.popleft()
        if children:
            return And(children)
        else:
            return true
    elif cur == '+(':
        children = []
        while tokens[0] != ')':
            children.append(_parse_sat(tokens))
        tokens.popleft()
        if children:
            return Or(children)
        else:
            return false
    elif cur.isdigit():
        return Var(int(cur))
    else:
        raise ValueError("Found unexpected token {!r}".format(cur))


def _load_cnf(fp: t.TextIO) -> NNF:
    tokens = []  # type: t.List[str]
    for line in fp:
        if line.startswith('c'):
            continue
        tokens.extend(
            line.replace('-', ' -')
                .split()
        )
    return _parse_cnf(tokens)


def _parse_cnf(tokens: t.Iterable[str]) -> NNF:
    clauses = set()  # type: t.Set[Or]
    clause = set()  # type: t.Set[Var]
    for token in tokens:
        if token == '0':
            if clause:
                clauses.add(Or(clause))
            clause = set()
        elif token == '%':
            # Some example files end with:
            # 0
            # %
            # 0
            # I don't know why.
            pass
        elif token.startswith('-'):
            clause.add(Var(int(token[1:]), False))
        else:
            clause.add(Var(int(token)))
    if clause:
        # A file may or may not end with a 0
        # Adding an empty clause is not desirable
        clauses.add(Or(clause))
    return And(clauses)
