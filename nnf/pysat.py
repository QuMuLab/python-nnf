import importlib
import typing as t

from nnf import And, Or, Var, config, true
from nnf.util import Model, Name

try:
    from pysat.solvers import Solver
except ImportError:
    available = False
    try:
        importlib.import_module("pysat")
    except ImportError:
        _wrong_pysat = False
    else:
        _wrong_pysat = True
else:
    #: Indicates whether the PySAT library is installed and available for use.
    available = True
    _wrong_pysat = False


__all__ = ("satisfiable", "solve", "models", "available")


def _encode_CNF(
    sentence: And[Or[Var]],
) -> t.Tuple[t.List[t.List[int]], t.Dict[int, Name]]:
    """Encode a CNF sentence into a list of lists of ints fit for PySAT."""

    if not sentence.is_CNF():
        raise ValueError("Sentence must be in CNF")

    decode = dict(enumerate(sentence.vars(), start=1))
    encode = {v: k for k, v in decode.items()}

    clauses = [
        [encode[var.name] if var.true else -encode[var.name] for var in clause]
        for clause in sentence
    ]

    return clauses, decode


def _solver_for(
    sentence: And[Or[Var]], name: t.Optional[str] = None
) -> t.Tuple["Solver", t.Dict[int, Name]]:
    """Return a Solver loaded with the sentence and a dictionary to decode it.

    Callers of this function MUST use the solver as a context manager or ensure
    that ``.delete()`` is called on it, to prevent memory leaks.
    """

    if not available:
        if _wrong_pysat:
            raise RuntimeError(
                "There is a `pysat` module, but it isn't PySAT. "
                "Did you pip install `pysat` instead of `python-sat`?"
            )
        raise RuntimeError(
            "`pysat` is not installed. Try `pip install python-sat`?"
        )

    if name is None:
        name = config.pysat_solver

    clauses, decode = _encode_CNF(sentence)
    solver = Solver(bootstrap_with=clauses, name=name)

    return solver, decode


def satisfiable(sentence: And[Or[Var]]) -> bool:
    """Return whether a CNF sentence is satisfiable."""
    if sentence == true:
        # Feeding this to MapleSAT causes a segfault
        return True
    solver, _ = _solver_for(sentence)
    with solver:
        return solver.solve()


def solve(sentence: And[Or[Var]]) -> t.Optional[Model]:
    """Return a model of a CNF sentence, or ``None`` if unsatisfiable."""
    if sentence == true:
        return {}
    solver, decode = _solver_for(sentence)
    with solver:
        if not solver.solve():
            return None
        return {decode[abs(num)]: num > 0 for num in solver.get_model()}


def models(sentence: And[Or[Var]]) -> t.Iterator[Model]:
    """Yield all models of a CNF sentence."""
    if sentence == true:
        yield {}
        return
    solver, decode = _solver_for(sentence)
    with solver:
        if not solver.solve():
            return
        for model in solver.enum_models():
            yield {decode[abs(num)]: num > 0 for num in model}
