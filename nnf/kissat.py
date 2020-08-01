"""Interoperability with `kissat <http://fmv.jku.at/kissat/>`_.

``solve`` invokes the SAT solver directly on the given theory.
"""

import os
import shutil
import subprocess
import typing as t

from nnf import And, Or, Var, dimacs
from nnf.util import Model

__all__ = ('solve',)


def solve(
        sentence: And[Or[Var]],
        extra_args: t.Sequence[str] = ()
) -> t.Optional[Model]:
    """Run kissat to compute a satisfying assignment.

    :param sentence: The CNF sentence to solve.
    :param extra_args: Extra arguments to pass to kissat.
    """

    if not sentence.is_CNF():
        raise ValueError("Sentence must be in CNF")

    if shutil.which('kissat') is not None:
        SOLVER = 'kissat'
    else:
        SOLVER = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'bin', 'kissat'
        )
    assert os.path.isfile(SOLVER), "Cannot seem to find kissat solver."

    args = [SOLVER]
    args.extend(extra_args)

    var_labels = dict(enumerate(sentence.vars(), start=1))
    var_labels_inverse = {v: k for k, v in var_labels.items()}

    cnf = dimacs.dumps(sentence, mode='cnf', var_labels=var_labels_inverse)

    try:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            universal_newlines=True
        )
        log, _ = proc.communicate(cnf)
    except OSError as err:
        if err.errno == 8:
            print("Error: Attempting to run the kissat binary on an")
            print("       incompatible system. Consider compiling kissat")
            print("       natively so it is accessible via the command line.")
        raise

    # Two known exit codes for the solver
    if proc.returncode not in [10, 20]:
        raise RuntimeError(
            "kissat failed with code {}. Log:\n\n{}".format(
                proc.returncode, log
            )
        )

    # Unsatisfiable
    if proc.returncode == 20:
        return None

    assert proc.returncode == 10, "Bad error code. Log:\n\n{}".format(log)

    variable_lines = [
        line[2:] for line in log.split("\n") if line.startswith("v ")
    ]
    literals = [int(num) for line in variable_lines for num in line.split()]
    assert literals[-1] == 0, "Last entry should be 0. Log:\n\n{}".format(log)
    literals.pop()
    model = {var_labels[abs(lit)]: lit > 0 for lit in literals}

    return model
