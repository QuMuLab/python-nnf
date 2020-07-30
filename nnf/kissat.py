"""Interoperability with `kissat <http://fmv.jku.at/kissat/>`_.

``solve`` invokes the SAT solver directly on the given theory.
"""

import io
import os
import subprocess
import tempfile
import typing as t

from nnf import NNF, And, Or, Var, dimacs, Model

__all__ = ('solve')



def solve(
        sentence: And[Or[Var]],
        extra_args: t.Sequence[str] = []
) -> t.Optional[Model]:
    """Run kissat to compute a satisfying assignment.

    :param sentence: The CNF sentence to solve.
    :param extra_args: Extra arguments to pass to kissat.
    """

    if not sentence.is_CNF():
        raise ValueError("Sentence must be in CNF")

    SOLVER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'bin', 'kissat')
    assert os.path.isfile(SOLVER), "Cannot seem to find kissat solver."

    args = [SOLVER] + extra_args

    var_labels = dict(enumerate(sentence.vars(), start=1))
    var_labels_inverse = {v: k for k, v in var_labels.items()}

    infd, infname = tempfile.mkstemp(text=True)
    try:
        with open(infd, 'w') as f:
            dimacs.dump(sentence, f, mode='cnf', var_labels=var_labels_inverse)

        proc = subprocess.Popen(
            args + [infname],
            stdout=subprocess.PIPE,
            universal_newlines=True
        )
        log, _ = proc.communicate()

    finally:
        os.remove(infname)

    # Two known exit codes for the solver
    if proc.returncode not in [10, 20]:
        raise RuntimeError(
            "kissat failed with code {}. Log:\n\n{}".format(
                proc.returncode, log
            )
        )

    if 's UNSATISFIABLE' in log:
        return None

    if 's SATISFIABLE' not in log:
        raise RuntimeError("Something went wrong. Log:\n\n{}".format(log))


    model = log.split('\nv ')[1].split('\n')[0] # Line that starts with 'v ...'
    model = model[:-2] # Strip off the final '0'
    model = map(int, model.strip().split(' ')) # Individual numbered literals
    model = {var_labels_inverse[abs(lit)]: lit>0 for lit in model}

    return model
