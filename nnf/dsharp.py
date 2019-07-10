"""Interoperability with `DSHARP <https://bitbucket.org/haz/dsharp>`_.

``load`` and ``loads`` can be used to parse files created by DSHARP's
``-Fnnf`` option.

``compile`` invokes DSHARP directly to compile a sentence. This requires
having DSHARP installed.

The parser was derived by studying DSHARP's output and source code. This
format might be some sort of established standard, in which case this
parser might reject or misinterpret some valid files in the format.

DSHARP may not work properly for some (usually trivially) unsatisfiable
sentences, incorrectly reporting there's a solution. This bug dates back to
sharpSAT, on which DSHARP was based:

https://github.com/marcthurley/sharpSAT/issues/5

It was independently discovered by hypothesis during testing of this module.
"""

import io
import os
import subprocess
import tempfile
import typing as t

from nnf import NNF, And, Or, Var, false, dimacs, Name

__all__ = ('load', 'loads', 'compile')


def load(
        fp: t.TextIO, var_labels: t.Optional[t.Dict[int, Name]] = None
) -> NNF:
    """Load a sentence from an open file.

    An optional ``var_labels`` dictionary can map integers to other names.
    """

    def decode_name(num: int) -> Name:
        if var_labels is not None:
            return var_labels[num]
        return num

    fmt, nodecount, edges, varcount = fp.readline().split()
    node_specs = dict(enumerate(line.split() for line in fp))
    assert fmt == 'nnf'
    nodes = {}  # type: t.Dict[int, NNF]
    for num, spec in node_specs.items():
        if spec[0] == 'L':
            if spec[1].startswith('-'):
                nodes[num] = Var(decode_name(int(spec[1][1:])), False)
            else:
                nodes[num] = Var(decode_name(int(spec[1])))
        elif spec[0] == 'A':
            nodes[num] = And(nodes[int(n)] for n in spec[2:])
        elif spec[0] == 'O':
            nodes[num] = Or(nodes[int(n)] for n in spec[3:])
        else:
            raise ValueError("Can't parse line {}: {}".format(num, spec))
    if int(nodecount) == 0:
        raise ValueError("The sentence doesn't have any nodes.")
    return nodes[int(nodecount) - 1]


def loads(s: str, var_labels: t.Optional[t.Dict[int, Name]] = None) -> NNF:
    """Load a sentence from a string."""
    return load(io.StringIO(s), var_labels)


def compile(
        sentence: And[Or[Var]],
        executable: str = 'dsharp',
        smooth: bool = False,
        timeout: t.Optional[int] = None,
        extra_args: t.Sequence[str] = ()
) -> NNF:
    """Run DSHARP to compile a CNF sentence to (s)d-DNNF.

    This requires having DSHARP installed.

    :param sentence: The CNF sentence to compile.
    :param executable: The path of the ``dsharp`` executable. If the
                       executable is in your PATH there's no need to set this.
    :param smooth: Whether to produce a smooth sentence.
    :param timeout: Tell DSHARP to give up after a number of seconds.
    :param extra_args: Extra arguments to pass to DSHARP.
    """
    args = [executable]
    if smooth:
        args.append('-smoothNNF')
    if timeout is not None:
        args.extend(['-t', str(timeout)])
    args.extend(extra_args)

    if not sentence.is_CNF():
        raise ValueError("Sentence must be in CNF")

    var_labels = dict(enumerate(sentence.vars(), start=1))
    var_labels_inverse = {v: k for k, v in var_labels.items()}

    infd, infname = tempfile.mkstemp(text=True)
    try:
        with open(infd, 'w') as f:
            dimacs.dump(sentence, f, mode='cnf', var_labels=var_labels_inverse)
        outfd, outfname = tempfile.mkstemp()
        try:
            os.close(outfd)
            proc = subprocess.Popen(
                args + ['-Fnnf', outfname, infname],
                stdout=subprocess.PIPE,
                universal_newlines=True
            )
            log, _ = proc.communicate()
            with open(outfname) as f:
                out = f.read()
        finally:
            os.remove(outfname)
    finally:
        os.remove(infname)

    if proc.returncode != 0:
        raise RuntimeError(
            "DSHARP failed with code {}. Log:\n\n{}".format(
                proc.returncode, log
            )
        )

    if out == 'nnf 0 0 0\n' or 'problem line expected' in log:
        raise RuntimeError("Something went wrong. Log:\n\n{}".format(log))

    if 'TIMEOUT' in log:
        raise RuntimeError("DSHARP timed out after {} seconds".format(timeout))

    if 'Theory is unsat' in log:
        return false

    if not out:
        raise RuntimeError("Couldn't read file output. Log:\n\n{}".format(log))

    return loads(out, var_labels=var_labels)
