"""A command line interface for some of the package's functionality.

It can be used by invoking nnf as a module (``python3 -m nnf``) or by
running the ``pynnf`` script installed with the package.
"""

import argparse
import contextlib
import subprocess
import sys
import time
import typing as t

from types import SimpleNamespace

from nnf import NNF, dimacs

# todo: convert format
#       PI
#       model enumeration
#       ...


@contextlib.contextmanager
def timer(args: argparse.Namespace) -> t.Iterator[SimpleNamespace]:
    begin = time.monotonic()
    ns = SimpleNamespace(begin=begin, end=None, time=None)
    try:
        yield ns
    finally:
        ns.end = time.monotonic()
        ns.time = ns.end - ns.begin
    if args.verbose:
        print("Done after {:.3g} seconds.".format(ns.time))
        print()


def open_read(fname: str) -> t.TextIO:
    if fname == '-':
        return sys.stdin
    return open(fname)


def open_write(fname: str) -> t.TextIO:
    if fname == '-':
        return sys.stdout
    return open(fname, 'w')


def main(argv: t.Sequence[str] = sys.argv[1:]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print extra statistics.")
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="Suppress all non-essential output.")
    subparsers = parser.add_subparsers(title="subcommands")

    sat_parser = subparsers.add_parser(
        'sat',
        help="Test whether a sentence is satisfiable."
    )
    sat_parser.add_argument(
        'file', type=str, help="The file with the sentence to test."
    )
    sat_parser.set_defaults(func=sat)

    sharpsat_parser = subparsers.add_parser(
        'sharpsat', help="Count how many models a sentence has."
    )
    sharpsat_parser.add_argument(
        'file', type=str, help="The file with the sentence to test."
    )
    sharpsat_parser.add_argument(
        '-d', '--deterministic', help="Treat the sentence as deterministic.",
        action='store_true'
    )
    sharpsat_parser.set_defaults(func=sharpsat)

    info_parser = subparsers.add_parser(
        'info', help="View basic information about a sentence."
    )
    info_parser.add_argument(
        'file', type=str, help="The file with the sentence to inspect."
    )
    info_parser.set_defaults(func=info)

    draw_parser = subparsers.add_parser(
        'draw', help="Draw a sentence with graphviz."
    )
    draw_parser.add_argument(
        'file', type=str, help="The file with the sentence to draw."
    )
    draw_parser.add_argument(
        'out', type=str, help="The destination to write the drawing to."
    )
    draw_parser.add_argument(
        '-s', '--symbol', action='store_true',
        help="Use symbols instead of text."
    )
    draw_parser.add_argument(
        '-c', '--color', action='store_true', help="Color the nodes."
    )
    draw_parser.add_argument(
        '-f', '--format', type=str, default=None,
        help="Override the output format. May be useful if writing to - "
             "(stdout). Should be a valid value for dot's -T argument."
    )
    draw_parser.set_defaults(func=draw)

    args = parser.parse_args(argv)
    if args.quiet and args.verbose:
        print("Error: can't be quiet and verbose at the same time")
        return 1

    if not hasattr(args, 'func'):
        parser.print_help()
        return 1

    return args.func(args)  # type: ignore


def sentence_stats(verbose: bool, sentence: NNF) -> SimpleNamespace:
    stats = SimpleNamespace()
    stats.decomposable = sentence.decomposable()
    stats.cnf = sentence.is_CNF()
    stats.smooth = sentence.smooth()
    stats.vars = len(sentence.vars())
    stats.size = sentence.size()
    stats.clauses = None
    stats.clause_size = None
    if verbose:
        if stats.cnf:
            print("Sentence is in CNF.")
        if stats.decomposable:
            print("Sentence is decomposable.")
        if stats.smooth:
            print("Sentence is smooth.")
        print("Variables:   {}".format(stats.vars))
        print("Size:        {}".format(stats.size))
        if stats.cnf:
            stats.clauses = len(sentence)  # type: ignore
            print("Clauses:     {}".format(stats.clauses))
            sizes = {len(clause) for clause in sentence}  # type: ignore
            stats.clause_size = min(sizes), max(sizes)
            if stats.clause_size[0] == stats.clause_size[1]:
                print("Clause size: {}".format(stats.clause_size[0]))
            else:
                print("Clause size: {}-{}".format(*stats.clause_size))
    return stats


def sat(args: argparse.Namespace) -> int:
    with open_read(args.file) as f:
        sentence = dimacs.load(f)
    stats = sentence_stats(args.verbose, sentence)
    with timer(args):
        sat = sentence.satisfiable(decomposable=stats.decomposable,
                                   cnf=stats.cnf)
    if sat:
        if not args.quiet:
            print("SATISFIABLE")
        return 0
    else:
        if not args.quiet:
            print("UNSATISFIABLE")
        return 1


def sharpsat(args: argparse.Namespace) -> int:
    with open_read(args.file) as f:
        sentence = dimacs.load(f)
    stats = sentence_stats(args.verbose, sentence)
    with timer(args):
        num = sentence.model_count(decomposable=stats.decomposable,
                                   deterministic=args.deterministic,
                                   smooth=stats.smooth)
    if args.quiet:
        print(num)
    else:
        print("{} solutions found.".format(num))
    if num == 0:
        return 1
    return 0


def info(args: argparse.Namespace) -> int:
    with open_read(args.file) as f:
        sentence = dimacs.load(f)
    sentence_stats(True, sentence)
    return 0


dot_formats = {'ps', 'pdf', 'svg', 'fig', 'png', 'gif', 'jpg', 'jpeg'}


def extension(fname: str) -> t.Optional[str]:
    if '.' not in fname:
        return None
    return fname.rsplit('.', 1)[-1].casefold()


def draw(args: argparse.Namespace) -> int:
    with open_read(args.file) as f:
        sentence = dimacs.load(f)
    label = 'symbol' if args.symbol else 'text'
    dot = sentence.to_DOT(color=args.color, label=label)

    ext = extension(args.out)
    if ext in dot_formats or args.format is not None:
        argv = ['dot', '-T' + (ext if args.format is None  # type: ignore
                               else args.format)]
        if args.out != '-':
            argv.append('-o' + args.out)
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                    universal_newlines=True)
        except FileNotFoundError:
            print("Can't find `dot` executable. Is it installed and in your "
                  "PATH?")
            return 1
        proc.stdin.write(dot)
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            print("dot failed with status code {}".format(ret))
        return ret

    else:
        with open_write(args.out) as f:
            f.write(dot)
        return 0
