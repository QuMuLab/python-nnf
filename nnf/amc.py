import functools
import math
import operator

import typing as t

from nnf import NNF, And, Var, Or, Internal, Name, true, false

T = t.TypeVar('T')
memoize = functools.lru_cache(maxsize=None)


def eval(
        node: NNF,
        add: t.Callable[[T, T], T],
        mul: t.Callable[[T, T], T],
        add_neut: T,
        mul_neut: T,
        labeling: t.Callable[[Var], T],
) -> T:
    """Execute an AMC technique, given a semiring and a labeling function."""
    @memoize
    def do_eval(node: NNF) -> T:
        if node == true:
            return mul_neut
        elif node == false:
            return add_neut
        elif isinstance(node, Var):
            return labeling(node)
        assert isinstance(node, Internal)
        if len(node.children) == 1:
            return do_eval(*node.children)
        if isinstance(node, Or):
            return functools.reduce(
                add,
                (do_eval(child) for child in node.children),
                add_neut
            )
        assert isinstance(node, And)
        return functools.reduce(
            mul,
            (do_eval(child) for child in node.children),
            mul_neut
        )

    return do_eval(node)


def _prob_label(probs: t.Dict[Name, float]) -> t.Callable[[Var], float]:
    """Generate a labeling function for probabilities from a dictionary."""
    def label(leaf: Var) -> float:
        if leaf.true:
            return probs[leaf.name]
        else:
            return 1.0 - probs[leaf.name]

    return label


def SAT(node: NNF) -> bool:
    """Determine whether a DNNF sentence is satisfiable."""
    return eval(node, operator.or_, operator.and_, False, True,
                lambda leaf: True)


def NUM_SAT(node: NNF) -> int:
    """Determine the number of models that satisfy a sd-DNNF sentence."""
    # General ×
    # Non-idempotent +
    # Non-neutral +
    # = sd-DNNF
    return eval(node, operator.add, operator.mul, 0, 1, lambda leaf: 1)


def WMC(node: NNF, weights: t.Callable[[Var], float]) -> float:
    # General ×
    # Non-idempotent +
    # Non-neutral +
    # = sd-DNNF
    return eval(node, operator.add, operator.mul, 0.0, 1.0, weights)


def PROB(node: NNF, probs: t.Dict[Name, float]) -> float:
    # General ×
    # Non-idempotent +
    # Neutral +
    # = d-DNNF
    return eval(node, operator.add, operator.mul, 0.0, 1.0, _prob_label(probs))


GradProb = t.Tuple[float, float]


def GRAD(
        node: NNF,
        probs: t.Dict[Name, float],
        k: t.Optional[Name] = None
) -> GradProb:
    # General ×
    # Neutral +
    # Non-idempotent +
    # = d-DNNF

    def add(a: GradProb, b: GradProb) -> GradProb:
        return a[0] + b[0], a[1] + b[1]

    def mul(a: GradProb, b: GradProb) -> GradProb:
        return a[0] * b[0], a[0] * b[1] + a[1] * b[0]

    def label(var: Var) -> GradProb:
        if var.true:
            if var.name == k:
                return probs[var.name], 1
            else:
                return probs[var.name], 0
        else:
            if var.name == k:
                return 1 - probs[var.name], -1
            else:
                return 1 - probs[var.name], 0

    return eval(node, add, mul, (0.0, 0.0), (1.0, 0.0), label)


def MPE(node: NNF, probs: t.Dict[Name, float]) -> float:
    # General ×
    # Non-neutral +
    # Idempotent +
    # = s-DNNF
    return eval(node, max, operator.mul, 0.0, 1.0, _prob_label(probs))


def reduce(
        node: NNF,
        add_key: t.Callable[[T], t.Any],
        mul: t.Callable[[T, T], T],
        add_neut: T,
        mul_neut: T,
        labeling: t.Callable[[Var], T],
) -> NNF:
    def add(a: T, b: T) -> T:
        return max((a, b), key=add_key)

    if isinstance(node, Or):
        best = add_neut
        candidates: t.List[NNF] = []
        for child in node.children:
            value = eval(child, add, mul, add_neut, mul_neut, labeling)
            if value > best:  # type: ignore
                best = value
                candidates = [child]
            elif value == best:
                candidates.append(child)
        return Or(reduce(candidate, add_key, mul, add_neut, mul_neut, labeling)
                  for candidate in candidates)
    elif isinstance(node, And):
        return And(reduce(child, add_key, mul, add_neut, mul_neut, labeling)
                   for child in node.children)
    else:
        return node


def maxplus_reduce(node: NNF, labels: t.Dict[Var, float]) -> NNF:
    def labeling(v: Var) -> float:
        return labels[v]
    return reduce(node, lambda n: n, operator.add, -math.inf, 0, labeling)
