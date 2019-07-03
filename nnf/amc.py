"""An implementation of
`algebraic model counting <https://arxiv.org/abs/1211.4475>`_."""

import functools
import operator

import typing as t

from nnf import NNF, And, Var, Or, Internal, Name, true, false

neg_inf = float('-inf')

T = t.TypeVar('T')
memoize = functools.lru_cache(maxsize=None)

__all__ = ('eval', 'reduce', 'SAT', 'NUM_SAT', 'WMC', 'PROB', 'GRAD', 'MPE',
           'maxplus_reduce')


def eval(
        node: NNF,
        add: t.Callable[[T, T], T],
        mul: t.Callable[[T, T], T],
        add_neut: T,
        mul_neut: T,
        labeling: t.Callable[[Var], T],
) -> T:
    """Execute an AMC technique, given a semiring and a labeling function.

    :param node: The sentence to calculate the value of.
    :param add: The ⊕ operator, to combine :class:`nnf.Or` nodes.
    :param mul: The ⊗ operator, to combine :class:`nnf.And` nodes.
    :param add_neut: e^⊕, the neutral element of the ⊕ operator.
    :param mul_neut: e^⊗, the neutral element of the ⊗ operator.
    :param labeling: The labeling function, to assign a value to each
                     variable node.
    """
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
    """Model counting of sd-DNNF sentences, weighted by variables.

    :param node: The sentence to measure.
    :param weights: A dictionary mapping variable nodes to weights.
    """
    # General ×
    # Non-idempotent +
    # Non-neutral +
    # = sd-DNNF
    return eval(node, operator.add, operator.mul, 0.0, 1.0, weights)


def PROB(node: NNF, probs: t.Dict[Name, float]) -> float:
    """Model counting of d-DNNF sentences, weighted by probabilities.

    :param node: The sentence to measure.
    :param probs: A dictionary mapping variable names to probabilities.
    """
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
    """Calculate a gradient of a d-DNNF sentence being true depending on the
    value of a variable, given probabilities for all variables.

    :param node: The sentence.
    :param probs: A dictionary mapping variable names to probabilities.
    :param k: The name of the variable to check relative to.

    :return: A tuple of two floats (probability, gradient).
    """
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
        add_key: t.Optional[t.Callable[[T], t.Any]],
        mul: t.Callable[[T, T], T],
        add_neut: T,
        mul_neut: T,
        labeling: t.Callable[[Var], T],
) -> NNF:
    """Execute AMC reduction on a sentence.

    In AMC reduction, the ⊕ operator must be ``max`` on some total order,
    and the branches of the sentence that don't contribute to the maximum
    value are removed. This leaves a simpler sentence with only the models
    with a maximum value.

    :param node: The sentence.
    :param add_key: A function given to ``max``'s ``key`` argument to
                    determine the total order of the ⊕ operator. Pass
                    ``None`` to use the default ordering.
    :param mul: See :func:`eval`.
    :param add_neut: See :func:`eval`.
    :param mul_neut: See :func:`eval`.
    :param labeling: See :func:`eval`.

    :return: The transformed sentence.
    """
    if add_key is not None:
        add_key_ = add_key
    else:
        def add_key_(n: T) -> t.Any:
            return n

    def add(a: T, b: T) -> T:
        return max((a, b), key=add_key_)

    @memoize
    def eval_(node: NNF) -> T:
        return eval(node, add, mul, add_neut, mul_neut, labeling)

    @memoize
    def reduce_(node: NNF) -> NNF:
        if isinstance(node, Or):
            best = add_neut
            candidates = []  # type: t.List[NNF]
            for child in node.children:
                value = eval_(child)
                if value > best:  # type: ignore
                    best = value
                    candidates = [child]
                elif value == best:
                    candidates.append(child)
            return Or(reduce_(candidate) for candidate in candidates)
        elif isinstance(node, And):
            return And(reduce_(child) for child in node.children)
        else:
            return node

    return reduce_(node)


def maxplus_reduce(node: NNF, labels: t.Dict[Var, float]) -> NNF:
    """Execute AMC reduction using the maxplus algebra.

    :param node: The sentence.
    :param labels: A dictionary mapping variable nodes to numbers.
    """
    def labeling(v: Var) -> float:
        return labels[v]
    return reduce(node, None, operator.add, neg_inf, 0, labeling)
