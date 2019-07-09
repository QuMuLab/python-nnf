"""Convenience functions for logical relationships that are not part of NNF.

These functions will simulate those relationships, often by doubling
sentences or altering their structure to negate them. This makes them
inefficient.
"""

import typing as t

from nnf import NNF, And, Or, T_NNF, U_NNF

__all__ = ('xor', 'nand', 'nor', 'implies', 'implied_by', 'iff', 'and_', 'or_')


def xor(a: NNF, b: NNF) -> Or[And[NNF]]:
    """Exactly one of the operands is true."""
    return (a.negate() & b) | (a & b.negate())


def nand(a: NNF, b: NNF) -> Or[NNF]:
    """At least one of the operands is false."""
    return a.negate() | b.negate()


def nor(a: NNF, b: NNF) -> And[NNF]:
    """Both of the operands are false."""
    return a.negate() & b.negate()


def implies(a: NNF, b: NNF) -> Or[NNF]:
    """``b`` is true whenever ``a`` is true."""
    return a.negate() | b


def implied_by(a: NNF, b: NNF) -> Or[NNF]:
    """``a`` is true whenever ``b`` is true."""
    return a | b.negate()


def iff(a: NNF, b: NNF) -> Or[And[NNF]]:
    """``a`` is true if and only if ``b`` is true."""
    return (a & b) | (a.negate() & b.negate())


def and_(a: T_NNF, b: U_NNF) -> And[t.Union[T_NNF, U_NNF]]:
    """``a`` and ``b`` are both true. Included for completeness."""
    return a & b


def or_(a: T_NNF, b: U_NNF) -> Or[t.Union[T_NNF, U_NNF]]:
    """``a`` or ``b`` is true. Included for completeness."""
    return a | b
