"""Utility definitions for internal use. Not part of the public API."""

import functools
import typing as t

if t.TYPE_CHECKING:
    from nnf import NNF  # noqa: F401

Name = t.Hashable
Model = t.Dict[Name, bool]

T = t.TypeVar("T")
U = t.TypeVar("U")
T_NNF = t.TypeVar("T_NNF", bound="NNF")
U_NNF = t.TypeVar("U_NNF", bound="NNF")
T_NNF_co = t.TypeVar("T_NNF_co", bound="NNF", covariant=True)
_Tristate = t.Optional[bool]

# Bottom type with no values
# This works in mypy but not pytype
# t.Union[()] works too but not at runtime
# NoReturn doesn't exist in some Python releases, hence the guard
if t.TYPE_CHECKING:
    Bottom = t.NoReturn
else:
    Bottom = None

memoize = t.cast(t.Callable[[T], T], functools.lru_cache(maxsize=None))
