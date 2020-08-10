"""Utility definitions for internal use. Not part of the public API."""

import functools
import typing as t
import weakref

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


def weakref_memoize(
    func: t.Callable[[T_NNF], T]
) -> "_WeakrefMemoized[T_NNF, T]":
    """Make a function cache its return values using weakrefs.

    This makes it possible to remember sentences' properties without keeping
    them in memory forever.

    To keep memory use reasonable, this decorator should only be used on
    methods that will only be called on full sentences, not individual nodes
    within a sentence.

    The current implementation has a problem: WeakKeyDictionary operates on
    object equality, not identity. Consider the following:

        >>> d = weakref.WeakKeyDictionary()
        >>> s = nnf.And()
        >>> t = nnf.And()
        >>> d[s] = 3
        >>> t in d
        True
        >>> del s
        >>> t in d
        False

    When ``s`` is garbage collected, ``t`` can no longer be looked up in the
    dictionary, even though we did look it up before.

    This might be acceptable because the methods this wraps around aren't
    prohibitively expensive.

    For a solution that can't easily be applied here, see the implementation of
    :meth:`nnf.NNF.mark_deterministic`.
    """
    memo = (
        weakref.WeakKeyDictionary()
    )  # type: weakref.WeakKeyDictionary[T_NNF, T]

    @functools.wraps(func)
    def wrapped(self: T_NNF) -> T:
        try:
            return memo[self]
        except KeyError:
            ret = func(self)
            memo[self] = ret
            return ret

    wrapped.memo = memo  # type: ignore
    wrapped.set = memo.__setitem__  # type: ignore
    return t.cast(_WeakrefMemoized[T_NNF, T], wrapped)


class _WeakrefMemoized(t.Generic[T_NNF, T]):
    """Fake class for typechecking. Should never be instantiated."""
    def __init__(self) -> None:
        assert t.TYPE_CHECKING, "Not a real class"
        self.memo = NotImplemented  # type: weakref.WeakKeyDictionary[T_NNF, T]
        self.__wrapped__ = NotImplemented  # type: t.Callable[[T_NNF], T]

    @t.overload
    def __get__(self: U, instance: None, owner: t.Type[T_NNF]) -> "U":
        ...

    @t.overload  # noqa: F811
    def __get__(  # noqa: F811
        self, instance: T_NNF, owner: t.Optional[t.Type[T_NNF]] = None
    ) -> t.Callable[[], T]:
        ...

    def __get__(  # noqa: F811
        self, instance: object, owner: object = None
    ) -> t.Any:
        ...

    def __call__(self, sentence: T_NNF) -> T:
        ...

    def set(self, sentence: T_NNF, value: T) -> None:
        ...
