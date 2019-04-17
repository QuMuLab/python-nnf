# Copyright 2018 Jan Verbeek <jan.verbeek@posteo.nl>
#
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from __future__ import annotations

import itertools
import typing as t

from dataclasses import dataclass

Name = t.Hashable
Model = t.Dict[Name, bool]

# TODO:
#   - Make Internal inherit directly from frozenset?
#   - Stop using dataclasses?
#   - Compatibility with earlier Python versions?
#   - __slots__ (blocked by dataclass default values)
#   - A way to deduplicate objects in sentences


def all_models(names: t.Collection[Name]) -> t.Iterator[Model]:
    """Yield dictionaries with all possible boolean values for the names.

    >>> list(all_models(["a", "b"]))
    [{'a': True, 'b': True}, {'a': False, 'b': True}, ...
    """
    if not names:
        yield {}
    else:
        name, *rest = names
        for model in all_models(rest):
            yield {name: True, **model}
            yield {name: False, **model}


class NNF:
    """Base class for all NNF sentences."""

    def __and__(self, other: NNF) -> NNF:
        """And({self, other})"""
        return And({self, other})

    def __or__(self, other: NNF) -> NNF:
        """Or({self, other})"""
        return Or({self, other})

    def walk(self) -> t.Iterator[NNF]:
        """Yield all nodes in the sentence, depth-first."""
        yield self

    def size(self) -> int:
        """The number of edges in the sentence."""
        return 0

    def height(self) -> int:
        """The number of edges between here and the furthest leaf."""
        return 0

    def leaf(self) -> bool:
        """True if the node doesn't have children."""
        return True

    def flat(self) -> bool:
        """A sentence is flat if its height is at most 2.

        That is, there are at most two layers below the root node.
        """
        return self.height() <= 2

    def simply_disjunct(self) -> bool:
        """The children of Or nodes are leaves that don't share variables."""
        return all(node.is_simple()
                   for node in self.walk()
                   if isinstance(node, Or))

    def simply_conjunct(self) -> bool:
        """The children of And nodes are leaves that don't share variables."""
        return all(node.is_simple()
                   for node in self.walk()
                   if isinstance(node, And))

    def vars(self) -> t.FrozenSet[Name]:
        """The names of all variables that appear in the sentence."""
        return frozenset()

    def decomposable(self) -> bool:
        """The children of each And node don't share variables, recursively."""
        for node in self.walk():
            if isinstance(node, And):
                seen: t.Set[Name] = set()
                for child in node.children:
                    for name in child.vars():
                        if name in seen:
                            return False
                        seen.add(name)
        return True

    def deterministic(self) -> bool:
        """The children of each Or node contradict each other.

        Warning: expensive!
        """
        for node in self.walk():
            if isinstance(node, Or):
                for a, b in itertools.combinations(node.children, 2):
                    if not a.contradicts(b):
                        return False
        return True

    def smooth(self) -> bool:
        """The children of each Or node all use the same variables."""
        for node in self.walk():
            if isinstance(node, Or):
                expected = None
                for child in node.children:
                    if expected is None:
                        expected = child.vars()
                    else:
                        if child.vars() != expected:
                            return False
        return True

    def decision_node(self) -> bool:
        """The sentence is a valid binary decision diagram (BDD)."""
        return False

    def satisfied_by(self, model: Model) -> bool:
        """The given dictionary of values makes the sentence correct."""
        raise NotImplementedError

    def satisfiable(self) -> bool:
        """Some set of values exists that makes the sentence correct."""
        return any(self.satisfied_by(model)
                   for model in all_models(self.vars()))

    def models(self) -> t.Iterator[Model]:
        """Yield all dictionaries of values that make the sentence correct."""
        for model in all_models(self.vars()):
            if self.satisfied_by(model):
                yield model

    def contradicts(self, other: NNF) -> bool:
        """There is no set of values that satisfies both sentences."""
        for model in self.models():
            if other.satisfied_by(model):
                return False
        return True

    def to_MODS(self) -> NNF:
        """Convert the sentence to a MODS sentence."""
        return Or(And(Var(name, val)
                      for name, val in model.items())
                  for model in self.models())

    def instantiate(self, model: Model) -> NNF:
        """Fill in all the values in the dictionary."""
        return self

    def simplify(self) -> NNF:
        """Apply the following transformations to make the sentence simpler:

        - If an And node has `false` as a child, replace it by `false`
        - If an Or node has `true` as a child, replace it by `true`
        - Remove children of And nodes that are `true`
        - Remove children of Or nodes that are `false`
        - If an And or Or node only has one child, replace it by that child
        """
        # TODO: which properties does this preserve?
        return self


@dataclass(frozen=True)
class Var(NNF):
    """A variable, or its negation.

    If its name is a string, its repr will use that name directly.
    Otherwise it will use more ordinary constructor syntax.

    >>> a = Var('a')
    >>> a
    a
    >>> ~a
    ~a
    >>> b = Var('b')
    >>> a | ~b == Or({Var('a', True), Var('b', False)})
    True
    >>> Var(10)
    Var(10)
    >>> Var(('a', 'b'), False)
    ~Var(('a', 'b'))
    """

    name: Name
    true: bool = True

    def __repr__(self) -> str:
        if isinstance(self.name, str):
            return f"{self.name}" if self.true else f"~{self.name}"
        else:
            base = f"{self.__class__.__name__}({self.name!r})"
            return base if self.true else f"~{base}"

    def __invert__(self) -> Var:
        return Var(self.name, not self.true)

    def vars(self) -> t.FrozenSet[Name]:
        return frozenset({self.name})

    def satisfied_by(self, model: Model) -> bool:
        return model[self.name] if self.true else not model[self.name]

    def instantiate(self, model: Model) -> NNF:
        if self.name in model:
            if self.true == model[self.name]:
                return true
            else:
                return false
        else:
            return self


@dataclass(frozen=True, init=False)
class Internal(NNF):
    """Base class for internal nodes, i.e. And and Or nodes."""

    # add __len__, __iter__ etc for .children?
    children: t.FrozenSet[NNF]

    def __init__(self, children: t.Iterable[NNF] = ()) -> None:
        # needed because the dataclass is frozen
        object.__setattr__(self, 'children', frozenset(children))

    def __repr__(self) -> str:
        if self.children:
            return (f"{self.__class__.__name__}"
                    f"({{{', '.join(map(repr, self.children))}}})")
        else:
            return f"{self.__class__.__name__}()"

    def walk(self) -> t.Iterator[NNF]:
        yield self
        for child in self.children:
            yield from child.walk()

    def size(self) -> int:
        return sum(1 + child.size()
                   for child in self.children)

    def height(self) -> int:
        if self.children:
            return 1 + max(child.height()
                           for child in self.children)
        return 0

    def leaf(self) -> bool:
        if self.children:
            return False
        return True

    def is_simple(self) -> bool:
        """Whether all children are leaves that don't share variables."""
        variables: t.Set[Name] = set()
        for child in self.children:
            if not child.leaf():
                return False
            if isinstance(child, Var):
                if child.name in variables:
                    return False
                variables.add(child.name)
        return True

    def vars(self) -> t.FrozenSet[Name]:
        return frozenset(name
                         for child in self.children
                         for name in child.vars())

    def instantiate(self, model: Model) -> NNF:
        return self.__class__(child.instantiate(model)
                              for child in self.children)


class And(Internal):
    """Conjunction nodes, which are only true if all of their children are."""
    def satisfied_by(self, model: Model) -> bool:
        return all(child.satisfied_by(model)
                   for child in self.children)

    def simplify(self) -> NNF:
        if false in self.children:
            return false
        new = {child.simplify() for child in self.children} - {true}
        if not new:
            return true
        if false in new:
            return false
        if len(new) == 1:
            return list(new)[0]
        return self.__class__(new)

    def decision_node(self) -> bool:
        if not self.children:
            return True
        return False

    def __repr__(self) -> str:
        if not self.children:
            return 'true'
        return super().__repr__()


class Or(Internal):
    """Disjunction nodes, which are true if any of their children are."""
    def satisfied_by(self, model: Model) -> bool:
        return any(child.satisfied_by(model)
                   for child in self.children)

    def decision_node(self) -> bool:
        if not self.children:
            return True  # boolean
        if len(self.children) != 2:
            return False
        child1, child2 = self.children
        if not (isinstance(child1, And) and isinstance(child2, And)):
            return False
        if not (len(child1.children) == 2 and len(child2.children) == 2):
            return False

        child11, child12 = child1.children
        child21, child22 = child2.children

        if isinstance(child11, Var):
            child1var = child11
            if not child12.decision_node():
                return False
        elif isinstance(child12, Var):
            child1var = child12
            if not child11.decision_node():
                return False
        else:
            return False

        if isinstance(child21, Var):
            child2var = child21
            if not child22.decision_node():
                return False
        elif isinstance(child22, Var):
            child2var = child22
            if not child21.decision_node():
                return False
        else:
            return False

        if child1var.name != child2var.name:
            return False

        if child1var.true == child2var.true:
            return False

        return True

    def simplify(self) -> NNF:
        if true in self.children:
            return true
        new = {child.simplify() for child in self.children} - {false}
        if not new:
            return false
        if true in new:
            return true
        if len(new) == 1:
            return list(new)[0]
        return self.__class__(new)

    def __repr__(self) -> str:
        if not self.children:
            return 'false'
        return super().__repr__()


def decision(var: Var, if_true: NNF, if_false: NNF) -> Or:
    """Create a decision node with a variable and two branches."""
    return Or({And({var, if_true}), And({~var, if_false})})


true = And()
false = Or()
