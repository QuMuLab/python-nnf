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
#   - __slots__


def all_models(names: t.Iterable[Name]) -> t.Iterator[Model]:
    if not names:
        yield {}
    else:
        name, *rest = names
        for model in all_models(rest):
            yield {name: True, **model}
            yield {name: False, **model}


class NNF:
    def __and__(self, other: NNF) -> NNF:
        return And({self, other})

    def __or__(self, other: NNF) -> NNF:
        return Or({self, other})

    def walk(self) -> t.Iterator[NNF]:
        yield self

    def size(self) -> int:
        raise NotImplementedError

    def height(self) -> int:
        raise NotImplementedError

    def flat(self) -> bool:
        return self.size() <= 2

    def simply_disjunct(self) -> bool:
        return all(node.is_simple()
                   for node in self.walk()
                   if isinstance(node, Or))

    def simply_conjunct(self) -> bool:
        return all(node.is_simple()
                   for node in self.walk()
                   if isinstance(node, And))

    def vars(self) -> t.FrozenSet[Name]:
        return frozenset()

    def decomposable(self) -> bool:
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
        for node in self.walk():
            if isinstance(node, Or):
                for a, b in itertools.combinations(node.children, 2):
                    if not a.contradicts(b):
                        return False
        return True

    def smooth(self) -> bool:
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
        return False

    def satisfied_by(self, model: Model) -> bool:
        raise NotImplementedError

    def satisfiable(self) -> bool:
        return any(self.satisfied_by(model)
                   for model in all_models(self.vars()))

    def models(self) -> t.Iterator[Model]:
        for model in all_models(self.vars()):
            if self.satisfied_by(model):
                yield model

    def contradicts(self, other: NNF) -> bool:
        for model in self.models():
            if other.satisfied_by(model):
                return False
        return True

    def to_MODS(self) -> NNF:
        return Or(And(Var(name, val)
                      for name, val in model.items())
                  for model in self.models())

    def instantiate(self, model: Model) -> NNF:
        return self

    def simplify(self) -> NNF:
        # TODO: which properties does this preserve?
        return self


class Leaf(NNF):
    def size(self) -> int:
        return 0

    def height(self) -> int:
        return 0


@dataclass(frozen=True)
class Var(Leaf):
    name: Name
    true: bool = True

    def __repr__(self) -> str:
        return f"{self.name}" if self.true else f"~{self.name}"

    def __invert__(self) -> Leaf:
        return Var(self.name, not self.true)

    def vars(self) -> t.FrozenSet[Name]:
        return frozenset({self.name})

    def satisfied_by(self, model: Model) -> bool:
        return model[self.name] if self.true else not model[self.name]

    def instantiate(self, model: Model) -> Leaf:
        if self.name in model:
            if self.true == model[self.name]:
                return true
            else:
                return false
        else:
            return self


@dataclass(frozen=True)
class Bool(Leaf):
    true: bool

    def __repr__(self) -> str:
        return 'true' if self.true else 'false'

    def __invert__(self) -> Bool:
        return Bool(not self.true)

    def satisfied_by(self, model: Model) -> bool:
        return self.true

    def decision_node(self) -> bool:
        return True


true = Bool(True)
false = Bool(False)


@dataclass(frozen=True, init=False)
class Internal(NNF):
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
        else:
            return 0

    def is_simple(self) -> bool:
        variables: t.Set[Name] = set()
        for child in self.children:
            if not isinstance(child, Leaf):
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
    def satisfied_by(self, model: Model) -> bool:
        return all(child.satisfied_by(model)
                   for child in self.children)

    def simplify(self) -> NNF:
        new = {child.simplify() for child in self.children} - {true}
        if not new:
            return true
        if false in new:
            return false
        return self.__class__(new)


class Or(Internal):
    def satisfied_by(self, model: Model) -> bool:
        return any(child.satisfied_by(model)
                   for child in self.children)

    def decision_node(self) -> bool:
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
        new = {child.simplify() for child in self.children} - {false}
        if not new:
            return false
        if true in new:
            return true
        return self.__class__(new)


def decision(var: Var, if_true: NNF, if_false: NNF) -> Or:
    return Or({And({var, if_true}), And({~var, if_false})})
