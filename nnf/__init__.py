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

import functools
import itertools
import typing as t

from dataclasses import dataclass

import nnf

Name = t.Hashable
Model = t.Dict[Name, bool]

memoize = functools.lru_cache(maxsize=None)

# TODO:
#   - Make Internal inherit directly from frozenset?
#     - No, code becomes less readable and equality and operators go bad
#   - Stop using dataclasses?
#   - Compatibility with earlier Python versions?
#   - __slots__ (blocked by dataclass default values)
#   - Generic types for NNF and Internal?
#   - isinstance(true, Internal) is currently true, look for weird effects
#   - add __all__
#   - try using memoize in more places
#   - to_model(self) -> Model


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


T = t.TypeVar('T')


class NNF:
    """Base class for all NNF sentences."""

    def __and__(self, other: NNF) -> NNF:
        """And({self, other})"""
        return And({self, other})

    def __or__(self, other: NNF) -> NNF:
        """Or({self, other})"""
        return Or({self, other})

    def walk(self) -> t.Iterator[NNF]:
        """Yield all nodes in the sentence, depth-first.

        Nodes that appear multiple times are yielded only once.
        """
        # Could be made width-first by using a deque and popping from the left
        seen = {self}
        nodes = [self]
        while nodes:
            node = nodes.pop()
            yield node
            if isinstance(node, Internal):
                for child in node.children:
                    if child not in seen:
                        seen.add(child)
                        nodes.append(child)

    def size(self) -> int:
        """The number of edges in the sentence."""
        def edge_count(transform: t.Callable[[NNF], int], node: NNF) -> int:
            if isinstance(node, Internal):
                return len(node.children) + sum(transform(child)
                                                for child in node.children)
            return 0

        return self.transform(edge_count)

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
        # Could be sped up by returning as soon as a path longer than 2 is
        # found, instead of computing the full height
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
        return frozenset(node.name
                         for node in self.walk()
                         if isinstance(node, Var))

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
        # todo: if decomposable, use less expensive check
        return any(self.satisfied_by(model)
                   for model in all_models(self.vars()))

    consistent = satisfiable  # synonym

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

    def to_model(self) -> Model:
        """If the sentence directly represents a model, convert it to that.

        A sentence directly represents a model if it's a conjunction of
        (unique) variables, or a single variable.
        """
        if isinstance(self, Var):
            return {self.name: self.true}
        if not isinstance(self, And):
            raise TypeError("A sentence can only be converted to a model if "
                            "it's a conjunction of variables.")
        model: Model = {}
        for child in self.children:
            if not isinstance(child, Var):
                raise TypeError("A sentence can only be converted to a "
                                "model if it's a conjunction of variables.")
            if child.name in model:
                raise ValueError(f"{child.name!r} appears multiple times.")
            model[child.name] = child.true

        return model

    def condition(self, model: Model) -> NNF:
        """Fill in all the values in the dictionary."""
        return self

    def simplify(self) -> NNF:
        """Apply the following transformations to make the sentence simpler:

        - If an And node has `false` as a child, replace it by `false`
        - If an Or node has `true` as a child, replace it by `true`
        - Remove children of And nodes that are `true`
        - Remove children of Or nodes that are `false`
        - If an And or Or node only has one child, replace it by that child
        - If an And or Or node has a child of the same type, merge them
        """
        # TODO: which properties does this preserve?

        @memoize
        def simple(node: NNF) -> NNF:
            if isinstance(node, Var):
                return node
            new_children: t.Set[NNF] = set()
            if isinstance(node, Or):
                for child in map(simple, node.children):
                    if child == true:
                        return true
                    elif child == false:
                        pass
                    elif isinstance(child, Or):
                        new_children.update(child.children)
                    else:
                        new_children.add(child)
                if len(new_children) == 0:
                    return false
                elif len(new_children) == 1:
                    return list(new_children)[0]
                return Or(new_children)
            elif isinstance(node, And):
                for child in map(simple, node.children):
                    if child == false:
                        return false
                    elif child == true:
                        pass
                    elif isinstance(child, And):
                        new_children.update(child.children)
                    else:
                        new_children.add(child)
                if len(new_children) == 0:
                    return true
                elif len(new_children) == 1:
                    return list(new_children)[0]
                return And(new_children)
            else:
                raise TypeError(node)

        return simple(self)

    def deduplicate(self) -> NNF:
        """Return a copy of the sentence without any duplicate objects.

        If a node has multiple parents, it's possible for it to be
        represented by two separate objects. This method gets rid of that
        duplication.

        In a lot of cases it's better to avoid the duplication in the first
        place, for example with a Builder object.
        """
        new_nodes: t.Dict[NNF, NNF] = {}

        def recreate(node: NNF) -> NNF:
            if node not in new_nodes:
                if isinstance(node, Var):
                    new_nodes[node] = node
                elif isinstance(node, Or):
                    new_nodes[node] = Or(recreate(child)
                                         for child in node.children)
                elif isinstance(node, And):
                    new_nodes[node] = And(recreate(child)
                                          for child in node.children)
            return new_nodes[node]

        return recreate(self)

    def object_count(self) -> int:
        """Return the number of distinct node objects in the sentence."""
        ids: t.Set[int] = set()

        def count(node: NNF) -> None:
            ids.add(id(node))
            if isinstance(node, Internal):
                for child in node.children:
                    if id(child) not in ids:
                        count(child)

        count(self)
        return len(ids)

    def to_DOT(self, color: bool = False) -> str:
        """Output a representation of the sentence in the DOT language.

        DOT is a graph visualization language.
        """
        # TODO: sort in some clever way for deterministic output
        # TODO: offer more knobs
        #       - add own directives
        #       - set different palette
        counter = itertools.count()
        names: t.Dict[NNF, t.Tuple[int, str, str]] = {}
        arrows: t.Set[t.Tuple[int, int]] = set()

        def name(node: NNF) -> int:
            if node not in names:
                number = next(counter)
                if isinstance(node, Var):
                    label = str(node.name).replace('"', r'\"')
                    color = 'chartreuse'
                    if not node.true:
                        label = '¬' + label
                        color = 'pink'
                    names[node] = (number, label, color)
                elif node == true:
                    names[node] = (number, "⊤", 'green')
                elif node == false:
                    names[node] = (number, "⊥", 'red')
                elif isinstance(node, And):
                    names[node] = (number, "∧", 'lightblue')
                elif isinstance(node, Or):
                    names[node] = (number, "∨", 'yellow')
                else:
                    raise TypeError(f"Can't handle node of type {type(node)}")
            return names[node][0]

        for node in self.walk():
            name(node)
            if isinstance(node, Internal):
                for child in node.children:
                    arrows.add((name(node), name(child)))

        return '\n'.join([
            'digraph {',
            *(
                f'    {number} [label="{label}"'
                + (f' fillcolor="{fillcolor}" style=filled]' if color else ']')
                for number, label, fillcolor in names.values()
            ),
            *(
                f'    {src} -> {dst}'
                for src, dst in arrows
            ),
            '}\n'
        ])

    def transform(
            self,
            func: t.Callable[[t.Callable[[NNF], T], NNF], T]
    ) -> T:
        """A helper function to apply a transformation with memoization.

        It should be passed a function that takes as its first argument a
        function that wraps itself, to use for recursive calls.

        For example::

            def vars(transform, node):
                if isinstance(node, Var):
                    return {node.name}
                else:
                    return {name for child in node.children
                           for name in transform(child)}

            names = sentence.transform(vars)
        """
        @memoize
        def transform(node: NNF) -> T:
            return func(transform, node)

        return transform(self)

    def models_smart(self) -> t.Iterator[Model]:
        """An alternative to .models().

        Potentially much faster if there are few models, but potentially
        much slower if there are many models.

        A pathological case is `Or({Var(1), Var(2), Var(3), ...})`.
        """
        ModelInt = t.FrozenSet[t.Tuple[Name, bool]]

        def compatible(a: ModelInt, b: ModelInt) -> bool:
            if len(a) > len(b):
                a, b = b, a
            return not any((name, not value) in b for name, value in a)

        def extract(
                transform: t.Callable[[NNF], t.Iterable[ModelInt]],
                node: NNF
        ) -> t.Set[ModelInt]:
            if isinstance(node, Var):
                return {frozenset(((node.name, node.true),))}
            elif isinstance(node, Or):
                return {model
                        for child in node.children
                        for model in transform(child)}
            elif isinstance(node, And):
                models: t.Set[ModelInt] = {frozenset()}
                for child in node.children:
                    models = {existing | new
                              for new in transform(child)
                              for existing in models
                              if compatible(existing, new)}
                return models
            raise TypeError(node)

        names = self.vars()
        full_models: t.Set[ModelInt] = set()

        def complete(
                model: ModelInt,
                names: t.List[Name]
        ) -> t.Iterator[ModelInt]:
            for expansion in all_models(names):
                yield frozenset(model | expansion.items())

        for model in self.transform(extract):
            missing_names = list(names - {name for name, value in model})
            if not missing_names:
                full_models.add(model)
            else:
                full_models.update(complete(model, missing_names))

        for full_model in full_models:
            yield dict(full_model)


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

    def satisfied_by(self, model: Model) -> bool:
        return model[self.name] if self.true else not model[self.name]

    def condition(self, model: Model) -> NNF:
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

    __slots__ = ()

    def __init__(self, children: t.Iterable[NNF] = ()) -> None:
        # needed because the dataclass is frozen
        object.__setattr__(self, 'children', frozenset(children))

    def __repr__(self) -> str:
        if self.children:
            return (f"{self.__class__.__name__}"
                    f"({{{', '.join(map(repr, self.children))}}})")
        else:
            return f"{self.__class__.__name__}()"

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

    def condition(self, model: Model) -> NNF:
        return self.__class__(child.condition(model)
                              for child in self.children)


class And(Internal):
    """Conjunction nodes, which are only true if all of their children are."""
    def satisfied_by(self, model: Model) -> bool:
        return all(child.satisfied_by(model)
                   for child in self.children)

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

    def __repr__(self) -> str:
        if not self.children:
            return 'false'
        return super().__repr__()


def decision(var: Var, if_true: NNF, if_false: NNF) -> Or:
    """Create a decision node with a variable and two branches."""
    return Or({And({var, if_true}), And({~var, if_false})})


true = And()
false = Or()


class Builder:
    """Automatically deduplicates NNF nodes as you make them.

    Usage:

    >>> builder = Builder()
    >>> var = builder.Var('A')
    >>> var2 = builder.Var('A')
    >>> var is var2
    True
    """
    # TODO: deduplicate vars that are negated using the operator
    def __init__(self, seed: t.Iterable[NNF] = ()):
        self.stored: t.Dict[NNF, NNF] = {true: true, false: false}
        for node in seed:
            self.stored[node] = node
        self.true = true
        self.false = false

    def Var(self, name: Name, true: bool = True) -> nnf.Var:
        ret = Var(name, true)
        return self.stored.setdefault(ret, ret)  # type: ignore

    def And(self, children: t.Iterable[NNF] = ()) -> nnf.And:
        ret = And(children)
        return self.stored.setdefault(ret, ret)  # type: ignore

    def Or(self, children: t.Iterable[NNF] = ()) -> nnf.Or:
        ret = Or(children)
        return self.stored.setdefault(ret, ret)  # type: ignore
