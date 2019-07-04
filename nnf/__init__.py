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

__version__ = '0.1.2'

import abc
import functools
import itertools
import typing as t

if t.TYPE_CHECKING:
    import nnf

Name = t.Hashable
Model = t.Dict[Name, bool]

memoize = functools.lru_cache(maxsize=None)

__all__ = ('NNF', 'Internal', 'And', 'Or', 'Var', 'Builder', 'all_models',
           'decision', 'true', 'false', 'dsharp', 'dimacs', 'amc')


def all_models(names: 't.Iterable[Name]') -> t.Iterator[Model]:
    """Yield dictionaries with all possible boolean values for the names.

    >>> list(all_models(["a", "b"]))
    [{'a': False, 'b': False}, {'a': False, 'b': True}, ...
    """
    if not names:
        yield {}
    else:
        *rest, name = names
        for model in all_models(rest):
            new = model.copy()
            new[name] = False
            yield new
            new = model.copy()
            new[name] = True
            yield new


T = t.TypeVar('T')
_Tristate = t.Optional[bool]


class NNF(metaclass=abc.ABCMeta):
    """Base class for all NNF sentences."""
    __slots__ = ()

    def __and__(self, other: 'NNF') -> 'NNF':
        """And({self, other})"""
        return And({self, other})

    def __or__(self, other: 'NNF') -> 'NNF':
        """Or({self, other})"""
        return Or({self, other})

    def walk(self) -> t.Iterator['NNF']:
        """Yield all nodes in the sentence, depth-first.

        Nodes with multiple parents are yielded only once.
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
        """The number of edges in the sentence.

        Note that sentences are rooted DAGs, not trees. If a node has
        multiple parents its edges will still be counted just once.
        """
        return sum(len(node.children)
                   for node in self.walk()
                   if isinstance(node, Internal))

    def height(self) -> int:
        """The number of edges between here and the furthest leaf."""
        @memoize
        def height(node: NNF) -> int:
            if isinstance(node, Internal) and node.children:
                return 1 + max(height(child) for child in node.children)
            return 0

        return height(self)

    def leaf(self) -> bool:
        """True if the node doesn't have children.

        That is, if the node is a variable, or one of ``true`` and ``false``.
        """
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
        return all(node._is_simple()
                   for node in self.walk()
                   if isinstance(node, Or))

    def simply_conjunct(self) -> bool:
        """The children of And nodes are leaves that don't share variables."""
        return all(node._is_simple()
                   for node in self.walk()
                   if isinstance(node, And))

    def vars(self) -> t.FrozenSet[Name]:
        """The names of all variables that appear in the sentence."""
        return frozenset(node.name
                         for node in self.walk()
                         if isinstance(node, Var))

    def decomposable(self) -> bool:
        """The children of each And node don't share variables, recursively."""
        @memoize
        def var(node: NNF) -> t.FrozenSet[Name]:
            return node.vars()

        for node in self.walk():
            if isinstance(node, And):
                seen = set()  # type: t.Set[Name]
                for child in node.children:
                    for name in var(child):
                        if name in seen:
                            return False
                        seen.add(name)
        return True

    def deterministic(self) -> bool:
        """The children of each Or node contradict each other.

        May be very expensive.
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
            if isinstance(node, Or) and len(node.children) > 1:
                expected = None
                for child in node.children:
                    if expected is None:
                        expected = child.vars()
                    else:
                        if child.vars() != expected:
                            return False
        return True

    @abc.abstractmethod
    def decision_node(self) -> bool:
        """The sentence is a valid binary decision diagram (BDD)."""
        ...

    def satisfied_by(self, model: Model) -> bool:
        """The given dictionary of values makes the sentence correct."""
        @memoize
        def sat(node: NNF) -> bool:
            if isinstance(node, Var):
                if node.name not in model:
                    # Note: because any and all are lazy, it's possible for
                    # this error not to occur even if a variable is missing.
                    # In such a case including the variable with any value
                    # would not affect the return value though.
                    raise ValueError("Model does not contain variable {!r}"
                                     .format(node.name))
                return model[node.name] == node.true
            elif isinstance(node, Or):
                return any(sat(child) for child in node.children)
            elif isinstance(node, And):
                return all(sat(child) for child in node.children)
            else:
                raise TypeError(node)

        return sat(self)

    def satisfiable(self, decomposable: _Tristate = None) -> bool:
        """Some set of values exists that makes the sentence correct."""
        if not self._satisfiable_decomposable():
            return False

        if decomposable is None:
            decomposable = self.decomposable()

        if decomposable:
            # Would've been picked up already if not satisfiable
            return True

        return any(self.satisfied_by(model)
                   for model in all_models(self.vars()))

    def _satisfiable_decomposable(self) -> bool:
        """Checks satisfiability of decomposable sentences.

        If the sentence is not decomposable, it may return True even if the
        sentence is not satisfiable. But if it returns False the sentence is
        certainly not satisfiable.
        """
        @memoize
        def sat(node: NNF) -> bool:
            """Check satisfiability of DNNF."""
            if isinstance(node, Or):
                # note: if node == false this path is followed
                return any(sat(child) for child in node.children)
            elif isinstance(node, And):
                return all(sat(child) for child in node.children)
            return True

        return sat(self)

    def _consistent_with_model(self, model: Model) -> bool:
        """A combination of `condition` and `satisfiable`.

        Only works on decomposable sentences, but doesn't check for the
        property. Use with care.
        """
        @memoize
        def con(node: NNF) -> bool:
            if isinstance(node, Var):
                if node.name not in model:
                    return True
                if model[node.name] == node.true:
                    return True
                return False
            elif isinstance(node, Or):
                return any(con(child) for child in node.children)
            elif isinstance(node, And):
                return all(con(child) for child in node.children)
            else:
                raise TypeError(node)

        return con(self)

    consistent = satisfiable  # synonym

    def models(
            self,
            decomposable: _Tristate = None,
            deterministic: bool = False
    ) -> t.Iterator[Model]:
        """Yield all dictionaries of values that make the sentence correct.

        Much faster on sentences that are deterministic or decomposable or
        both.

        The algorithm for deterministic sentences works on non-deterministic
        sentences, but may be much slower for such sentences. Using
        ``deterministic=True`` for sentences that aren't deterministic can
        be a reasonable decision.

        :param decomposable: Whether to assume the sentence is
                             decomposable. If ``None`` (the default),
                             the sentence is automatically checked.
        :param deterministic: Indicate whether the sentence is
                              deterministic. Set this to ``True`` if you
                              know it to be deterministic, or want it to be
                              treated as deterministic.
        """
        if decomposable is None:
            decomposable = self.decomposable()
        if deterministic:
            yield from self._models_deterministic(decomposable=decomposable)
        elif decomposable:
            yield from self._models_decomposable()
        else:
            for model in all_models(self.vars()):
                if self.satisfied_by(model):
                    yield model

    def contradicts(
            self,
            other: 'NNF',
            decomposable: _Tristate = None
    ) -> bool:
        """There is no set of values that satisfies both sentences.

        May be very expensive.
        """
        if decomposable is None:
            decomposable = self.decomposable() and other.decomposable()

        if len(self.vars()) > len(other.vars()):
            # The one with the fewest vars has the smallest models
            a, b = other, self
        else:
            a, b = self, other

        if decomposable:
            for model in a.models(decomposable=True):
                if b._consistent_with_model(model):
                    return False
            return True

        for model in b.models():
            # Hopefully, a.vars() <= b.vars() and .satisfiable() is fast
            if a.condition(model).satisfiable():
                return False
        return True

    def equivalent(self, other: 'NNF') -> bool:
        """Test whether two sentences have the same models.

        If two sentences don't mention the same variables they're not
        considered equivalent.

        Warning: may be very slow. Decomposability helps.
        """
        if self == other:
            return True
        vars_a = self.vars()
        vars_b = other.vars()
        if vars_a != vars_b:
            return False
        models_a = list(self.models())
        models_b = list(other.models())
        if len(models_a) != len(models_b):
            return False
        if models_a == models_b:
            return True

        def dict_hashable(
                model: t.Dict[Name, bool]
        ) -> t.FrozenSet[t.Tuple[Name, bool]]:
            return frozenset((name, model[name]) for name in vars_a)

        return (set(map(dict_hashable, models_a)) ==
                set(map(dict_hashable, models_b)))

    def to_MODS(self) -> 'NNF':
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
        model = {}  # type: Model
        for child in self.children:
            if not isinstance(child, Var):
                raise TypeError("A sentence can only be converted to a "
                                "model if it's a conjunction of variables.")
            if child.name in model:
                raise ValueError("{!r} appears multiple times."
                                 .format(child.name))
            model[child.name] = child.true

        return model

    def condition(self, model: Model) -> 'NNF':
        """Fill in all the values in the dictionary."""
        @memoize
        def cond(node: NNF) -> NNF:
            if isinstance(node, Var):
                if node.name not in model:
                    return node
                if model[node.name] == node.true:
                    return true
                return false
            elif isinstance(node, Internal):
                new = node.__class__(map(cond, node.children))
                if new != node:
                    return new
                return node
            else:
                raise TypeError(type(node))

        return cond(self)

    def make_smooth(self) -> 'NNF':
        """Transform the sentence into an equivalent smooth sentence."""
        @memoize
        def filler(name: Name) -> 'Or':
            return Or({Var(name), Var(name, False)})

        @memoize
        def smooth(node: NNF) -> NNF:
            if isinstance(node, And):
                new = And(smooth(child)
                          for child in node.children)  # type: NNF
            elif isinstance(node, Var):
                return node
            elif isinstance(node, Or):
                names = node.vars()
                children = {smooth(child) for child in node.children}
                smoothed = set()  # type: t.Set[NNF]
                for child in children:
                    child_names = child.vars()
                    if len(child_names) < len(names):
                        child_children = {child}
                        child_children.update(filler(name)
                                              for name in names - child_names)
                        child = And(child_children)
                    smoothed.add(child)
                new = Or(smoothed)
            else:
                raise TypeError(node)

            if new == node:
                return node
            return new

        return smooth(self)

    def simplify(self, merge_nodes: bool = True) -> 'NNF':
        """Apply the following transformations to make the sentence simpler:

        - If an And node has `false` as a child, replace it by `false`
        - If an Or node has `true` as a child, replace it by `true`
        - Remove children of And nodes that are `true`
        - Remove children of Or nodes that are `false`
        - If an And or Or node only has one child, replace it by that child
        - If an And or Or node has a child of the same type, merge them

        :param merge_nodes: if ``False``, don't merge internal nodes. In
                            certain cases, merging them may increase the
                            size of the sentence.
        """
        # TODO: which properties does this preserve?

        @memoize
        def simple(node: NNF) -> NNF:
            if isinstance(node, Var):
                return node
            new_children = set()  # type: t.Set[NNF]
            if isinstance(node, Or):
                for child in map(simple, node.children):
                    if child == true:
                        return true
                    elif child == false:
                        pass
                    elif isinstance(child, Or) and merge_nodes:
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
                    elif isinstance(child, And) and merge_nodes:
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

    def deduplicate(self) -> 'NNF':
        """Return a copy of the sentence without any duplicate objects.

        If a node has multiple parents, it's possible for it to be
        represented by two separate objects. This method gets rid of that
        duplication.

        In a lot of cases it's better to avoid the duplication in the first
        place, for example with a Builder object.
        """
        new_nodes = {}  # type: t.Dict[NNF, NNF]

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
        ids = set()  # type: t.Set[int]

        def count(node: NNF) -> None:
            ids.add(id(node))
            if isinstance(node, Internal):
                for child in node.children:
                    if id(child) not in ids:
                        count(child)

        count(self)
        return len(ids)

    def to_DOT(
            self,
            *,
            color: bool = False,
            color_dict: t.Optional[t.Dict[str, str]] = None,
            label: str = 'text',
            label_dict: t.Optional[t.Dict[str, str]] = None
    ) -> str:
        """Return a representation of the sentence in the DOT language.

        `DOT <https://en.wikipedia.org/wiki/DOT_(graph_description_language)>`_
        is a graph visualization language.

        :param color: If ``True``, color the nodes. This is a bit of an
                      eyesore, but might make them easier to understand.
        :param label: If ``'text'``, the default, label nodes with "AND",
                      "OR", etcetera. If ``'symbol'``, label them with
                      unicode symbols like "\u2227" and "\u22a5".
        :param color_dict: Use an alternative palette. This should be a
                           dictionary with keys ``'and'``, ``'or'``,
                           ``'true'``, ``'false'``, ``'var'`` and ``'neg'``.
                           Not all keys have to be included. Passing a
                           dictionary implies ``color=True``.
        :param label_dict: Use alternative labels for nodes. This should be
                           a dictionary with keys ``'and'``, ``'or'``,
                           ``'true'`` and ``'false'``. Not all keys have to
                           be included.
        """
        colors = {
            'and': 'lightblue',
            'or': 'yellow',
            'true': 'green',
            'false': 'red',
            'var': 'chartreuse',
            'neg': 'pink',
        }

        if color_dict is not None:
            color = True
            colors.update(color_dict)

        if label == 'text':
            labels = {
                'and': 'AND',
                'or': 'OR',
                'true': 'TRUE',
                'false': 'FALSE',
            }
        elif label == 'symbol':
            labels = {
                'and': '\u2227',  # \wedge
                'or': '\u2228',  # \vee
                'true': '\u22a4',  # \top
                'false': '\u22a5',  # \bot
            }
        else:
            raise ValueError("Unknown label style {!r}".format(label))

        if label_dict is not None:
            labels.update(label_dict)

        counter = itertools.count()
        names = {}  # type: t.Dict[NNF, t.Tuple[int, str, str]]
        arrows = []  # type: t.List[t.Tuple[int, int]]

        def name(node: NNF) -> int:
            if node not in names:
                number = next(counter)
                if isinstance(node, Var):
                    label = str(node.name).replace('"', r'\"')
                    color = colors['var']
                    if not node.true:
                        label = '¬' + label
                        color = colors['neg']
                    names[node] = (number, label, color)
                    return number
                elif node == true:
                    kind = 'true'
                elif node == false:
                    kind = 'false'
                elif isinstance(node, And):
                    kind = 'and'
                elif isinstance(node, Or):
                    kind = 'or'
                else:
                    raise TypeError("Can't handle node of type {}"
                                    .format(type(node)))
                names[node] = (number, labels[kind], colors[kind])
            return names[node][0]

        for node in sorted(self.walk()):
            name(node)
        for node in sorted(self.walk()):
            if isinstance(node, Internal):
                for child in sorted(node.children):
                    arrows.append((name(node), name(child)))

        return '\n'.join(
            ['digraph {'] +
            [
                '    {} [label="{}"'.format(number, label)
                + (' fillcolor="{}" style=filled]'.format(fillcolor)
                   if color else ']')
                for number, label, fillcolor in names.values()
            ] +
            [
                '    {} -> {}'.format(src, dst)
                for src, dst in arrows
            ] +
            ['}\n']
        )

    def _models_deterministic(
            self,
            decomposable: _Tristate = None
    ) -> t.Iterator[Model]:
        """Model enumeration for deterministic sentences.

        Slightly faster for sentences that are also decomposable.
        """
        ModelInt = t.FrozenSet[t.Tuple[Name, bool]]

        if decomposable is None:
            decomposable = self.decomposable()

        if decomposable:
            def compatible(a: ModelInt, b: ModelInt) -> bool:
                return True
        else:
            def compatible(a: ModelInt, b: ModelInt) -> bool:
                if len(a) > len(b):
                    a, b = b, a
                return not any((name, not value) in b for name, value in a)

        @memoize
        def extract(node: NNF) -> t.Set[ModelInt]:
            if isinstance(node, Var):
                return {frozenset(((node.name, node.true),))}
            elif isinstance(node, Or):
                return {model
                        for child in node.children
                        for model in extract(child)}
            elif isinstance(node, And):
                models = {frozenset()}  # type: t.Set[ModelInt]
                for child in node.children:
                    models = {existing | new
                              for new in extract(child)
                              for existing in models
                              if compatible(existing, new)}
                return models
            raise TypeError(node)

        names = self.vars()
        full_models = set()  # type: t.Set[ModelInt]

        def complete(
                model: ModelInt,
                names: t.List[Name]
        ) -> t.Iterator[ModelInt]:
            for expansion in all_models(names):
                yield frozenset(model | expansion.items())

        for model in extract(self):
            missing_names = list(names - {name for name, value in model})
            if not missing_names:
                full_models.add(model)
            else:
                full_models.update(complete(model, missing_names))

        for full_model in full_models:
            yield dict(full_model)

    def _models_decomposable(self) -> t.Iterator[Model]:
        """Model enumeration for decomposable sentences."""
        if not self.satisfiable(decomposable=True):
            return
        names = tuple(self.vars())
        model_tree = {}  # type: t.Dict[bool, t.Any]

        def leaves(
                tree: t.Dict[bool, t.Any],
                path: t.Tuple[bool, ...] = ()
        ) -> t.Iterator[t.Tuple[t.Dict[bool, t.Any], t.Tuple[bool, ...]]]:
            if not tree:
                yield tree, path
            else:
                for key, val in tree.items():
                    yield from leaves(val, path + (key,))

        for var in names:
            for leaf, path in leaves(model_tree):
                model = dict(zip(names, path))
                model[var] = True
                if self._consistent_with_model(model):
                    leaf[True] = {}
                model[var] = False
                if self._consistent_with_model(model):
                    leaf[False] = {}
                assert leaf  # at least one of them has to be satisfiable

        for leaf, path in leaves(model_tree):
            yield dict(zip(names, path))

    @abc.abstractmethod
    def _sorting_key(self) -> t.Tuple[t.Any, ...]:
        """Used for sorting nodes in a (mostly) consistent order.

        The sorting is fairly arbitrary, and mostly tuned to make .to_DOT()
        output nice. The rules are approximately:
        - Variables first
        - Variables with lower-sorting stringified names first
        - Negations last
        - Nodes with a lower height first
        - Nodes with fewer children first
        - Nodes with higher-sorting children last

        Note that Var(10) and Var("10") are sorted as equal.
        """
        ...

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, NNF):
            return NotImplemented
        return self._sorting_key() < other._sorting_key()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, NNF):
            return NotImplemented
        return self._sorting_key() <= other._sorting_key()

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, NNF):
            return NotImplemented
        return self._sorting_key() > other._sorting_key()

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, NNF):
            return NotImplemented
        return self._sorting_key() >= other._sorting_key()


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

    :ivar name: The name of the variable. Can be any hashable object.
    :ivar true: Whether the variable is true. If ``False``, the variable is
                negated.
    """

    __slots__ = ('name', 'true')

    if t.TYPE_CHECKING:
        def __init__(self, name: Name, true: bool = True) -> None:
            # For the typechecker
            self.name = name
            self.true = true
    else:
        def __init__(self, name: Name, true: bool = True) -> None:
            # For immutability
            object.__setattr__(self, 'name', name)
            object.__setattr__(self, 'true', true)

    def __eq__(self, other: t.Any) -> t.Any:
        if self.__class__ is other.__class__:
            return self.name == other.name and self.true == other.true
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.name, self.true))

    def __setattr__(self, key: str, value: object) -> None:
        raise TypeError("{} objects are immutable"
                        .format(self.__class__.__name__))

    def __delattr__(self, name: str) -> None:
        raise TypeError("{} objects are immutable"
                        .format(self.__class__.__name__))

    def __repr__(self) -> str:
        if isinstance(self.name, str):
            return str(self.name) if self.true else "~{}".format(self.name)
        else:
            base = "{}({!r})".format(self.__class__.__name__, self.name)
            return base if self.true else '~' + base

    def __invert__(self) -> 'Var':
        return Var(self.name, not self.true)

    def decision_node(self) -> bool:
        return False

    def _sorting_key(self) -> t.Tuple[bool, str, bool]:
        return False, str(self.name), not self.true


class Internal(NNF):
    """Base class for internal nodes, i.e. And and Or nodes."""
    __slots__ = ('children',)

    if t.TYPE_CHECKING:
        def __init__(self, children: t.Iterable[NNF] = ()) -> None:
            # For the typechecker
            self.children = frozenset(children)
    else:
        def __init__(self, children: t.Iterable[NNF] = ()) -> None:
            # For immutability
            object.__setattr__(self, 'children', frozenset(children))

    def __eq__(self, other: t.Any) -> t.Any:
        if self.__class__ is other.__class__:
            return self.children == other.children
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.children,))

    def __setattr__(self, key: str, value: object) -> None:
        raise TypeError("{} objects are immutable"
                        .format(self.__class__.__name__))

    def __delattr__(self, name: str) -> None:
        raise TypeError("{} objects are immutable"
                        .format(self.__class__.__name__))

    def __repr__(self) -> str:
        if self.children:
            return ("{}({{{}}})"
                    .format(self.__class__.__name__,
                            ', '.join(map(repr, self.children))))
        else:
            return "{}()".format(self.__class__.__name__)

    def leaf(self) -> bool:
        if self.children:
            return False
        return True

    def _is_simple(self) -> bool:
        """Whether all children are leaves that don't share variables."""
        variables = set()  # type: t.Set[Name]
        for child in self.children:
            if not child.leaf():
                return False
            if isinstance(child, Var):
                if child.name in variables:
                    return False
                variables.add(child.name)
        return True

    def _sorting_key(self) -> t.Tuple[bool, int, int, str, t.List[NNF]]:
        return (True, self.height(), len(self.children),
                self.__class__.__name__, sorted(self.children, reverse=True))


class And(Internal):
    """Conjunction nodes, which are only true if all of their children are."""
    __slots__ = ()

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
    __slots__ = ()

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
    """Create a decision node with a variable and two branches.

    :param var: The variable node to decide on.
    :param if_true: The branch if the decision is true.
    :param if_false: The branch if the decision is false.
    """
    return Or({And({var, if_true}), And({~var, if_false})})


#: A node that's always true. Technically an And node without children.
true = And()
#: A node that's always false. Technically an Or node without children.
false = Or()


class Builder:
    """Automatically deduplicates NNF nodes as you make them, to save memory.

    Usage:

    >>> builder = Builder()
    >>> var = builder.Var('A')
    >>> var2 = builder.Var('A')
    >>> var is var2
    True

    As long as the Builder object exists, the nodes it made will be kept in
    memory. Make sure not to keep it around longer than you need.

    It's often a better idea to avoid creating nodes multiple times in the
    first place. That will save processing time as well as memory.

    If you use a Builder, avoid using operators. Even negating variables
    should be done with ``builder.Var(name, False)`` or they won't be
    deduplicated.
    """
    def __init__(self, seed: t.Iterable[NNF] = ()):
        """:param seed: Nodes to store for reuse in advance."""
        self.stored = {true: true, false: false}  # type: t.Dict[NNF, NNF]
        for node in seed:
            self.stored[node] = node
        self.true = true
        self.false = false

    def Var(self, name: Name, true: bool = True) -> 'nnf.Var':
        ret = Var(name, true)
        return self.stored.setdefault(ret, ret)  # type: ignore

    def And(self, children: t.Iterable[NNF] = ()) -> 'nnf.And':
        ret = And(children)
        return self.stored.setdefault(ret, ret)  # type: ignore

    def Or(self, children: t.Iterable[NNF] = ()) -> 'nnf.Or':
        ret = Or(children)
        return self.stored.setdefault(ret, ret)  # type: ignore
