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

__version__ = '0.2.1'

import abc
import functools
import itertools
import operator
import os
import shutil
import subprocess
import threading
import typing as t
import uuid
import weakref

from collections import Counter

from nnf.util import (
    memoize,
    weakref_memoize,
    T_NNF,
    U_NNF,
    T_NNF_co,
    Bottom,
    Name,
    Model,
    T,
    ReusableLazyIterable,
)


__all__ = (
    "NNF",
    "Internal",
    "And",
    "Or",
    "Var",
    "Aux",
    "all_models",
    "complete_models",
    "decision",
    "true",
    "false",
    "dsharp",
    "dimacs",
    "amc",
    "kissat",
    "config",
    "tseitin",
    "operators",
    "pysat",
)


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


class NNF(metaclass=abc.ABCMeta):
    """Base class for all NNF sentences."""
    __slots__ = ("__weakref__",)

    def __and__(self: T_NNF, other: U_NNF) -> 'And[t.Union[T_NNF, U_NNF]]':
        """And({self, other})"""
        return And({self, other})

    def __or__(self: T_NNF, other: U_NNF) -> 'Or[t.Union[T_NNF, U_NNF]]':
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

    @weakref_memoize
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
        """The children of Or nodes are variables that don't share names."""
        return all(node._is_simple()
                   for node in self.walk()
                   if isinstance(node, Or))

    def simply_conjunct(self) -> bool:
        """The children of And nodes are variables that don't share names."""
        return all(node._is_simple()
                   for node in self.walk()
                   if isinstance(node, And))

    @weakref_memoize
    def vars(self) -> t.FrozenSet[Name]:
        """The names of all variables that appear in the sentence."""
        return frozenset(node.name
                         for node in self.walk()
                         if isinstance(node, Var))

    def _memoized_vars(self) -> t.Callable[["NNF"], t.FrozenSet[Name]]:
        """Return a memoized alternative to the .vars() method.

        You should use this if you're going to query the variables of many
        nodes within a sentence.

        This returns a function so the cache is cleared when it's garbage
        collected.
        """

        @memoize
        def vars_(node: NNF) -> t.FrozenSet[Name]:
            if isinstance(node, Var):
                return frozenset({node.name})
            assert isinstance(node, Internal)
            return frozenset(
                v for child in node.children for v in vars_(child)
            )

        return vars_

    @weakref_memoize
    def decomposable(self) -> bool:
        """The children of each And node don't share variables, recursively."""
        vars_ = self._memoized_vars()

        for node in self.walk():
            if isinstance(node, And):
                seen = set()  # type: t.Set[Name]
                for child in node.children:
                    for name in vars_(child):
                        if name in seen:
                            return False
                        seen.add(name)
        return True

    # We want something set-like, but WeakSets are unreliable for reasons
    # explained in nnf.util.weakref_memoize.
    # We use a WeakValueDictionary that maps object IDs to the objects
    # themselves. This has the properties we want:
    # - Objects disappear from the dictionary when collected.
    # - We never forget that an object was marked.
    # Unlike @weakref_memoize, a result for one object never transfers to
    # an equal object. But since .mark_deterministic() is a manual optimization
    # it's probably better that way.
    _deterministic_sentences = (
        weakref.WeakValueDictionary()
    )  # type: weakref.WeakValueDictionary[int, NNF]

    def mark_deterministic(self) -> None:
        """Declare for optimization that this sentence is deterministic.

        Note that this goes by object identity, not equality. This may matter
        in obscure cases where you instantiate the same sentence multiple
        times.
        """
        NNF._deterministic_sentences[id(self)] = self

    def marked_deterministic(self) -> bool:
        """Whether this sentence has been marked as deterministic."""
        return id(self) in NNF._deterministic_sentences

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

    @weakref_memoize
    def smooth(self) -> bool:
        """The children of each Or node all use the same variables."""
        vars_ = self._memoized_vars()

        for node in self.walk():
            if isinstance(node, Or) and len(node.children) > 1:
                expected = None
                for child in node.children:
                    if expected is None:
                        expected = vars_(child)
                    else:
                        if vars_(child) != expected:
                            return False
        return True

    @abc.abstractmethod
    def decision_node(self) -> bool:
        """The sentence is a valid binary decision diagram (BDD)."""
        ...

    def clause(self) -> bool:
        """The sentence is a clause.

        Clauses are Or nodes with variable children that don't share names.
        """
        return isinstance(self, Or) and self._is_simple()

    def term(self) -> bool:
        """The sentence is a term.

        Terms are And nodes with variable children that don't share names.
        """
        return isinstance(self, And) and self._is_simple()

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

    def satisfiable(self) -> bool:
        """Some set of values exists that makes the sentence correct.

        This method doesn't necessarily try to find an example, which can
        make it faster. It's decent at decomposable sentences and sentences in
        CNF, and bad at other sentences.
        """
        if not self._satisfiable_decomposable():
            return False

        if self.decomposable():
            # Would've been picked up already if not satisfiable
            return True

        if self.is_CNF():
            return self._cnf_satisfiable()

        return self.to_CNF()._cnf_satisfiable()

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

    def valid(self) -> bool:
        """Check whether the sentence is valid (i.e. always true).

        This can be done efficiently for sentences that are decomposable and
        deterministic.
        """
        if self.marked_deterministic() and self.decomposable():
            # mypy is unsure that 2**<int> is actually an int
            # but len(self.vars()) >= 0, so it definitely is
            max_num_models = 2**len(self.vars())  # type: int
            return max_num_models == self.model_count()

        return not self.negate().satisfiable()

    def implies(self, other: 'NNF') -> bool:
        """Return whether ``other`` is always true if the sentence is true.

        This is faster if ``self`` is a term or ``other`` is a clause.
        """
        if other.clause():
            return not self.condition(other.negate().to_model()).satisfiable()
        if self.term():
            return not other.negate().condition(self.to_model()).satisfiable()

        if not self.vars() & other.vars():
            return not self.satisfiable() or other.valid()

        return not (self & other.negate()).satisfiable()

    entails = implies

    def models(self) -> t.Iterator[Model]:
        """Yield all dictionaries of values that make the sentence correct.

        Much faster on sentences that are decomposable. Even faster if they're
        also deterministic.
        """
        if self.is_CNF():
            yield from self._cnf_models()
        elif self.decomposable():
            if self.marked_deterministic():
                yield from self._models_deterministic()
            else:
                yield from self._models_decomposable()
        else:
            names = self.vars()
            for model in complete_models(self.to_CNF().models(), names):
                yield {
                    name: value
                    for name, value in model.items()
                    if name in names
                }

    def solve(self) -> t.Optional[Model]:
        """Return a satisfying model, or ``None`` if unsatisfiable."""
        if self.is_CNF():
            return self._cnf_solve()
        elif self.decomposable():
            # No special handling for d-DNNF, _decomposable_solve() already
            # uses a similar strategy to _models_deterministic()
            return self._decomposable_solve()
        else:
            solution = self.to_CNF()._cnf_solve()
            if solution is None:
                return None
            for key in solution.keys() - self.vars():
                del solution[key]
            for key in self.vars() - solution.keys():
                solution[key] = True
            return solution

    def _cnf_solve(self) -> t.Optional[Model]:
        # _cnf_satisfiable() always uses the native solver for very small
        # sentences because it outperforms pysat significantly for those,
        # but there's no such difference here, so we don't bother
        backend = config.sat_backend
        if backend == "auto":
            backend = "pysat" if pysat.available else "native"

        if backend == "native":
            for model in self._cnf_models_native():
                return model
            return None
        elif backend == "pysat":
            return pysat.solve(t.cast("And[Or[Var]]", self))
        raise AssertionError(config.sat_backend)

    def _decomposable_solve(self) -> t.Optional[Model]:
        @memoize
        def solve(node: NNF) -> t.Optional[Model]:
            if isinstance(node, Var):
                return {node.name: node.true}
            elif isinstance(node, And):
                model = {}
                for child in node:
                    extra = solve(child)
                    if extra is None:
                        return None
                    model.update(extra)
                return model
            elif isinstance(node, Or):
                for child in node:
                    solution = solve(child)
                    if solution is not None:
                        return solution
                return None
            raise AssertionError(node)

        solution = solve(self)
        if solution is not None:
            for key in self.vars() - solution.keys():
                solution[key] = True
        return solution

    def model_count(self) -> int:
        """Return the number of models the sentence has.

        This can be done efficiently for sentences that are decomposable and
        deterministic.
        """
        decomposable = self.decomposable()
        deterministic = self.marked_deterministic()
        made_smooth = False

        sentence = self
        if decomposable and deterministic and not sentence.smooth():
            sentence = sentence.make_smooth()
            made_smooth = True

        if (
            decomposable
            and deterministic
            and (made_smooth or sentence.smooth())
        ):
            @memoize
            def count(node: NNF) -> int:
                if isinstance(node, Var):
                    return 1
                elif isinstance(node, Or):
                    return sum(count(child) for child in node.children)
                elif isinstance(node, And):
                    return functools.reduce(
                        operator.mul,
                        (count(child) for child in node.children)
                    )
                else:
                    raise TypeError(node)

            return count(sentence)

        return sum(1 for _ in sentence.models())

    def contradicts(self, other: "NNF") -> bool:
        """There is no set of values that satisfies both sentences."""
        if not self.vars() & other.vars():
            return not (self.satisfiable() and other.satisfiable())

        return not (self & other).satisfiable()

    def equivalent(self, other: 'NNF') -> bool:
        """Test whether two sentences have the same models.

        If the sentences don't contain the same variables they are
        considered equivalent if the variables that aren't shared are
        independent, i.e. their value doesn't affect the value of the sentence.
        """
        if self == other:
            return True

        return not (
            (self & other.negate()) | (self.negate() & other)
        ).satisfiable()

    def negate(self) -> 'NNF':
        """Return a new sentence that's true iff the original is false."""
        @memoize
        def neg(node: NNF) -> NNF:
            if isinstance(node, Var):
                return ~node
            elif isinstance(node, And):
                return Or(neg(child) for child in node.children)
            elif isinstance(node, Or):
                return And(neg(child) for child in node.children)
            else:
                raise TypeError(node)

        return neg(self)

    def to_CNF(self) -> 'And[Or[Var]]':
        """Compile theory to a semantically equivalent CNF formula."""
        return tseitin.to_CNF(self)

    def _cnf_satisfiable(self) -> bool:
        """Call a SAT solver on the presumed CNF theory."""
        self = t.cast("And[Or[Var]]", self)
        if len(self) <= 5:
            # Always faster for such small sentences
            return self._cnf_satisfiable_native()

        backend = config.sat_backend
        if backend == "auto":
            backend = "pysat" if pysat.available else "native"

        if backend == "native":
            return self._cnf_satisfiable_native()
        elif backend == "pysat":
            return pysat.satisfiable(self)
        elif backend == "kissat":
            return kissat.solve(self) is not None
        raise AssertionError(config.sat_backend)

    def _cnf_satisfiable_native(self) -> bool:
        """A naive DPLL SAT solver."""
        def DPLL(clauses: t.FrozenSet[t.FrozenSet[Var]]) -> bool:
            if not clauses:
                return True
            if frozenset() in clauses:
                return False
            to_accept = set()  # type: t.Set[Var]
            to_remove = set()  # type: t.Set[Var]
            while True:
                for clause in clauses:
                    if len(clause) == 1:
                        var, = clause
                        if var in to_remove:  # contradiction
                            return False
                        to_accept.add(var)
                        to_remove.add(~var)
                        acc = {var}
                        rem = {~var}
                        clauses = frozenset(
                            clause - rem
                            for clause in clauses
                            if not clause & acc
                        )
                        if frozenset() in clauses:
                            return False
                        break
                else:
                    break
            if not clauses:
                return True
            remaining = frozenset(var for clause in clauses for var in clause)
            pure = frozenset(var for var in remaining if ~var not in remaining)
            clauses = frozenset(
                clause for clause in clauses
                if not clause & pure
            )
            if not clauses:
                return True
            new_var = Counter(var.name
                              for clause in clauses
                              for var in clause).most_common(1)[0][0]
            return (DPLL(clauses | {frozenset({Var(new_var)})}) or
                    DPLL(clauses | {frozenset({Var(new_var, False)})}))

        return DPLL(
            frozenset(
                frozenset(clause.children)
                for clause in self.children  # type: ignore
            )
        )

    def _cnf_models(self) -> t.Iterator[Model]:
        if config.models_backend in {"native", "auto"}:
            return self._cnf_models_native()
        elif config.models_backend == "pysat":
            return pysat.models(self)  # type: ignore
        raise AssertionError(config.sat_backend)

    def _cnf_models_native(self) -> t.Iterator[Model]:
        """A naive DPLL SAT solver, modified to find all solutions."""
        def DPLL_models(
                clauses: t.FrozenSet[t.FrozenSet[Var]]
        ) -> t.Iterator[Model]:
            if not clauses:
                yield {}
                return
            if frozenset() in clauses:
                return
            to_accept = set()  # type: t.Set[Var]
            to_remove = set()  # type: t.Set[Var]
            while True:
                for clause in clauses:
                    if len(clause) == 1:
                        var, = clause
                        if var in to_remove:  # contradiction
                            return
                        to_accept.add(var)
                        to_remove.add(~var)
                        acc = {var}
                        rem = {~var}
                        clauses = frozenset(
                            clause - rem
                            for clause in clauses
                            if not clause & acc
                        )
                        if frozenset() in clauses:
                            return
                        break
                else:
                    break
            solution = {var.name: var.true for var in to_accept}
            if not clauses:
                yield solution
                return
            new_var = Counter(var.name
                              for clause in clauses
                              for var in clause).most_common(1)[0][0]
            for refined_solution in itertools.chain(
                    DPLL_models(clauses | {frozenset({Var(new_var)})}),
                    DPLL_models(clauses | {frozenset({Var(new_var, False)})})
            ):
                assert not solution.keys() & refined_solution.keys()
                refined_solution.update(solution)
                yield refined_solution

        if not self.is_CNF():
            raise ValueError("Sentence not in CNF")

        names = self.vars()

        for model in DPLL_models(
            frozenset(
                frozenset(clause.children)
                for clause in self.children  # type: ignore
            )
        ):
            for missing in all_models(names - model.keys()):
                missing.update(model)
                yield missing

    def _do_PI(self) -> t.Tuple[t.Set['And[Var]'], t.Set['Or[Var]']]:
        """Compute the prime implicants and implicates of the sentence.

        This uses an algorithm adapted straightforwardly from
        PREVITI, Alessandro, et al. Prime compilation of non-clausal formulae.
        In: Twenty-Fourth International Joint Conference on Artificial
        Intelligence. 2015.
        """
        Model = t.Dict[t.Tuple[Name, bool], bool]

        def MaxModel(sentence: And[Or[Var]]) -> t.Optional[Model]:
            try:
                return max(sentence._cnf_models(),  # type: ignore
                           key=lambda model: sum(model.values()))
            except ValueError:
                return None

        def Map(model: Model) -> t.Iterator[Var]:
            for (var, truthness), value in model.items():
                if value:
                    yield Var(var, truthness)

        def ReduceImplicant(model: t.Set[Var], sentence: NNF) -> And[Var]:
            while True:
                for var in model:
                    if And(model - {var}).implies(sentence):
                        model.remove(var)
                        break
                else:
                    break
            return And(model)

        def ReduceImplicate(model: t.Set[Var], sentence: NNF) -> Or[Var]:
            model = {~var for var in model}
            while True:
                for var in model:
                    if sentence.implies(Or(model - {var})):
                        model.remove(var)
                        break
                else:
                    break
            return Or(model)

        F = self
        F_neg = self.negate()

        implicants = set()  # type: t.Set[And[Var]]
        implicates = set()  # type: t.Set[Or[Var]]

        H = And(Var((v, True), False) | Var((v, False), False)
                for v in self.vars())  # type: And[Or[Var]]
        while True:
            A_H = MaxModel(H)
            if A_H is None:
                return implicants, implicates
            A_F = set(Map(A_H))
            if not And(A_F | {F_neg}).satisfiable():
                I_n = ReduceImplicant(A_F, F)
                implicants.add(I_n)
                b = {Var((v.name, v.true), False)
                     for v in I_n.children}
            else:
                I_e = ReduceImplicate(A_F, F)
                implicates.add(I_e)
                b = {Var((v.name, v.true), True)
                     for v in I_e.children}
            H = And(H.children | {Or(b)})

    def implicants(self) -> 'Or[And[Var]]':
        """Extract the prime implicants of the sentence.

        Prime implicants are the minimal terms that imply the sentence. This
        method returns a disjunction of terms that's equivalent to the
        original sentence, and minimal, meaning that there are no terms that
        imply the sentence that are strict subsets of any of the terms in
        this representation, so no terms could be made smaller.
        """
        return Or(self._do_PI()[0])

    def implicates(self) -> 'And[Or[Var]]':
        """Extract a prime implicate cover of the sentence.

        Prime implicates are the minimal implied clauses. This method
        returns a conjunction of clauses that's equivalent to the original
        sentence, and minimal, meaning that there are no clauses implied by
        the sentence that are strict subsets of any of the clauses in this
        representation, so no clauses could be made smaller.

        While :meth:`implicants` returns all implicants, this method may
        only return some of the implicates.
        """
        return And(self._do_PI()[1])

    def to_MODS(self) -> 'Or[And[Var]]':
        """Convert the sentence to a MODS sentence."""
        new = Or(
            And(Var(name, val) for name, val in model.items())
            for model in self.models()
        )
        NNF.is_MODS.set(new, True)
        NNF._is_DNF_loose.set(new, True)
        NNF._is_DNF_strict.set(new, True)
        return new

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
                new = node.map(cond)
                if new != node:
                    return new
                return node
            else:
                raise TypeError(type(node))

        return cond(self)

    def make_smooth(self) -> 'NNF':
        """Transform the sentence into an equivalent smooth sentence."""
        vars_ = self._memoized_vars()

        @memoize
        def filler(name: Name) -> 'Or[Var]':
            return Or({Var(name), Var(name, False)})

        @memoize
        def smooth(node: NNF) -> NNF:
            if isinstance(node, And):
                new = And(smooth(child)
                          for child in node.children)  # type: NNF
            elif isinstance(node, Var):
                return node
            elif isinstance(node, Or):
                names = vars_(node)
                children = {smooth(child) for child in node.children}
                smoothed = set()  # type: t.Set[NNF]
                for child in children:
                    child_names = vars_(child)
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

        ret = smooth(self)
        NNF.smooth.set(ret, True)
        return ret

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

    def make_pairwise(self) -> 'NNF':
        """Alter the sentence so that all internal nodes have two children.

        This can be easier to handle in some cases.
        """
        sentence = self.simplify()

        if sentence == true or sentence == false:
            return sentence

        @memoize
        def pair(node: NNF) -> NNF:
            if isinstance(node, Var):
                return node
            elif isinstance(node, Internal):
                # After simplification, there are >=2 children
                a, *rest = node.children
                if len(rest) == 1:
                    return node.map(pair)
                else:
                    return type(node)({pair(a), pair(type(node)(rest))})
            else:
                raise TypeError(node)

        return pair(sentence)

    def project(self, names: 't.Iterable[Name]') -> 'NNF':
        """Dual of :meth:`forget`: will forget all variables not given"""
        return self.forget(self.vars() - frozenset(names))

    def forget_aux(self) -> 'NNF':
        """Returns a theory that forgets all of the auxillary variables"""
        return self.forget(v for v in self.vars() if isinstance(v, Aux))

    def forget(self, names: 't.Iterable[Name]') -> 'NNF':
        """Forget a set of variables from the theory.

        Has the effect of returning a theory without the variables provided,
        such that every model of the new theory has an extension (i.e., an
        assignment) to the forgotten variables that is a model of the original
        theory.

        :param names: An iterable of the variable names to be forgotten
        """

        if self.decomposable():
            return self._forget_with_subs(names)
        else:
            return self._forget_with_shannon(names)

    def _forget_with_subs(self, names: 't.Iterable[Name]') -> 'NNF':

        names = frozenset(names)

        @memoize
        def forget_recurse(node: NNF) -> NNF:
            if isinstance(node, Var) and node.name in names:
                return true
            elif isinstance(node, Var):
                return node
            elif isinstance(node, Internal):
                return node.map(forget_recurse)
            else:
                raise TypeError(node)

        return forget_recurse(self).simplify()

    def _forget_with_shannon(self, names: 't.Iterable[Name]') -> 'NNF':
        T = self
        for v in frozenset(names) & self.vars():
            T = T.condition({v: True}) | T.condition({v: False})
        return T.simplify()

    def deduplicate(self: T_NNF) -> T_NNF:
        """Return a copy of the sentence without any duplicate objects.

        If a node has multiple parents, it's possible for it to be
        represented by two separate objects. This method gets rid of that
        duplication.

        It's better to avoid the duplication in the first place. This method is
        for diagnostic purposes, in combination with :meth:`object_count`.
        """
        new_nodes = {}  # type: t.Dict[NNF, NNF]

        def recreate(node: U_NNF) -> U_NNF:
            if node not in new_nodes:
                if isinstance(node, Var):
                    new_nodes[node] = node
                elif isinstance(node, Or):
                    new_nodes[node] = node.map(recreate)
                elif isinstance(node, And):
                    new_nodes[node] = node.map(recreate)
            return new_nodes[node]  # type: ignore

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
                    if isinstance(node.name, Aux):
                        # This matches the repr, but in this context it could
                        # be reasonable to number them instead
                        label = "<{}>".format(node.name.hex[:4])
                    else:
                        label = str(node.name)
                    label = label.replace('"', r'\"')
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

    if shutil.which("dot"):

        def _repr_svg_(self) -> str:
            """Pretty rendering in Jupyter notebooks using graphviz.

            Inspired by the `graphviz<https://pypi.org/p/graphviz>`_ Python
            package, which implements _repr_svg_ the same way.
            """
            src = self.to_DOT()
            proc = subprocess.Popen(
                ["dot", "-Tsvg"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                universal_newlines=True,
            )
            out, _ = proc.communicate(src)
            return out

    def _models_deterministic(self) -> t.Iterator[Model]:
        """Model enumeration for deterministic decomposable sentences."""
        ModelInt = t.FrozenSet[t.Tuple[Name, bool]]

        def lazyproduct(
            iterables: t.Iterator[t.Iterable[ModelInt]],
        ) -> t.Iterator[ModelInt]:
            """Very specialized itertools.product alternative.

            Relies on iterables being an iterator, so we can take one and pass
            it on, while each iterable yielded by iterables is reusable.
            """
            iterable = next(iterables, None)
            if iterable is None:
                yield frozenset()
                return
            for model in lazyproduct(iterables):
                for own_model in iterable:
                    yield model | own_model

        @memoize
        def extract(node: NNF) -> t.Iterable[ModelInt]:
            if isinstance(node, Var):
                return [frozenset(((node.name, node.true),))]
            elif isinstance(node, Or):
                return ReusableLazyIterable(
                    model
                    for child in node.children
                    for model in extract(child)
                )
            elif isinstance(node, And):
                return ReusableLazyIterable(
                    lazyproduct(extract(child) for child in node.children)
                )
            else:
                raise TypeError(node)

        names = self.vars()

        def complete(
                model: ModelInt,
                names: t.FrozenSet[Name]
        ) -> t.Iterator[ModelInt]:
            for expansion in all_models(names):
                yield frozenset(model | expansion.items())

        for model in extract(self):
            missing_names = names - {name for name, _ in model}
            if not missing_names:
                yield dict(model)
            else:
                for full_model in complete(model, missing_names):
                    yield dict(full_model)

    def _models_decomposable(self) -> t.Iterator[Model]:
        """Model enumeration for decomposable sentences."""
        if not self.satisfiable():
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

    def is_CNF(self, strict: bool = False) -> bool:
        """Return whether the sentence is in the Conjunctive Normal Form.

        :param strict: If ``True``, follow the definition of the
            `Knowledge Compilation Map
            <https://jair.org/index.php/jair/article/view/10311>`_,
            requiring that a variable doesn't appear multiple times
            in a single clause.
        """
        if strict:
            return self._is_CNF_strict()
        return self._is_CNF_loose()

    @weakref_memoize
    def _is_CNF_loose(self) -> bool:
        return isinstance(self, And) and all(
            isinstance(child, Or)
            and all(isinstance(grandchild, Var) for grandchild in child)
            for child in self
        )

    @weakref_memoize
    def _is_CNF_strict(self) -> bool:
        return isinstance(self, And) and all(
            child.clause() for child in self.children
        )

    def is_DNF(self, strict: bool = False) -> bool:
        """Return whether the sentence is in the Disjunctive Normal Form.

        :param strict: If ``True``, follow the definition of the
            `Knowledge Compilation Map
            <https://jair.org/index.php/jair/article/view/10311>`_,
            requiring that a variable doesn't appear multiple times
            in a single term.
        """
        if strict:
            return self._is_DNF_strict()
        return self._is_DNF_loose()

    @weakref_memoize
    def _is_DNF_loose(self) -> bool:
        return isinstance(self, Or) and all(
            isinstance(child, And)
            and all(isinstance(grandchild, Var) for grandchild in child)
            for child in self
        )

    @weakref_memoize
    def _is_DNF_strict(self) -> bool:
        return isinstance(self, Or) and all(child.term()
                                            for child in self.children)

    @weakref_memoize
    def is_MODS(self) -> bool:
        """Return whether the sentence is in MODS form.

        MODS sentences are disjunctions of terms representing models,
        making the models trivial to enumerate.
        """
        return self.is_DNF(strict=True) and self.smooth()

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

    def __copy__(self: T_NNF) -> T_NNF:
        # Nodes are immutable, so this is ok
        return self

    def __deepcopy__(self: T_NNF, memodict: t.Dict[t.Any, t.Any]) -> T_NNF:
        return self


class Aux(uuid.UUID):
    """Unique UUID labels for auxiliary variables.

    Don't instantiate directly, call :meth:`Var.aux` instead.
    """

    __slots__ = ()


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

    __slots__ = {
        "name": "The name of the variable. Can be any hashable object.",
        "true": (
            "Whether the variable is true. If ``False``, the variable is "
            "negated."
        ),
    }

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
            base = str(self.name)
        elif isinstance(self.name, Aux):
            base = "<{}>".format(self.name.hex[:4])
        else:
            base = "{}({!r})".format(self.__class__.__name__, self.name)
        return base if self.true else '~' + base

    def __invert__(self) -> 'Var':
        return Var(self.name, not self.true)

    def decision_node(self) -> bool:
        return False

    def _sorting_key(self) -> t.Tuple[bool, str, bool]:
        return False, str(self.name), not self.true

    if t.TYPE_CHECKING:
        # Needed because mypy doesn't like it when you type the self
        # parameter with a subclass, even when using @overload
        def negate(self) -> 'Var':
            ...

        def make_smooth(self) -> 'Var':
            ...

    def __getstate__(self) -> t.Tuple[Name, bool]:
        return self.name, self.true

    def __setstate__(self, state: t.Tuple[Name, bool]) -> None:
        object.__setattr__(self, 'name', state[0])
        object.__setattr__(self, 'true', state[1])

    @staticmethod
    def aux() -> 'Var':
        """Create an auxiliary variable with a unique label."""
        # See implementation of uuid.uuid4()
        return Var(Aux(bytes=os.urandom(16), version=4))


class Internal(NNF, t.Generic[T_NNF_co]):
    """Base class for internal nodes, i.e. And and Or nodes."""
    __slots__ = ('children',)

    if t.TYPE_CHECKING:
        def __init__(self, children: t.Iterable[T_NNF_co] = ()) -> None:
            # For the typechecker
            self.children = frozenset(children)
    else:
        def __init__(self, children: t.Iterable[T_NNF_co] = ()) -> None:
            # For immutability
            object.__setattr__(self, 'children', frozenset(children))

    def __eq__(self, other: t.Any) -> t.Any:
        if self.__class__ is other.__class__:
            return self.children == other.children
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.__class__, self.children))

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

    def _repr_pretty_(self, printer: t.Any, cycle: bool) -> None:
        """Pretty-printing for IPython and Hypothesis.

        https://ipython.rtfd.io/en/stable/api/generated/IPython.lib.pretty.html
        """
        if not self.children:
            printer.text(repr(self))
            return
        name = self.__class__.__name__
        if cycle:
            # Impossible except by malicious use of object.__setattr__
            printer.text("{}(...)".format(name))
            return
        with printer.group(len(name) + 2, name + "({", "})"):
            for ind, child in enumerate(self):
                if ind:
                    printer.text(",")
                    printer.breakable()
                printer.pretty(child)

    def leaf(self) -> bool:
        if self.children:
            return False
        return True

    def _is_simple(self) -> bool:
        """Whether all children are variables that don't share names."""
        variables = set()  # type: t.Set[Name]
        for child in self.children:
            if not isinstance(child, Var):
                return False
            if child.name in variables:
                return False
            variables.add(child.name)
        return True

    def _sorting_key(self) -> t.Tuple[bool, int, int, str, t.List[NNF]]:
        return (True, self.height(), len(self.children),
                self.__class__.__name__, sorted(self.children, reverse=True))

    def map(self, func: t.Callable[[T_NNF_co], U_NNF]) -> 'Internal[U_NNF]':
        """Apply ``func`` to all of the node's children."""
        return type(self)(func(child)  # type: ignore
                          for child in self.children)

    def __iter__(self) -> t.Iterator[T_NNF_co]:
        """A shortcut for iterating over a node's children.

        This makes some code more natural, e.g.
        ``for implicant in node.implicants(): ...``
        """
        return iter(self.children)

    def __len__(self) -> int:
        """A shorcut for checking how many children a node has."""
        return len(self.children)

    def __contains__(self, item: NNF) -> bool:
        """A shorcut for checking if a node has a child."""
        return item in self.children

    def __bool__(self) -> bool:
        """Override the default behavior of empty nodes being ``False``.

        That would mean that ``bool(nnf.false) == bool(nnf.true) == False``,
        which is unreasonable. All nodes are truthy instead.
        """
        return True

    def __getstate__(self) -> t.FrozenSet[T_NNF_co]:
        return self.children

    def __setstate__(self, state: t.FrozenSet[T_NNF_co]) -> None:
        object.__setattr__(self, 'children', state)


class And(Internal[T_NNF_co]):
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

    def _repr_pretty_(self, printer: t.Any, cycle: bool) -> None:
        # An explicit definition is necessary or it will be ignored
        return super()._repr_pretty_(printer, cycle)

    if t.TYPE_CHECKING:
        def negate(self) -> 'Or[NNF]':
            ...

        def condition(self, model: Model) -> 'And[NNF]':
            ...

        def make_smooth(self) -> 'And[NNF]':
            ...

        def map(self, func: t.Callable[[T_NNF_co], U_NNF]) -> 'And[U_NNF]':
            ...


class Or(Internal[T_NNF_co]):
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

    def _repr_pretty_(self, printer: t.Any, cycle: bool) -> None:
        return super()._repr_pretty_(printer, cycle)

    if t.TYPE_CHECKING:
        def negate(self) -> 'And[NNF]':
            ...

        def condition(self, model: Model) -> 'Or[NNF]':
            ...

        def make_smooth(self) -> 'Or[NNF]':
            ...

        def map(self, func: t.Callable[[T_NNF_co], U_NNF]) -> 'Or[U_NNF]':
            ...


def complete_models(
        models: t.Iterable[Model],
        names: t.Iterable[Name]
) -> t.Iterator[Model]:
    names = frozenset(names)
    diff = None
    for model in models:
        if diff is None:
            diff = names - model.keys()
        for supplement in all_models(diff):
            new = model.copy()
            new.update(supplement)
            yield new


def decision(
        var: Var,
        if_true: T_NNF,
        if_false: U_NNF
) -> Or[t.Union[And[t.Union[Var, T_NNF]], And[t.Union[Var, U_NNF]]]]:
    """Create a decision node with a variable and two branches.

    :param var: The variable node to decide on.
    :param if_true: The branch if the decision is true.
    :param if_false: The branch if the decision is false.
    """
    return (var & if_true) | (~var & if_false)


#: A node that's always true. Technically an And node without children.
true = And()  # type: And[Bottom]
#: A node that's always false. Technically an Or node without children.
false = Or()  # type: Or[Bottom]


class _Setting(t.Generic[T]):
    """Use the descriptor protocol for a smart settings system."""
    def __init__(
        self, default: T, choices: t.Optional[t.Set[T]] = None
    ) -> None:
        self.choices = choices
        self.default = default
        self.local = threading.local()
        self.name = None  # type: t.Optional[str]

    def __set_name__(self, owner: object, name: str) -> None:
        self.name = name

    def __get__(self, instance: object, owner: object = None) -> T:
        return getattr(self.local, "value", self.default)  # type: ignore

    def __set__(self, instance: object, value: T) -> None:
        if self.choices is not None and value not in self.choices:
            if self.name is None:
                raise ValueError("Invalid value {!r}".format(value))
            raise ValueError(
                "Invalid value {!r} for setting {!r}".format(value, self.name)
            )
        self.local.value = value


_Func = t.TypeVar("_Func", bound=t.Callable[..., object])


class _ConfigContext:
    """An object to apply configuration as a context manager or decorator."""
    def __init__(self, settings: t.Dict[str, t.Any]) -> None:
        self.settings = settings
        self.old_settings = threading.local()

    def __enter__(self) -> None:
        self.old_settings.__dict__.setdefault("stack", []).append(
            {name: getattr(config, name) for name in self.settings}
        )
        for name, value in self.settings.items():
            setattr(config, name, value)

    def __exit__(self, *exc: object) -> None:
        for name, value in self.old_settings.stack.pop().items():
            setattr(config, name, value)

    def __call__(self, func: _Func) -> _Func:
        @functools.wraps(func)
        def newfunc(*args: t.Any, **kwargs: t.Any) -> t.Any:
            with self:
                return func(*args, **kwargs)

        return newfunc  # type: ignore


class _Config:
    """Configuration management.

    We need to instantiate this class to make the __set__ part of the
    descriptor protocol work and to take advantage of __slots__ so people can't
    misspell a setting without noticing.
    """

    # Remember to update the doc comment below whenever adding a setting
    sat_backend = _Setting("auto", {"auto", "native", "kissat", "pysat"})
    models_backend = _Setting("auto", {"auto", "native", "pysat"})
    pysat_solver = _Setting("minisat22")

    __slots__ = ()

    def __call__(self, **settings: str) -> _ConfigContext:
        return _ConfigContext(settings)


#: Configuration management.
#:
#: There are three ways to change a setting. Scoped::
#:
#:   >>> with config(sat_backend="native"):
#:   ...     do_something()
#:
#: Indefinite::
#:
#:   >>> config.sat_backend = "native"
#:   >>> do_something()
#:
#: And as a decorator::
#:
#:   >>> @config(sat_backend="native")
#:   ... def some_func():
#:   ...     do_something()
#:
#: Configuration is isolated per thread.
#:
#: The following settings are available:
#:
#: - ``sat_backend``: The backend used for SAT solving. Implicitly used by
#:   many methods.
#:
#:   - ``native``: A slow Python implementation. Always available.
#:   - ``pysat``: An implementation using the PySAT library. Generally faster
#:     than ``native``. Only available if the library is installed.
#:   - ``kissat``: An implementation using `kissat` as an external program.
#:     Fast, but high overhead, so relatively slow on small problems.
#:   - ``auto`` (default): Use ``pysat`` if available, otherwise ``native``.
#:
#: - ``models_backend``: The backend used for model enumeration.
#:
#:   - ``native``: A slow Python implementation. Always available.
#:   - ``pysat``: An implementation using PySAT. Often much faster than
#:     ``native``, but much slower on small problems with many models.
#:   - ``auto`` (default): Use ``native``. Behavior may change in the future.
#:
#: - ``pysat_solver``: The solver to use for `PySAT
#:   <https://pysathq.github.io/>`_. Can be any of the names in
#:   `pysat.solvers.SolverNames
#:   <https://pysathq.github.io/docs/html/api/solvers.html
#:   #pysat.solvers.SolverNames>`_. Default: ``minisat22``.
config = _Config()


from nnf import amc, dsharp, kissat, operators, pysat, tseitin  # noqa: E402
