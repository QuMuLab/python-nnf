import copy
import pickle
import shutil
import os

from pathlib import Path

import pytest

from hypothesis import (assume, event, given, strategies as st, settings,
                        HealthCheck)

import nnf

from nnf import Var, And, Or, amc, dimacs, dsharp, operators, true, false, tseitin

settings.register_profile('patient', deadline=2000,
                          suppress_health_check=(HealthCheck.too_slow,))
settings.load_profile('patient')

a, b, c = Var('a'), Var('b'), Var('c')

fig1a = (~a & b) | (a & ~b)
fig1b = (~a | ~b) & (a | b)

uf20 = [
    dsharp.load(file.open())
    for file in (Path(os.path.dirname(__file__))
                 / 'testdata' / 'satlib' / 'uf20').glob('*.nnf')
]
uf20_cnf = [
    dimacs.load(file.open())
    for file in (Path(os.path.dirname(__file__))
                 / 'testdata' / 'satlib' / 'uf20').glob('*.cnf')
]


def test_all_models_basic():
    assert list(nnf.all_models([])) == [{}]
    assert list(nnf.all_models([1])) == [{1: False}, {1: True}]
    assert len(list(nnf.all_models(range(10)))) == 1024


@given(st.sets(st.integers(), max_size=8))
def test_all_models(names):
    result = list(nnf.all_models(names))
    # Proper result size
    assert len(result) == 2**len(names)
    # Only real names, only booleans
    assert all(name in names and isinstance(value, bool)
               for model in result
               for name, value in model.items())
    # Only complete models
    assert all(len(model) == len(names)
               for model in result)
    # No duplicate models
    assert len({tuple(model.items()) for model in result}) == len(result)


def test_basic():
    assert a.satisfied_by(dict(a=True))
    assert (a | b).satisfied_by(dict(a=False, b=True))
    assert not (a & b).satisfied_by(dict(a=True, b=False))

    assert (a & b).satisfiable()
    assert not (a & ~a).satisfiable()
    assert not (a & (~a & b)).satisfiable()

    assert ((a | b) & (b | c)).satisfiable()


def test_amc():
    assert amc.NUM_SAT(fig1a) == 2
    assert amc.NUM_SAT(fig1b) == 4

    assert amc.GRAD(a, {'a': 0.5}, 'a') == (0.5, 1)


names = st.integers(1, 8)


@st.composite
def variables(draw):
    return Var(draw(names), draw(st.booleans()))


@st.composite
def booleans(draw):
    return draw(st.sampled_from((nnf.true, nnf.false)))


@st.composite
def leaves(draw):
    return draw(st.one_of(variables(), booleans()))


@st.composite
def terms(draw):
    return And(Var(name, draw(st.booleans()))
               for name in draw(st.sets(names)))


@st.composite
def clauses(draw):
    return Or(Var(name, draw(st.booleans()))
              for name in draw(st.sets(names)))


@st.composite
def DNF(draw):
    return Or(draw(st.frozensets(terms())))


@st.composite
def CNF(draw):
    sentence = And(draw(st.frozensets(clauses())))
    assume(len(sentence.children) > 0)
    return sentence


@st.composite
def models(draw):
    return And(Var(name, draw(st.booleans()))
               for name in range(1, 9))


@st.composite
def MODS(draw):
    return Or(draw(st.frozensets(models())))


@st.composite
def internal(draw, children):
    return draw(st.sampled_from((And, Or)))(draw(st.frozensets(children)))


@st.composite
def NNF(draw):
    return draw(st.recursive(variables(), internal))


@st.composite
def DNNF(draw):
    sentence = draw(NNF())
    assume(sentence.decomposable())
    return sentence


@given(DNF())
def test_hyp(sentence: nnf.Or):
    assume(len(sentence.children) != 0)
    assume(sentence.decomposable())
    assert sentence.satisfiable()
    assert sentence.vars() <= set(range(1, 9))


@given(MODS())
def test_MODS(sentence: nnf.Or):
    assert sentence.smooth()
    assert sentence.flat()
    assert sentence.decomposable()
    assert sentence.simply_conjunct()


@given(MODS())
def test_MODS_satisfiable(sentence: nnf.Or):
    if len(sentence.children) > 0:
        assert sentence.satisfiable()
    else:
        assert not sentence.satisfiable()


@pytest.fixture(scope='module', params=[True, False])
def merge_nodes(request):
    return request.param


@given(sentence=DNNF())
def test_DNNF_sat_strategies(sentence: nnf.NNF, merge_nodes):
    sat = sentence.satisfiable()
    if sat:
        assert sentence.simplify(merge_nodes) != nnf.false
        assert amc.SAT(sentence)
        event("Sentence satisfiable")
    else:
        assert sentence.simplify(merge_nodes) == nnf.false
        assert not amc.SAT(sentence)
        event("Sentence not satisfiable")


def test_amc_numsat():
    for sentence in uf20:
        assert (amc.NUM_SAT(sentence.make_smooth())
                == len(list(sentence.models())))


@given(sentence=NNF())
def test_idempotent_simplification(sentence: nnf.NNF, merge_nodes):
    sentence = sentence.simplify(merge_nodes)
    assert sentence.simplify(merge_nodes) == sentence


@given(sentence=NNF())
def test_simplify_preserves_meaning(sentence: nnf.NNF, merge_nodes):
    simple = sentence.simplify(merge_nodes)
    assert sentence.equivalent(simple)
    for model in sentence.models():
        assert simple.satisfied_by(model)
    for model in simple.models():
        assert sentence.condition(model).simplify(merge_nodes) == nnf.true


@given(sentence=NNF())
def test_simplify_eliminates_bools(sentence: nnf.NNF, merge_nodes):
    assume(sentence != nnf.true and sentence != nnf.false)
    if any(node == nnf.true or node == nnf.false
           for node in sentence.walk()):
        event("Sentence contained booleans originally")
    sentence = sentence.simplify(merge_nodes)
    if sentence == nnf.true or sentence == nnf.false:
        event("Sentence simplified to boolean")
    else:
        for node in sentence.walk():
            assert node != nnf.true and node != nnf.false


@given(NNF())
def test_simplify_merges_internal_nodes(sentence: nnf.NNF):
    if any(any(type(node) == type(child)
               for child in node.children)
           for node in sentence.walk()
           if isinstance(node, nnf.Internal)):
        event("Sentence contained immediately mergeable nodes")
        # Nodes may also be merged after intermediate nodes are removed
    for node in sentence.simplify().walk():
        if isinstance(node, nnf.Internal):
            for child in node.children:
                assert type(node) != type(child)


@given(sentence=DNNF())
def test_simplify_solves_DNNF_satisfiability(sentence: nnf.NNF, merge_nodes):
    if sentence.satisfiable():
        event("Sentence is satisfiable")
        assert sentence.simplify(merge_nodes) != nnf.false
    else:
        event("Sentence is not satisfiable")
        assert sentence.simplify(merge_nodes) == nnf.false


def test_dimacs_sat_serialize():
    # http://www.domagoj-babic.com/uploads/ResearchProjects/Spear/dimacs-cnf.pdf
    sample_input = """c Sample SAT format
c
p sat 4
(*(+(1 3 -4)
   +(4)
   +(2 3)))
"""
    assert dimacs.loads(sample_input) == And({
        Or({Var(1), Var(3), ~Var(4)}),
        Or({Var(4)}),
        Or({Var(2), Var(3)})
    })


@pytest.mark.parametrize(
    'serialized, sentence',
    [
        ('p sat 2\n(+((1)+((2))))', Or({Var(1), Or({Var(2)})}))
    ]
)
def test_dimacs_sat_weird_input(serialized: str, sentence: nnf.NNF):
    assert dimacs.loads(serialized) == sentence


def test_dimacs_cnf_serialize():
    sample_input = """c Example CNF format file
c
p cnf 4 3
1 3 -4 0
4 0 2
-3
"""
    assert dimacs.loads(sample_input) == And({
        Or({Var(1), Var(3), ~Var(4)}),
        Or({Var(4)}),
        Or({Var(2), ~Var(3)})
    })


@given(NNF())
def test_arbitrary_dimacs_sat_serialize(sentence: nnf.NNF):
    assert dimacs.loads(dimacs.dumps(sentence)) == sentence
    # Removing spaces may change the meaning, but shouldn't make it invalid
    # At least as far as our parser is concerned, a more sophisticated one
    # could detect variables with too high names
    serial = dimacs.dumps(sentence).split('\n')
    serial[1] = serial[1].replace(' ', '')
    dimacs.loads('\n'.join(serial))


@given(CNF())
def test_arbitrary_dimacs_cnf_serialize(sentence: nnf.And):
    assume(all(len(clause.children) > 0 for clause in sentence.children))
    assert dimacs.loads(dimacs.dumps(sentence, mode='cnf')) == sentence


@given(NNF())
def test_dimacs_cnf_serialize_accepts_only_cnf(sentence: nnf.NNF):
    if (isinstance(sentence, And)
            and all(isinstance(clause, Or)
                    and all(isinstance(var, Var)
                            for var in clause.children)
                    and len(clause.children) > 0
                    for clause in sentence.children)):
        event("CNF sentence")
        dimacs.dumps(sentence, mode='cnf')
    else:
        event("Not CNF sentence")
        with pytest.raises(TypeError):
            dimacs.dumps(sentence, mode='cnf')


@pytest.mark.parametrize(
    'fname, clauses',
    [
        ('bf0432-007.cnf', 3667),
        ('sw100-1.cnf', 3100),
        ('uuf250-01.cnf', 1065),
        ('uf20-01.cnf', 90),
    ]
)
def test_cnf_benchmark_data(fname: str, clauses: int):
    with open(os.path.dirname(__file__) + '/testdata/satlib/' + fname) as f:
        sentence = dimacs.load(f)
    assert isinstance(sentence, And) and len(sentence.children) == clauses


@pytest.mark.parametrize(
    'fname',
    [
        'uf20-01'
    ]
)
def test_dsharp_output(fname: str):
    basepath = os.path.dirname(__file__) + '/testdata/satlib/' + fname
    with open(basepath + '.nnf') as f:
        sentence = dsharp.load(f)
    with open(basepath + '.cnf') as f:
        clauses = dimacs.load(f)
    assert sentence.decomposable()
    # this is not a complete check, but clauses.models() is very expensive
    assert all(clauses.satisfied_by(model) for model in sentence.models())


@given(NNF())
def test_walk_unique_nodes(sentence: nnf.NNF):
    result = list(sentence.walk())
    assert len(result) == len(set(result))
    assert len(result) <= sentence.size() + 1


@given(st.dictionaries(st.integers(), st.booleans()))
def test_to_model(model: dict):
    sentence = nnf.And(nnf.Var(k, v) for k, v in model.items())
    assert sentence.to_model() == model


@given(NNF())
def test_models_smart_equivalence(sentence: nnf.NNF):
    dumb = list(sentence.models())
    smart = list(sentence._models_deterministic())
    assert model_set(dumb) == model_set(smart)


@pytest.mark.parametrize(
    'sentence, size',
    [
        ((a & b), 2),
        (a & (a | b), 4),
        ((a | b) & (~a | ~b), 6),
        (And({
            Or({a, b}),
            And({a, Or({a, b})}),
        }), 6)
    ]
)
def test_size(sentence: nnf.NNF, size: int):
    assert sentence.size() == size


@pytest.mark.parametrize(
    'a, b, contradictory',
    [
        (a, ~a, True),
        (a, b, False),
        (a, a, False),
        (a & b, a & ~b, True),
        (a & (a | b), b, False),
        (a & (a | b), ~a, True),
    ]
)
def test_contradicts(a: nnf.NNF, b: nnf.NNF, contradictory: bool):
    assert a.contradicts(b) == contradictory


@given(NNF())
def test_false_contradicts_everything(sentence: nnf.NNF):
    assert nnf.false.contradicts(sentence)


@given(DNNF())
def test_equivalent(sentence: nnf.NNF):
    assert sentence.equivalent(sentence)
    assert sentence.equivalent(sentence | nnf.false)
    assert sentence.equivalent(sentence & (nnf.Var('A') | ~nnf.Var('A')))
    if sentence.satisfiable():
        assert not sentence.equivalent(sentence & nnf.false)
        assert not sentence.equivalent(sentence & nnf.Var('A'))
    else:
        assert sentence.equivalent(sentence & nnf.false)
        assert sentence.equivalent(sentence & nnf.Var('A'))


@given(NNF(), NNF())
def test_random_equivalent(a: nnf.NNF, b: nnf.NNF):
    if a.vars() != b.vars():
        if a.equivalent(b):
            event("Equivalent, different vars")
            assert b.equivalent(a)
            for model in a.models():
                assert b.condition(model).valid()
            for model in b.models():
                assert a.condition(model).valid()
        else:
            event("Not equivalent, different vars")
            assert (any(not b.condition(model).valid()
                        for model in a.models()) or
                    any(not a.condition(model).valid()
                        for model in b.models()))
    else:
        if a.equivalent(b):
            event("Equivalent, same vars")
            assert b.equivalent(a)
            assert model_set(a.models()) == model_set(b.models())
        else:
            event("Not equivalent, same vars")
            assert model_set(a.models()) != model_set(b.models())


@given(NNF())
def test_smoothing(sentence: nnf.NNF):
    if not sentence.smooth():
        event("Sentence not smooth yet")
        smoothed = sentence.make_smooth()
        assert type(sentence) is type(smoothed)
        assert smoothed.smooth()
        assert sentence.equivalent(smoothed)
        assert smoothed.make_smooth() == smoothed
    else:
        event("Sentence already smooth")
        assert sentence.make_smooth() == sentence


def hashable_dict(model):
    return frozenset(model.items())


def model_set(model_gen):
    return frozenset(map(hashable_dict, model_gen))


def test_uf20_models():

    for sentence in uf20:
        assert sentence.decomposable()
        m = list(sentence.models(deterministic=False,
                                 decomposable=True))
        models = model_set(m)
        assert len(m) == len(models)
        assert models == model_set(sentence.models(deterministic=True,
                                                   decomposable=False))
        assert models == model_set(sentence.models(deterministic=True,
                                                   decomposable=True))


@given(NNF())
def test_deterministic_models_always_works(sentence: nnf.NNF):
    if sentence.deterministic():
        event("Sentence is deterministic")
    else:
        event("Sentence is not deterministic")
    with_det = list(sentence.models(deterministic=True))
    no_det = list(sentence.models(deterministic=False))
    assert len(with_det) == len(no_det)
    assert model_set(with_det) == model_set(no_det)


def test_instantiating_base_classes_fails():
    with pytest.raises(TypeError):
        nnf.NNF()
    with pytest.raises(TypeError):
        nnf.Internal()
    with pytest.raises(TypeError):
        nnf.Internal({nnf.Var(3)})


@given(NNF())
def test_negation(sentence: nnf.NNF):
    n_vars = len(sentence.vars())
    models_orig = model_set(sentence.models())
    models_negated = model_set(sentence.negate().models())
    assert len(models_orig) + len(models_negated) == 2**n_vars
    assert len(models_orig | models_negated) == 2**n_vars


@given(NNF())
def test_model_counting(sentence: nnf.NNF):
    assert sentence.model_count() == len(list(sentence.models()))


def test_uf20_model_counting():
    for sentence in uf20:
        assert (sentence.model_count(deterministic=True)
                == len(list(sentence.models())))


@given(NNF())
def test_validity(sentence: nnf.NNF):
    if sentence.valid():
        event("Valid sentence")
        assert all(sentence.satisfied_by(model)
                   for model in nnf.all_models(sentence.vars()))
    else:
        event("Invalid sentence")
        assert any(not sentence.satisfied_by(model)
                   for model in nnf.all_models(sentence.vars()))


def test_uf20_validity():
    for sentence in uf20:
        assert not sentence.valid(deterministic=True)


@given(CNF())
def test_is_CNF(sentence: nnf.NNF):
    assert sentence.is_CNF()
    assert not sentence.is_DNF()


@given(DNF())
def test_is_DNF(sentence: nnf.NNF):
    assert sentence.is_DNF()
    assert not sentence.is_CNF()


@given(NNF())
def test_to_MODS(sentence: nnf.NNF):
    assume(len(sentence.vars()) <= 5)
    mods = sentence.to_MODS()
    assert mods.is_MODS()
    assert isinstance(mods, Or)
    assert mods.model_count() == len(mods.children)


@given(MODS())
def test_is_MODS(sentence: nnf.NNF):
    assert sentence.is_MODS()


@given(NNF())
def test_pairwise(sentence: nnf.NNF):
    new = sentence.make_pairwise()
    assert new.equivalent(sentence)
    if new not in {nnf.true, nnf.false}:
        assert all(len(node.children) == 2
                   for node in new.walk()
                   if isinstance(node, nnf.Internal))


@given(NNF())
def test_implicates(sentence: nnf.NNF):
    implicates = sentence.implicates()
    assert implicates.equivalent(sentence)
    assert implicates.is_CNF()
    assert not any(a.children < b.children
                   for a in implicates.children
                   for b in implicates.children)


@given(NNF())
def test_implicants(sentence: nnf.NNF):
    implicants = sentence.implicants()
    assert implicants.equivalent(sentence)
    assert implicants.is_DNF()
    assert not any(a.children < b.children
                   for a in implicants.children
                   for b in implicants.children)


@given(NNF())
def test_implicates_implicants_idempotent(sentence: nnf.NNF):
    assume(len(sentence.vars()) <= 6)
    implicants = sentence.implicants()
    implicates = sentence.implicates()
    assert implicants.implicants() == implicants
    assert implicates.implicates() == implicates
    assert implicants.implicates() == implicates
    assert implicates.implicants() == implicants


# TODO: This test fails, see the example below.
# I don't know if this is a bug in the test or in the implementation.
@pytest.mark.xfail
@given(NNF())
def test_implicates_implicants_negation_rule(sentence: nnf.NNF):
    assert sentence.negate().implicants().negate() == sentence.implicates()
    assert sentence.negate().implicates().negate() == sentence.implicants()


@pytest.mark.xfail(strict=True)
def test_implicates_implicants_negation_rule_example():
    sentence = Or({And({~Var(1), Var(2)}), And({~Var(3), Var(1)})})
    assert sentence.negate().implicants().negate() == sentence.implicates()
    assert sentence.negate().implicates().negate() == sentence.implicants()


@given(NNF(), NNF())
def test_implies(a: nnf.NNF, b: nnf.NNF):
    if a.implies(b):
        event("Implication")
        for model in a.models():
            assert b.condition(model).valid()
    else:
        event("No implication")
        assert any(not b.condition(model).valid()
                   for model in a.models())


@given(CNF())
def test_cnf_sat(sentence: nnf.NNF):
    assert sentence.is_CNF()
    assert sentence.satisfiable(cnf=True) == sentence.satisfiable(cnf=False)
    assert (model_set(sentence.models(cnf=True)) ==
            model_set(sentence.models(cnf=False, deterministic=True)))


def test_uf20_cnf_sat():
    for sentence in uf20_cnf:
        assert sentence.is_CNF()
        assert sentence.satisfiable()
        # It would be nice to compare .models() output to another algorithm
        # But even 20 variables is too much
        # So let's just hope that test_cnf_sat does enough
        at_least_one = False
        for model in sentence.models(cnf=True):
            assert sentence.satisfied_by(model)
            at_least_one = True
        assert at_least_one


@given(NNF(), NNF())
def test_xor(a: nnf.NNF, b: nnf.NNF):
    c = operators.xor(a, b)
    for model in nnf.all_models(c.vars()):
        assert (a.satisfied_by(model) ^ b.satisfied_by(model) ==
                c.satisfied_by(model))


@given(NNF(), NNF())
def test_nand(a: nnf.NNF, b: nnf.NNF):
    c = operators.nand(a, b)
    for model in nnf.all_models(c.vars()):
        assert ((a.satisfied_by(model) and b.satisfied_by(model)) !=
                c.satisfied_by(model))


@given(NNF(), NNF())
def test_nor(a: nnf.NNF, b: nnf.NNF):
    c = operators.nor(a, b)
    for model in nnf.all_models(c.vars()):
        assert ((a.satisfied_by(model) or b.satisfied_by(model)) !=
                c.satisfied_by(model))


@given(NNF(), NNF())
def test_implies(a: nnf.NNF, b: nnf.NNF):
    c = operators.implies(a, b)
    for model in nnf.all_models(c.vars()):
        assert ((a.satisfied_by(model) and not b.satisfied_by(model)) !=
                c.satisfied_by(model))


@given(NNF(), NNF())
def test_implied_by(a: nnf.NNF, b: nnf.NNF):
    c = operators.implied_by(a, b)
    for model in nnf.all_models(c.vars()):
        assert ((b.satisfied_by(model) and not a.satisfied_by(model)) !=
                c.satisfied_by(model))


@given(NNF(), NNF())
def test_iff(a: nnf.NNF, b: nnf.NNF):
    c = operators.iff(a, b)
    for model in nnf.all_models(c.vars()):
        assert ((a.satisfied_by(model) == b.satisfied_by(model)) ==
                c.satisfied_by(model))


@given(NNF())
def test_pickling(sentence: nnf.NNF):
    new = pickle.loads(pickle.dumps(sentence))
    assert sentence == new
    assert sentence is not new
    assert sentence.object_count() == new.object_count()


@given(NNF())
def test_copying_does_not_copy(sentence: nnf.NNF):
    assert sentence is copy.copy(sentence) is copy.deepcopy(sentence)
    assert copy.deepcopy([sentence])[0] is sentence


if shutil.which('dsharp') is not None:
    def test_dsharp_compile_uf20():
        for sentence in uf20_cnf:
            compiled = dsharp.compile(sentence)
            compiled_smooth = dsharp.compile(sentence, smooth=True)
            assert sentence.equivalent(compiled)
            assert sentence.equivalent(compiled_smooth)
            assert compiled.decomposable()
            assert compiled_smooth.decomposable()
            assert compiled_smooth.smooth()

    @given(CNF())
    def test_dsharp_compile(sentence: And[Or[Var]]):
        assume(all(len(clause) > 0 for clause in sentence))
        compiled = dsharp.compile(sentence)
        compiled_smooth = dsharp.compile(sentence, smooth=True)
        assert compiled.decomposable()
        assert compiled_smooth.decomposable()
        assert compiled_smooth.smooth()
        if sentence.satisfiable():  # See nnf.dsharp.__doc__
            assert sentence.equivalent(compiled)
            assert sentence.equivalent(compiled_smooth)

    @given(CNF())
    def test_dsharp_compile_converting_names(sentence: And[Or[Var]]):
        assume(all(len(clause) > 0 for clause in sentence))
        sentence = And(Or(Var(str(var.name), var.true) for var in clause)
                       for clause in sentence)
        compiled = dsharp.compile(sentence)
        assert all(isinstance(name, str) for name in compiled.vars())
        if sentence.satisfiable():
            assert sentence.equivalent(compiled)


@given(NNF())
def test_tseitin(sentence: nnf.NNF):

    # Assumption to reduce the time in testing
    assume(sentence.size() <= 10)

    T = tseitin.to_CNF(sentence)

    # TODO: Once forgetting/projection is implemented,
    #       do this more complete check
    # aux = filter(lambda x: 'aux' in str(x.name), T.vars())
    # assert T.forget(aux).equivalent(sentence)

    for mt in T.models():
        assert sentence.satisfied_by(mt)

    assert T.model_count() == sentence.model_count()
