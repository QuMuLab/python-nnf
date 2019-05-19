import os

import pytest

from hypothesis import assume, event, given, strategies as st

import nnf

from nnf import Var, And, Or, amc, dimacs

a, b, c = Var('a'), Var('b'), Var('c')

fig1a = (~a & b) | (a & ~b)
fig1b = (~a | ~b) & (a | b)


def test_all_models_basic():
    assert list(nnf.all_models([])) == [{}]
    assert list(nnf.all_models([1])) == [{1: True}, {1: False}]
    assert len(list(nnf.all_models(range(10)))) == 1024


@given(st.sets(st.integers(), max_size=10))
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


names = st.integers(1, 10)


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
    return Or(draw(st.sets(terms())))


@st.composite
def CNF(draw):
    sentence = And(draw(st.sets(clauses())))
    assume(len(sentence.children) > 0)
    return sentence


@st.composite
def models(draw):
    return And(Var(name, draw(st.booleans()))
               for name in range(1, 11))


@st.composite
def MODS(draw):
    return Or(draw(st.sets(models())))


@st.composite
def internal(draw, children):
    return draw(st.sampled_from((And, Or)))(draw(st.sets(children)))


@st.composite
def NNF(draw):
    return draw(st.recursive(leaves(), internal))


@st.composite
def DNNF(draw):
    sentence = draw(NNF())
    assume(sentence.decomposable())
    return sentence


@st.composite
def sd_DNNF(draw):
    sentence = draw(NNF())
    assume(sentence.decomposable())
    assume(sentence.smooth())
    assume(sentence.deterministic())
    return sentence


@given(DNF())
def test_hyp(sentence: nnf.Or):
    assume(len(sentence.children) != 0)
    assume(sentence.decomposable())
    assert sentence.satisfiable()
    assert sentence.vars() <= set(range(1, 11))


@given(MODS())
def test_MODS(sentence: nnf.Or):
    assert sentence.smooth()
    assert sentence.flat()
    assert sentence.decomposable()
    assert sentence.simply_conjunct()


@given(MODS())
def test_MODS_satisfiable(sentence: nnf.Or):
    assume(len(sentence.children) != 0)
    assert sentence.satisfiable()


@given(DNNF())
def test_DNNF_sat_strategies(sentence: nnf.NNF):
    sat = sentence.satisfiable()
    if sat:
        assert sentence.simplify() != nnf.false
        assert amc.SAT(sentence)
        event("Sentence satisfiable")
    else:
        assert sentence.simplify() == nnf.false
        assert not amc.SAT(sentence)
        event("Sentence not satisfiable")


@given(sd_DNNF())
def test_amc_numsat(sentence: nnf.NNF):
    if not sentence.satisfiable():
        event("Sentence not satisfiable")
    assert amc.NUM_SAT(sentence) == len(list(sentence.models()))


@given(NNF())
def test_idempotent_simplification(sentence: nnf.NNF):
    sentence = sentence.simplify()
    assert sentence.simplify() == sentence


@given(NNF())
def test_simplify_preserves_meaning(sentence: nnf.NNF):
    simple = sentence.simplify()
    for model in sentence.models():
        assert simple.satisfied_by(model)
    for model in simple.models():
        assert sentence.instantiate(model).simplify() == nnf.true


@given(NNF())
def test_simplify_eliminates_bools(sentence: nnf.NNF):
    assume(sentence != nnf.true and sentence != nnf.false)
    if any(node == nnf.true or node == nnf.false
           for node in sentence.walk()):
        event("Sentence contained booleans originally")
    sentence = sentence.simplify()
    if sentence == nnf.true or sentence == nnf.false:
        event("Sentence simplified to boolean")
    else:
        for node in sentence.simplify().walk():
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
    ]
)
def test_cnf_benchmark_data(fname: str, clauses: int):
    with open(os.path.dirname(__file__) + '/testdata/satlib/' + fname) as f:
        sentence = dimacs.load(f)
    assert isinstance(sentence, And) and len(sentence.children) == clauses


@given(NNF())
def test_walk_unique_nodes(sentence: nnf.NNF):
    result = list(sentence.walk())
    assert len(result) == len(set(result))
    assert len(result) <= sentence.size() + 1


@given(st.dictionaries(st.integers(), st.booleans()))
def test_to_model(model: dict):
    sentence = nnf.And(nnf.Var(k, v) for k, v in model.items())
    assert sentence.to_model() == model
