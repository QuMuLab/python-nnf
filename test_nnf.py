import nnf

from nnf import Var, And, Or, amc

from hypothesis import assume, event, given, strategies as st

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
def DNF(draw):
    return Or(draw(st.sets(terms())))


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
        assert amc.NUM_SAT(sentence) > 0
        event("Sentence satisfiable")
    else:
        assert sentence.simplify() == nnf.false
        assert not amc.SAT(sentence)
        assert amc.NUM_SAT(sentence) == 0
        event("Sentence not satisfiable")
