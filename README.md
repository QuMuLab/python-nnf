[![Build Status](https://travis-ci.org/blyxxyz/python-nnf.svg?branch=master)](https://travis-ci.org/blyxxyz/python-nnf)
[![Readthedocs](https://readthedocs.org/projects/python-nnf/badge/)](https://python-nnf.readthedocs.io)
![Python Versions](https://img.shields.io/pypi/pyversions/nnf.svg)
![License](https://img.shields.io/pypi/l/nnf.svg)

`nnf` is a Python package for creating and manipulating logical sentences
written in the
[negation normal form](https://en.wikipedia.org/wiki/Negation_normal_form)
(NNF).

NNF sentences make statements about any number of variables. Here's an example:

```pycon
>>> from nnf import Var
>>> a, b = Var('a'), Var('b')
>>> sentence = (a & b) | (a & ~b)
>>> sentence
Or({And({a, b}), And({a, ~b})})
```

This sentence says that either a is true and b is true, or a is true and b
is false.

You can do a number of things with such a sentence. For example, you can ask
 whether a particular set of values for the variables makes the sentence true:

```pycon
>>> sentence.satisfied_by({'a': True, 'b': False})
True
>>> sentence.satisfied_by({'a': False, 'b': False})
False
```

You can also fill in a value for some of the variables:

```pycon
>>> sentence.condition({'b': True})
Or({And({a, true}), And({a, false})})
```

And then reduce the sentence:

```pycon
>>> _.simplify()
a
```

This package takes much of its data model and terminology from
[*A Knowledge Compilation Map*](https://jair.org/index.php/jair/article/view/10311).

Complete documentation can be found at [readthedocs](https://python-nnf.readthedocs.io).

# Installing

```sh
pip install nnf
```

At least Python 3.4 is required.

# Serialization

A parser and serializer for the
[DIMACS sat format](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html) are
implemented in `nnf.dimacs`, with a standard `load`/`loads`/`dump`/`dumps`
interface.

# Algebraic Model Counting

`nnf.amc` has a basic implementation of
[Algebraic Model Counting](https://arxiv.org/abs/1211.4475).
