.. nnf documentation master file, created by
   sphinx-quickstart on Wed Jun 12 15:24:57 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ``python-nnf``'s documentation!
==========================================

``python-nnf`` is a package for working with logical sentences written in the `negation normal form <https://en.wikipedia.org/wiki/Negation_normal_form>`_.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   install
   nnf
   caveats

Introduction
------------

Sentences are made up of nodes. To start with, define some variables:

.. code-block:: pycon

   >>> from nnf import Var
   >>> A, B, C = Var('A'), Var('B'), Var('C')

Then, if you want to write the sentence "A or B":

.. code-block:: pycon

   >>> from nnf import Or
   >>> sentence = Or({A, B})
   >>> sentence = A | B  # alternative syntax

Or "B and not C":

.. code-block:: pycon

   >>> from nnf import And
   >>> sentence = And({B, ~C})
   >>> sentence = B & ~C

Of course you can nest these, for more interesting sentences:

.. code-block:: pycon

   >>> sentence = Or({And({A, B}), And({~B, C})})

You can ask queries, and perform transformations:

.. code-block:: pycon

   >>> sentence.decomposable()
   True
   >>> sentence.smooth()
   False
   >>> list(sentence.models())
   [{'A': True, 'B': True, 'C': True}, {'A': True, 'B': False, ...
   >>> new = sentence.condition({'B': True})
   >>> new
   Or({And({A, true}), And({false, C})})
   >>> list(new.models())
   [{'A': True, 'C': True}, {'A': True, 'C': False}]
   >>> new.simplify()
   A

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
