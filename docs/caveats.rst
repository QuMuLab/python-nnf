Caveats and warnings
====================

There are a few things to keep in mind when using ``python-nnf``.

Node duplication
----------------

If the same node occurs multiple times in a sentence, then it often pays to make sure that it isn't created multiple times.

Here's a (contrived) example of two ways to construct the same sentence:

.. code-block:: pycon

    >>> inefficient = And({
    ...     Or({A, B}),
    ...     And({A, Or({A, B})}),
    ... })
    >>> dup_node = Or({A, B})
    >>> efficient = And({
    ...     dup_node,
    ...     And({A, dup_node}),
    ... })

These objects behave identically, but the first one stores the node ``Or({A, B})`` twice, and the other stores it only once. That means the second one uses less memory.

For a lot of sentences, this isn't worth worrying about. But if you have many nodes that occur multiple times, and they descend from nodes that occur multiple times, you may end up using a lot more memory than necessary.

The ``.object_count()`` and ``.deduplicate()`` methods exist to diagnose this problem. ``.object_count()`` tells you how many actual objects are used to represent the sentence, and ``.deduplicate()`` returns a maximally compact copy.

If ``.deduplicate()`` changes the value of ``.object_count()`` a lot then the sentence could benefit from watching out not to create objects multiple times.

.. code-block:: pycon

    >>> inefficient.object_count()
    6
    >>> inefficient.deduplicate().object_count()
    5

In this case the difference is pretty small.

Decomposability and determinism
-------------------------------

A lot of methods are much faster to perform on sentences that are decomposable or deterministic, such as model enumeration.

Decomposability is automatically detected. However, you can skip the check if you already know whether the sentence is decomposable or not, by passing ``decomposable=True`` or ``decomposable=False`` as a keyword argument.

Determinism is too expensive to automatically detect, but it can give a huge speedup. If you know a sentence to be deterministic, pass ``deterministic=True`` as a keyword argument to take advantage.

A compiler like `DSHARP <https://bitbucket.org/haz/dsharp>`_ may be able to convert some sentences into equivalent deterministic decomposable sentences. The output of DSHARP can be loaded using the :mod:`nnf.dsharp` module.

Other duplication inefficiencies
--------------------------------

Even when properly deduplicated, the kind of sentence that's vulnerable to node duplication might still be inefficient to work with for some operations.

A known offender is equality (``==``). Currently, if two of such sentences are compared that are equal but don't share any objects, it takes a very long time even both sentences don't have any duplication within themselves.
