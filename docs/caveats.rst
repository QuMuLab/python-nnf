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

Decomposability
---------------

A lot of methods are much faster to perform on sentences that are decomposable, such as model enumeration. They use a different algorithm when they detect a sentence is decomposable.

They have a keyword argument that can be used to skip the check for decomposability. ``decomposable=True`` will always use the algorithm for decomposable sentences, and ``decomposable=False`` will use the slower algorithm. This may save a little time when doing a lot of operations on larger sentences, where determining whether they're decomposable every time takes too long.

A compiler like `DSHARP <https://bitbucket.org/haz/dsharp>`_ may be able to convert some non-decomposable sentences into equivalent decomposable sentences. The output of DSHARP can be loaded using the :mod:`nnf.dsharp` module.
