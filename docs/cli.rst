Command line interface
======================

Some of ``python-nnf``'s functionality is exposed through a command line tool. It can be invoked as ``pynnf`` or ``python3 -m nnf``.

SAT solving
-----------

``pynnf sat`` tests whether a sentence is satisfiable, while ``pynnf sharpsat`` counts how many solutions it has.

Add ``-v`` to get extra information about the sentence and the running time.

Example::

    $ pynnf sat uf20-01.cnf
    SATISFIABLE

Beware that it's much slower than dedicated solvers like `MiniSat <http://minisat.se/>`_.

Sentence summary
----------------

``pynnf info`` shows basic information about a sentence.

Examples::

    $ pynnf info uf20-01.cnf
    Sentence is in CNF.
    Variables:   20
    Size:        360
    Clauses:     90
    Clause size: 3

    $ pynnf info uf100-016.cnf.nnf
    Sentence is decomposable.
    Variables:   97
    Size:        109

Visualizing sentences
---------------------

``pynnf draw`` converts sentences to a `DOT <https://en.wikipedia.org/wiki/DOT_(graph_description_language)>`_ representation, and either outputs that or feeds it to ``dot`` to immediately output an image.

Immediately outputting an image requires having ``dot`` installed. It's done when the output file has an image extension, or when a format is passed with the ``-f`` flag.

Examples::

    $ pynnf draw uf20-01.nnf out.png  # Create a PNG image

    $ pynnf draw uf20-01.nnf out.gv   # Create a DOT representation

    $ pynnf draw uf20-01.nnf out.pdf  # Create a PDF vector image

    $ pynnf draw -f png uf20-01.nnf - | convert -flip - out.png  # Output a PNG image to be processed by imagemagick

See ``pynnf draw --help`` for more information.
