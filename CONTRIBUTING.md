# Contributing to python-nnf

## Running the tests and linters

[`tox`](https://tox.readthedocs.io/en/latest/) is used to run the tests and linters. After installing it, run:

```
tox
```

This will install and run all the tooling.

`tox` aborts early if one of the steps fails. To run just the tests (for example if you can't get `mypy` to work), install and run [`pytest`](https://docs.pytest.org/en/latest/getting-started.html). To run just one particular test, run `pytest -k <name of test>`.

To gather coverage information you can install `pytest-cov` and run `pytest --cov=nnf` followed by `coverage html`. This will generate a coverage report in a `htmlcov/` directory.

## Mypy

[Mypy](https://mypy.readthedocs.io/en/stable/) is used for static typing. This is also managed by `tox`.

You can look at the existing code for cues. If you can't figure it out, just leave it be and we'll look at it during code review.

## Hypothesis

[Hypothesis](https://hypothesis.readthedocs.io/en/latest/) is used for property-based testing: it generates random sentences for tests to use. See the existing tests for examples.

It's ideal for a feature to have both tests that do use Hypothesis and tests that don't.

## Memoization

[Memoization](https://en.wikipedia.org/wiki/Memoization) is used in various places to avoid computing the same thing multiple times. A memoized function remembers past calls so it can return the same return value again in the future.

A downside of memoization is that it increases memory usage.

It's used in two patterns throughout the codebase:

### Temporary internal memoization with `@memoize`

This is used for functions that run on individual nodes of sentences, within a method. For example:

```python
    def height(self) -> int:
        """The number of edges between here and the furthest leaf."""
        @memoize
        def height(node: NNF) -> int:
            if isinstance(node, Internal) and node.children:
                return 1 + max(height(child) for child in node.children)
            return 0

        return height(self)
```

Because the function is defined inside the method, it's garbage collected along with its cache when the method returns. This makes sure we don't keep all the individual node heights in memory indefinitely.

### Memoizing sentence properties with `@weakref_memoize`

This is used for commonly used methods that run on whole sentences. For example:

```python
    @weakref_memoize
    def vars(self) -> t.FrozenSet[Name]:
        """The names of all variables that appear in the sentence."""
        return frozenset(node.name
                         for node in self.walk()
                         if isinstance(node, Var))
```

This lets us call `.vars()` often without worrying about performance.

Unlike the other decorator, this one uses `weakref`, so it doesn't interfere with garbage collection. It's slightly less efficient though, so the temporary functions from the previous section are better off with `@memoize`.

## Documentation

Methods are documented with reStructuredText inside docstrings. This looks a little like markdown, but it's different, so take care and look at other docstrings for examples.

Documentation is automatically generated and ends up at [Read the Docs](https://python-nnf.readthedocs.io/en/latest/).

To build the documentation locally, run `make html` inside the `docs/` directory. This generates a manual in `docs/_build/html/`.

New modules have to be added to `docs/nnf.rst` to be documented.

## Style/miscellaneous

- Prefer sets over lists where it make sense. For example, `Or({~c for c in children} | {aux})` instead of `Or([~c for c in children] + [aux])`.
