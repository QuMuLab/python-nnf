import setuptools

with open("README.md") as f:
    long_description = f.read()

setuptools.setup(
    name="nnf",
    version="0.1.0",
    author="Jan Verbeek",
    author_email="jan.verbeek@posteo.nl",
    description="Manipulate NNF (Negation Normal Form) logical sentences",
    url="https://github.com/blyxxyz/python-nnf",
    packages=setuptools.find_packages(),
    license="ISC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: ISC License (ISCL)",
    ],
)
