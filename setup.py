import setuptools

with open("README.md") as f:
    long_description = f.read()

setuptools.setup(
    name="nnf",
    version='0.3.0',
    author="Jan Verbeek, Christian Muise",
    author_email="jan.verbeek@posteo.nl; christian.muise@queensu.ca",
    description="Manipulate NNF (Negation Normal Form) logical sentences",
    url="https://github.com/QuMuLab/python-nnf",
    packages=setuptools.find_packages(),
    package_data={
        'nnf': ['py.typed'],  # Mark package as having inline types
    },
    python_requires='>=3.4',
    install_requires=[
        'typing;python_version<"3.5"',
    ],
    extras_require={
        "pysat": ["python-sat"],
    },
    entry_points={
        'console_scripts': [
            'pynnf = nnf.cli:main',
        ],
    },
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
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: ISC License (ISCL)",
    ],
    keywords="logic nnf dimacs dsharp",
    project_urls={
        'Documentation': "https://python-nnf.readthedocs.io/",
        'Source': "https://github.com/QuMuLab/python-nnf",
    },
    include_package_data=True,
    zip_safe=False,
)
