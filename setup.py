import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tarpan",
    version="0.3.3",
    author="Evgenii Neumerzhitckii",
    author_email="sausageskin@gmail.com",
    description="A collection of tools for analysing cmdspanpy output, written in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evgenyneu/tarpan",
    packages=setuptools.find_packages(),
    install_requires=[
        'cmdstanpy>=0.8.0,<0.9.0',
        'scipy>=1.4.1,<2.0.0',
        'tabulate>=0.8.6,<0.9.0',
        'pandas>=0.25.3,<0.26.0',
        'numpy>=1.18.1,<2.0.0',
        'arviz>=0.6.1,<0.7.0',
        'matplotlib>=3.1.2,<4.0.0',
        'seaborn>=0.10.0,<0.11.0'
    ],
    tests_require=[
        'pytest==5.3.0'
    ],
    license='Public Domain',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Public Domain",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
