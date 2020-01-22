import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tarpan",
    version="0.1.1",
    author="Evgenii Neumerzhitckii",
    author_email="sausageskin@gmail.com",
    description="A collection of tools for analysing cmdspanpy output, written in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evgenyneu/tarpan",
    packages=setuptools.find_packages(),
    license='Public Domain',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Public Domain",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
