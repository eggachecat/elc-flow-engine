import setuptools

with open("README.md", "r", encoding='UTF-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="elc-flow",
    version="0.0.1",
    author="sunao",
    author_email="sunao_0626@hotmail.com",
    description="A graph flow engine for East Low Carbon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eggachecat/elc-flow-engine",
    packages=setuptools.find_packages(exclude=("tests",)),
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
)