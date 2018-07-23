import setuptools
from topojson import __version__ as version

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="topojson",
    version=version,
    author="Kevin Hill",
    author_email="kevin@humanpowered.tech",
    description="A package to manipulate topojson map data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kevhill/py-topojson",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
