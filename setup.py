from setuptools import find_packages, setup

import pathlib

# Commands: 
# python3 -m black src/ --line-length 11
# python3 -m isort src/
# python setup.py sdist
# twine upload dist/*
# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Dependencies required to use
INSTALL_REQS = [
    "torch == 1.13.1",
    "pytorch_lightning == 1.9.1",
    # "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl",
    # "spacy==3.5.0",
    "transformers",
    "pyserini==0.20.0",
    "matplotlib==3.6.3",
    "rank-bm25",
    "faiss-gpu",
    "flair",
    "nltk",
    "pandas",
    "tqdm",
    "accelerate",
    "black",
    "isort",
    "pexpect",
    # "git+https://github.com/castorini/pygaggle.git", # INSTALL MANUALLY
    "bs4",
    "pydantic",
    "spacy",
    "gdown",
    "jsonlines"
]

setup(
    name="admiral",
    version="1.0",
    description="Repository for 'Natural-logic guided multi-hop document retrieval for fact verification', EMNLP2022.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Raldir/AdMIRaL",
    author="Rami Aly",
    author_email="rmya2@cam.ac.uk",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=INSTALL_REQS,
    python_requires=">=3.8",
)