capstone_project
==============================

A short description of the project.

Project Organization
------------

    │
    ├── data/               <- The original, immutable data dump. 
    │
    ├── figures/            <- Figures saved by scripts or notebooks.
    │
    ├── notebooks/          <- Jupyter notebooks. Naming convention is a short `-` delimited 
    │                         description, a number (for ordering), and the creator's initials,
    │                        e.g. `initial-data-exploration-01-hg`.
    │
    ├── output/             <- Manipulated data, logs, etc.
    │
    ├── tests/              <- Unit tests.
    │
    ├── capstone_project/      <- Python module with source code of this project.
    │
    ├── environment.yml     <- conda virtual environment definition file.
    │
    ├── LICENSE
    │
    ├── Makefile            <- Makefile with commands like `make environment`
    │
    ├── README.md           <- The top-level README for developers using this project.
    │
    └── tox.ini             <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project structure based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</p>

------------

Install the virtual environment with conda and activate it:

```bash
$ conda env create -f environment.yml
$ source activate example-project 
```

Install `capstone_project` in the virtual environment:

```bash
$ pip install --editable .
```

--------

https://www.kaggle.com/artimous/quora-question-pairs/reach-the-count-words-benchmark

Frameworks/platforms

NB. If you publish work that uses NLTK, please cite the NLTK book as follows:

    Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.