capstone_project
==============================

A short description of the project.

Project Organization
------------

    │
    ├── data/               <- The original, immutable data dump. 
    │
    ├── reports/            <-  Report....
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
    ├── README.md           <- The top-level README for developers using this project.
    │
    └── setup.py             <- setup.py ....


--------

<p><small>Project structure based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</p>

------------

For google cloud: Follow steps 1-4 of:
https://medium.com/google-cloud/running-jupyter-notebooks-on-gpu-on-google-cloud-d44f57d22dbd
 

Install anaconda as explained in step 5 but instead of install tensorflow with 
pip install the complete virtual environment with conda and activate it:

```bash
$ conda env create -f environment.yml
$ source activate example-project 
```

then follow step 6-8:
https://medium.com/google-cloud/running-jupyter-notebooks-on-gpu-on-google-cloud-d44f57d22dbd

then

Install `capstone_project` in the virtual environment:

```bash
$ pip install --editable .
```

--------

https://www.kaggle.com/artimous/quora-question-pairs/reach-the-count-words-benchmark

Frameworks/platforms

NB. If you publish work that uses NLTK, please cite the NLTK book as follows:

    Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.