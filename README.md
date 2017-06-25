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
Installation:

If you want to setup google cloud with gpu support follow these steps:
 
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

git clone and cd to capstone_project

Install `capstone_project` in the virtual environment:

```bash
$ pip install --editable .
```

download data from:
https://www.kaggle.com/quora/question-pairs-dataset

and copy it into the data folder

then download the pretrained word2vec model:

https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

or from mirrored link:

https://github.com/mmihaltz/word2vec-GoogleNews-vectors

and copy it into the data folder if necessary

start a python console and execute

```python
import nltk
nltk.download()
```
then hit d and type stopwords in order to download the nltk stopwords.

then execute
```bash
python -m spacy.en.download all
```

to download the spacy model.

--------

Troubleshooting:

gensim seems to be broken on the google cloud platform and crashes on import

```
installing pip install google-compute-engine
```
should fix the problem https://github.com/RaRe-Technologies/gensim/issues/898

https://www.kaggle.com/artimous/quora-question-pairs/reach-the-count-words-benchmark

Frameworks/platforms

NB. If you publish work that uses NLTK, please cite the NLTK book as follows:

    Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.