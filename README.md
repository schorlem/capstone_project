Capstone Project
==============================

A short description of the project.

Project Organization
------------

    │
    ├── data/               <- The original, immutable data dump. 
    │
    ├── reports/            <- Final report 
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
    └── setup.py            


--------

<p><small>Project structure based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</p>

--------

Set up the python environment:

If you want to setup google cloud with gpu support follow these steps:
 
For google cloud: Follow steps 1-4 of:
https://medium.com/google-cloud/running-jupyter-notebooks-on-gpu-on-google-cloud-d44f57d22dbd

Make sure to create a instance with at least 15gb of ram.
Use machine type n1-standard-4 instead of n1-standard-2
 
Install anaconda as explained in step 5 but instead of install tensorflow with 
pip install the complete virtual environment with conda and activate it:

```bash
$ conda env create -f environment.yml
$ source activate example-project 
```

then follow step 6-8:
https://medium.com/google-cloud/running-jupyter-notebooks-on-gpu-on-google-cloud-d44f57d22dbd

then

INstallation of the capstone project

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
python -m spacy download en
```

to download the spacy model.

Usage:

The most important parts of the project are the jupyter notebooks. Several 
functions and classes that are used frequently are located in the capstone+project
folder.

I started of by spliting and tokenizing the data.

1) split-data-and-tokenize-01-als.ipynb <- Split the dataset into train/val/test set and tokenize the strings.
2) feature_engineering-01-als.ipynb     <- Create new features for classifiers 
3) logistic_regression-01-als.ipynb     <- Train simple logistic model on the features created in step 2 and use benchmark model
4) train_xgboost-01-als.ipynb           <- Improve predictions with boosted trees
5) train_lstm-01-als.ipynb              <- Train a LSTM model with word2vec embeddings (This notebook only depends on step 1)

Training of these algorithms will take a long time. If you 
are running remotely such as google cloud and if you want to 
disconnect while running you can export the notebook as a python script. 
The script can then run within tmux or screen which 
offer the option do detach a shell without stopping it. 

Troubleshooting
--------


The gensim module seems to be broken on the google cloud platform and it
crashes when importing it. In order to fix I ran the following command on
my gcloud instance: 

```
installing pip install google-compute-engine
```

See for more information: https://github.com/RaRe-Technologies/gensim/issues/898

------

https://www.kaggle.com/artimous/quora-question-pairs/reach-the-count-words-benchmark

Frameworks/platforms

NB. If you publish work that uses NLTK, please cite the NLTK book as follows:

    Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.