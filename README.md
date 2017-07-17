Capstone Project
==============================

Find duplicate questions in the Quora dataset.

Project Organization
------------

    │
    ├── data/               <- The original, immutable data dump. 
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

Setting up the Environment
--------

My project uses python and several machine learning packages. I am using the anaconda 
environment and all the installed packages are shown in the environment.yml file.
If you have setup where Anaconda and CUDa and CUDNN is already installed you can just 
run the following commands to run and activate my python environment.

```bash
$ conda env create -f environment.yml
$ source activate capstone_project 
```
The LSTM model I am using should be trained on a GPU. Otherwise training will be very slow.
If you don't have a powerful GPU you can use cloud services like amazon aws or the google cloud platform.
In the following I will go through the step to set up my project on google cloud.
The tutorial is based on this excellent blog post:

https://medium.com/google-cloud/running-jupyter-notebooks-on-gpu-on-google-cloud-d44f57d22dbd

which helped my a lot while I was setting up my compute engine for the first time.
 
1. You can follow steps 1-4 of the above mentioned blog post. However, make to
use machine type n1-standard-4 instead of n1-standard-2, because the instance that
we will be using needs at least 15gb of RAM.
 
2. Install anaconda as explained in step 5 of the blog post. Do not installing just
tensorflow and keras with pip install. Install the complete virtual environment 
with conda and activate it using the commands below:

    ```bash
    $ conda env create -f environment.yml
    $ source activate capstone_project 
    ```

3. Again you can follow the blog post. Just follow step 6-8. There is no need to
change anything this time. 

Your system should be ready now and you should be able to use jupyter notebooks 
that are running on the google cloud platform with GPU support. As a next step we 
need to install my project and download the data.

Installing the capstone project
--------

1. Use git clone to download the project and go into the folder.

2. Install the `capstone_project` in the virtual environment by executing the
following command:

```bash
$ pip install --editable .
```

3. Download the data from: https://www.kaggle.com/quora/question-pairs-dataset
and copy it into the data folder. You might have to create the data folder first 
since it has been added to my gitignore file.

4. Now we should download the pretrained word2vec model that I am using. I found
three locations:

    https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
    https://github.com/mmihaltz/word2vec-GoogleNews-vectors
    https://groups.google.com/forum/#!topic/word2vec-toolkit/z0Aw5powUco

    and copy it into the data folder. The first link needs you to log in which makes it
hard to download the data with wget or curl. If you running on the cloud check out the
other two links first. 

5. The Glove model is not necessarily needed for my project, but I have been playing with
it a bit. So, if you want to use the glove model download it here:

    https://nlp.stanford.edu/projects/glove/

6. We need some data from the nltk corpus for this project, To download it
start a python console and execute

    ```python
    import nltk
    nltk.download()
    ```
    Then hit d and type stopwords in order to download the nltk stopwords.

7. We also need some data from spacy since I am using this module too. In order to
download the needed library just execute
    
    ```bash
    python -m spacy download en
    ```
    in a bash shell to download the spacy model.


How to run the code
--------

The most important parts of the project are the jupyter notebooks. Several 
functions and classes that are used frequently are located in the capstone_project
folder. Here is a list of the most important notebooks:

1. split-data-and-tokenize-01-als.ipynb <- Split the dataset into train/val/test set and tokenize the strings.

2. feature_engineering-01-als.ipynb     <- Create new features for classifiers (Notebook 1. needs to be run first once.) 

3. logistic_regression-01-als.ipynb     <- Train simple logistic 
model on the features created in step 2 and use benchmark model
(Run notebook 1. and 2. first one time.)

4. train_xgboost-01-als.ipynb           <- Improve predictions with boosted trees
(Run notebook 1. and 2. first one time.)

5. train_lstm-01-als.ipynb <- Train a LSTM model with word2vec embeddings (This notebook only depends on step 1.)

6. train_lstm-02-als.ipynb <- Train the final model and check performance on the test set (This notebook only depends on step 1.)


You can just run all of them in order. The training of these algorithms can take a long time
depending on you computer. If you  are running remotely on the cloud and if you want
to  disconnect while running you can export the notebook as a python script. 
The script can then run within tmux or screen which offer the option to 
detach a shell without stopping it. 

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