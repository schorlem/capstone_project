"""Collections of functions that are used to preprocessing the Quora data"""
import os
import sys
import pickle
import string
import pandas as pd
import numpy as np
import spacy
import gensim
from scipy.sparse import hstack
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from scipy.stats import skew, kurtosis
from capstone_project.models import benchmark_model


# Create english spacy instance. Very slow to load and should thus only be done once
nlp = spacy.load("en")
# List of stopwords that will be removed during tokenization
STOPLIST = stopwords.words("english")
# List of punctuation symbols that will be removed during tokenization
SYMBOLS = " ".join(string.punctuation).split(" ")


def save_as_pickle(dataset, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir+filename, "wb") as handle:
        pickle.dump(dataset, handle)


def load_pickle(input_dir, filename):
    with open(input_dir+filename, "rb") as handle:
        return pickle.load(handle)


def tokenize(question):
    """Tokenize english text. The function takes a string as input and return a list of tokens."""
    try:
        unicode_string = unicode(question, "utf8")
    except TypeError:
        unicode_string = question  # Don't convert to utf8 if input is already utf8.

    doc = nlp(unicode_string)  # spacy expects unicode strings as input
    tokens = []

    # Next paragraph taken from tutorial:i https://nicschrading.com/project/Intro-to-NLP-with-spaCy/
    for token in doc:
        tokens.append(token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_)

    tokens = [token for token in tokens if token not in STOPLIST]
    tokens = [token for token in tokens if token not in SYMBOLS]
    tokens = np.array(tokens)

    return tokens


class Word2vecTransformer(BaseEstimator, TransformerMixin):
    """Takes a tokenized list of words and transforms it into word2vec vectors. 
    If the option sum is set to True the transformer returns a normalised sum of these vectors."""
    def __init__(self, model=None, sum_up=False):
        self.sum_up = sum_up
        self.model = model
        #self.model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

    def transform(self, df):
        new_df = pd.DataFrame()
        if self.sum_up:
            new_df["q1_vecsum"] = df["q1_tokens"].apply(self._question_to_vector).apply(self._sum_vectors)
            new_df["q2_vecsum"] = df["q2_tokens"].apply(self._question_to_vector).apply(self._sum_vectors)
        else:
            new_df["q1_word2vec"] = df["q1_tokens"].apply(self._question_to_vector)
            new_df["q2_word2vec"] = df["q2_tokens"].apply(self._question_to_vector)

        return new_df

    def fit(self, df, y=None, **fit_params):
        return self

    def _question_to_vector(self, question):
        """Takes a list words as input and returns the word2vec matrix for these words.
        The word2vec model was pretrained by google."""
        vectors = []
        for w in question:
            try:
                vectors.append(self.model[w])
            except KeyError:
                continue  # Ignore words that are not in the vocabulary

        if len(vectors) == 0:
            return np.zeros((1, 300)) #TODO change to match model length

        vectors = np.array(vectors)
        return vectors

    def _sum_vectors(self, vectors):
        vector = vectors.sum(axis=0)
        return vector / np.sqrt((vector ** 2).sum())


class TfidfTransformer(BaseEstimator, TransformerMixin):
    """Takes a dataframe, calculates the tfidf values for the column question 1 and question 2, 
    outputs a sparse array of tfidf values for both questions.
    """
    def __init__(self):
        self._tfidf = TfidfVectorizer(strip_accents=None, lowercase=False,
                                      preprocessor=None, tokenizer=tokenize)

    def transform(self, df):
        tfidfs_q1 = self._tfidf.transform(df["question1"])
        tfidfs_q2 = self._tfidf.transform(df["question2"])
        # Stack horizontally since the words in q1 and q2 should be arranged in separate columns
        tfidf_q1_q2 = hstack([tfidfs_q1, tfidfs_q2])

        return tfidf_q1_q2

    def fit(self, df, y=None, **fit_params):
        # Fit tfidfs on the whole corpus
        self._tfidf.fit(pd.concat([df["question1"], df["question2"]]))

        return self


class FeatureTransformer(BaseEstimator, TransformerMixin):

    def transform(self, df):
        """Transform tokenized words into features. My guideline for the feature engineering was
         the following article https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur.
        """
        new_data = pd.DataFrame()
        new_data["q1_length"] = df["question1"].apply(lambda question: len(str(question)))
        new_data["q2_length"] = df["question2"].apply(lambda question: len(str(question)))
        new_data["diff_length"] = new_data["q1_length"] - new_data["q2_length"]
        new_data["q1_n_words"] = df["q1_tokens"].apply(lambda words: len(words))
        new_data["q2_n_words"] = df["q2_tokens"].apply(lambda words: len(words))
        new_data["q1_len_word_ratio"] = new_data["q1_length"]/new_data["q1_n_words"]
        new_data["q2_len_word_ratio"] = new_data["q2_length"]/new_data["q2_n_words"]
        new_data["word_share"] = df.apply(benchmark_model.word_match_share, axis=1)

        new_data = new_data.fillna(new_data.mean()) # Fill nan values of distance measures. Caused by questions with stopwords only
        new_data[np.isinf(new_data)] = sys.float_info.max

        return new_data

    def fit(self, df, y=None, **fit_params):
        return self


class VectorFeatureTransformer(BaseEstimator, TransformerMixin):

    def transform(self, df):
        """Transform tokenized words into features. My guideline for the feature engineering was
         the following article https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur.
        """
        new_data = pd.DataFrame()
        new_data["word2vec_cosine_distance"] = df.apply(lambda x: cosine(np.nan_to_num(x["q1_vecsum"]), np.nan_to_num(x["q2_vecsum"])), axis=1)
        new_data["word2vec_cityblock_distance"] = df.apply(lambda x: cityblock(np.nan_to_num(x["q1_vecsum"]),
                                                                               np.nan_to_num(x["q2_vecsum"])), axis=1)
        new_data["word2vec_jaccard_distance"] = df.apply(lambda x: jaccard(np.nan_to_num(x["q1_vecsum"]), np.nan_to_num(x["q2_vecsum"])), axis=1)
        new_data["word2vec_canberra_distance"] = df.apply(lambda x: canberra(np.nan_to_num(x["q1_vecsum"]), np.nan_to_num(x["q2_vecsum"])), axis=1)
        new_data["word2vec_minkowski_distance"] = df.apply(lambda x:
                                                           minkowski(np.nan_to_num(x["q1_vecsum"]), np.nan_to_num(x["q2_vecsum"]), 3), axis=1)
        new_data["word2vec_euclidean_distance"] = df.apply(lambda x:
                                                           euclidean(np.nan_to_num(x["q1_vecsum"]), np.nan_to_num(x["q2_vecsum"])), axis=1)
        new_data["word2vec_braycurtis_distance"] = df.apply(lambda x:
                                                            braycurtis(np.nan_to_num(x["q1_vecsum"]), np.nan_to_num(x["q2_vecsum"])), axis=1)
        new_data["word2vec_skew_q1"] = df["q1_vecsum"].apply(lambda vector: skew(np.nan_to_num(vector)))
        new_data["word2vec_skew_q2"] = df["q2_vecsum"].apply(lambda vector: skew(np.nan_to_num(vector)))
        new_data["word2vec_kurtosis_q1"] = df["q1_vecsum"].apply(lambda vector: kurtosis(np.nan_to_num(vector)))
        new_data["word2vec_kurtosis_q2"] = df["q2_vecsum"].apply(lambda vector: kurtosis(np.nan_to_num(vector)))

        new_data = new_data.fillna(new_data.mean()) # Fill nan values of distance measures. Caused by questions with stopwords only

        return new_data

    def fit(self, df, y=None, **fit_params):
        return self
