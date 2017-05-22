"""Collections of functions that are used to preprocessing the Quora data"""
import string
import pandas as pd
import spacy
from scipy.sparse import hstack
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from capstone_project.models import benchmark_model

# Create english spacy instance. Very slow to load and should thus only be done once
nlp = spacy.load('en')
# List of stopwords that will be removed during tokenization
STOPLIST = stopwords.words("english")
# List of punctuation symbols that will be removed during tokenization
SYMBOLS = " ".join(string.punctuation).split(" ")


def tokenize(question):
    """Tokenize english text. The function takes a string as input and return a list of tokens."""
    doc = nlp(question)  # spacy expects unicode strings as input
    tokens = []

    #  Next paragraph taken from tutorial:i https://nicschrading.com/project/Intro-to-NLP-with-spaCy/
    for token in doc:
        tokens.append(token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_)

    tokens = [token for token in tokens if token not in STOPLIST]

    tokens = [token for token in tokens if token not in SYMBOLS]

    return tokens


class TfidfTransformer(BaseEstimator, TransformerMixin):
    """Takes a dataframe, calculates the tfidf values for the column question 1 and question 2, 
    outputs a sparse array of tfidf values for both questions.
    """
    def __init__(self):
        self.tfidf = TfidfVectorizer(strip_accents=None, lowercase=False,
                                     preprocessor=None, tokenizer=tokenize)

    def transform(self, df, **transform_params):
        tfidfs_q1 = self.tfidf.transform(df['question1'])
        tfidfs_q2 = self.tfidf.transform(df['question2'])
        # Stack horizontally since the words in q1 and q2 should be arranged in separate columns
        tfidf_q1_q2 = hstack([tfidfs_q1, tfidfs_q2])

        return tfidf_q1_q2

    def fit(self, df, y=None, **fit_params):
        # Fit tfidfs on the whole corpus
        self.tfidf.fit(pd.concat([df['question1'], df['question2']]))

        return self


class FeatureTransformer(BaseEstimator, TransformerMixin):

    def transform(self, df):
        # https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur
        # and https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question
        new_data = pd.DataFrame()
        new_data["q1_len"] = df["question1"].apply(lambda question: len(str(question)))
        new_data["q2_len"] = df["question2"].apply(lambda question: len(str(question)))
        new_data["word_share"] = df.apply(benchmark_model.word_match_share, axis=1)
        return new_data

    def fit(self, df, y=None, **fit_params):
        return self


def vector():
    # TODO: word2vec features cosine distance jaccard distance, other distances?
    pass
