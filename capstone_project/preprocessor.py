"""Collections of functions that are used to preprocessing the Quora data"""
import string
import spacy
from nltk.corpus import stopwords
from capstone_project.models import benchmark_model

# A custom stoplist
STOPLIST = stopwords.words("english")
# List of symbols we don't care about
SYMBOLS = " ".join(string.punctuation).split(" ")


def create_features(df):
    # https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur
    # and https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question

    new_data["q1_len"] = df["question1"].apply(lambda question: len(str(question)))
    new_data["q2_len"] = df["question2"].apply(lambda question: len(str(question)))
    # tf-idf
    new_data["word_share"] = df.apply(word_match_share, axis=1)
    # shared words benchmark model
    # ngrams vectorizer()
    # word2vec features cosine distance jaccard distance, other distances?
    return new_data

def tokenize(question):
    """Tokenize english text. The function takes a string as input and return a list of tokens."""
    nlp = spacy.load('en')  # Create english instance
    doc = nlp(unicode(question, 'utf-8'))  # spacy expects unicode strings as input
    tokens = []

    #  Next paragraph taken from tutorial:i https://nicschrading.com/project/Intro-to-NLP-with-spaCy/
    for token in doc:
        tokens.append(token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_)

    # stoplist the tokens
    tokens = [token for token in tokens if token not in STOPLIST]

    # stoplist symbols
    tokens = [token for token in tokens if token not in SYMBOLS]

    return tokens


def vector():
    pass
