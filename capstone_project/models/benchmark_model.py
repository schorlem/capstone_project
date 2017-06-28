""" This module implements the calcultation of out benchmark model. The benchmark model
The code for the benchmark model is based on this kaggle kernel:
https://www.kaggle.com/cgrimal/quora-question-pairs/words-in-common-benchmark/code
"""
from nltk.corpus import stopwords

STOPS = set(stopwords.words("english"))


def word_match_share(row):
    """Takes a row of a dataframe and returns the matched word share ratio. The input row needs to have two columns
    named question1 and question 2."""
    q1words = {}
    q2words = {}
    for word in str(row["question1"]).lower().split():
        if word not in STOPS:
            q1words[word] = 1
    for word in str(row["question2"]).lower().split():
        if word not in STOPS:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words = [w for w in q1words.keys() if w in q2words]

    return 0.5 * len(shared_words) * (1./len(q1words) + 1./len(q2words))
