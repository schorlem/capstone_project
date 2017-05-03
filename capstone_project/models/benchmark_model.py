#https://www.kaggle.com/cgrimal/quora-question-pairs/words-in-common-benchmark/code

from nltk.corpus import stopwords
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row["question1"]).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row["question2"]).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    shared_words = [w for w in q1words.keys() if w in q2words]
    print q1words, q2words

    #print shared_words_in_q1, " ", shared_words_in_q2
    #print len(shared_words_in_q1), " ", len(shared_words_in_q2)
    #return (0.5*len(shared_words_in_q1)/len(q1words) + 0.5*len(shared_words_in_q2)/len(q2words))
    return (0.5*len(shared_words)/len(q1words) + 0.5*len(shared_words)/len(q2words))
    return (0.5*len(shared_words)* (1/len(q1words) + 1/len(q2words)))


data = pd.read_csv("/home/andre/Documents/mooc/udacity/machinelearning/capstone_project/data/questions.csv")
labels = data["is_duplicate"]
data.drop("is_duplicate", inplace=True, axis=1)
train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.2, random_state=444)
train = pd.concat([train_x, train_y], axis=1)
test = test_x
#test = pd.read_csv("../input/test.csv")
stops = set(stopwords.words("english"))

sub = pd.DataFrame()
#sub['test_id'] = test['test_id']
#sub["is_duplicate"] = test.apply(word_match_share, axis=1, raw=True)
#sub.to_csv("count_words_benchmark.csv", index=False)
test_pred = test.apply(word_match_share, axis=1, raw=True)

#print test_pred

#print test_pred, type(test_pred)
print log_loss(test_y, test_pred)

