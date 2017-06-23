import pandas as pd
import capstone_project.preprocessor as pre
import numpy as np

def test_tokenize():
    test_strings = [("This is a test.",
                     np.array([u"test"])),
                    ("Are there any positive benefits of laziness and procrastination?",
                     np.array([u"positive", u"benefit", u"laziness", u"procrastination"])),
                    ("A question for testing/trying.",
                     np.array([u"question", u"testing/try"]))]

    for test_string, result in test_strings:
        output = pre.tokenize(test_string)
        assert np.equal(output, result)


def test_tokenize_unicode():
    test_strings = [(u"This is a test.",
                     np.array([u"test"])),
                    (u"Are there any positive benefits of laziness and procrastination?",
                     np.array([u"positive", u"benefit", u"laziness", u"procrastination"])),
                    (u"A question for testing/trying.",
                     np.array([u"question", u"testing/try"]))]

    for test_string, result in test_strings:
        output = pre.tokenize(test_string)
        assert np.equal(output, result)


def test_tfidftransformer():
    df = pd.DataFrame([["This is a test.", "Words are needed here."],
                      ["One two the owl is here.", "Why are you so smart"]],
                      columns=["question1", "question2"])

    tfidf = pre.TfidfTransformer()
    tfidf.fit(df)
    output = tfidf.transform(df)
    assert np.all(output.toarray() >= 0)
    assert np.all(output.toarray() <= 1)
    assert output.dtype == float
    assert df.shape[0] == output.shape[0]
    assert output.shape[1] % 2 == 0


def test_featuretransformer():
    df = pd.DataFrame([["This is a test.", "Words are needed here."],
                       ["One two the owl is here.", "Why are you so smart"]],
                      columns=["question1", "question2"])
    check_df = df.copy()  # Copy df in order to test that the input is not changed
    transformer = pre.FeatureTransformer()
    output = transformer.transform(df)
    assert check_df.equals(df)
    assert isinstance(output, pd.DataFrame)
    for dtype in output.dtypes:
        print dtype
        assert(dtype in [int, float])


#TODO test that no nan values are present in both feature transformers!!!
#print test[0], len(test[0])
#print np.any(np.isnan(test))
#print np.any(np.isnan(test), axis=0)
#print np.where(np.any(np.isnan(test),axis=1))
#print test[367]

#TODO make sure that no inf is present
#rint np.any(np.isinf(test), axis=0)
#print np.where(np.any(np.isinf(test),axis=1))
#pint test[1161]
#import sys
#sys.float_info.max

#TODO test get names check only if array of strings is returned


#test in train lstm that concatenate doubles length of all tokens
#print len(tokens_1), len(tokens_2), len(all_tokens)
#print tokens_1[-1]
#print all_tokens[-1]
#