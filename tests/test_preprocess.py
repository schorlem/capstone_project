import capstone_project.preprocessor as pre


def test_tokenize():
    test_strings = [(u"This is a test.",
                     [u"test"]),
                    (u"Are there any positive benefits of laziness and procrastination?",
                     [u"positive", u"benefit", u"laziness", u"procrastination"]),
                    (u"A question for testing/trying.",
                     [u"question", u"testing/try"])]

    for test_string, result in test_strings:
        output = pre.tokenize(test_string)
        assert output == result


def test_tfidftransformer():
    pass


def test_featuretransformer():
    pass
