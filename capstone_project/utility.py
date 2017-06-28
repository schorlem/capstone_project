import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc



def report(results, n_top=3):
    """Print fit results. Designed to work with random search results.
    Credit: http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results["mean_test_score"][candidate],
                  results["std_test_score"][candidate]))
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


def create_roc_plot(val_labels, predictions, name):
    """Helper function that creates a receiver operation characteristic."""
    fpr, tpr, thresholds = roc_curve(val_labels, predictions)
    roc_auc = auc(fpr, tpr)

    lw = 2
    plt.plot([0, 1], [0, 1], linestyle="--", lw=lw, color="k", label="Luck")
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic for {}".format(name))
    plt.legend(loc="lower right")
    return plt
