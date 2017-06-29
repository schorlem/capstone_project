"""Collection of useful functions for loading/saving the data as well as reporting on model performance."""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import DMatrix, cv
from sklearn import metrics

from sklearn.metrics import roc_curve, auc


def save_pickle(save_object, output_dir, filename):
    """Function creates the output folder if it does not exist yet and saves the passed object as a pickle"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir+filename, "wb") as handle:
        pickle.dump(save_object, handle)


def load_pickle(input_dir, filename):
    """Load pickle in read only mode"""
    with open(input_dir+filename, "rb") as handle:
        return pickle.load(handle)


def create_roc_plot(val_labels, predictions, name):
    """Helper function that creates a receiver operation characteristic.
    See http://scikit-learn.org for more details."""
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


def xgb_fit_model(alg, x_train, y_train, use_cv=True, cv_folds=5, early_stopping_rounds=50):
    """Fit an XGBoost model and print the results. If use_cv is enabled the function uses cross validation to find the
    best number of iterations. The model is the again fit on the whole dataset.
    Credit: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/"""
    if use_cv:
        xgb_param = alg.get_xgb_params()
        xgtrain = DMatrix(x_train, label=y_train)
        cvresult =cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='logloss', early_stopping_rounds=early_stopping_rounds, stratified=True)
        print("\nBest number of iterations: {}\n".format(cvresult.shape[0]))
        print(cvresult[-10:])
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(x_train, y_train, eval_metric='logloss')

    # Predict training set:
    train_predictions = alg.predict(x_train)
    train_predprob = alg.predict_proba(x_train)[:, 1]

    # Print model report:
    print("\nModel Report on full training set")
    print("Accuracy : {:.4g}".format(metrics.accuracy_score(y_train, train_predictions)))
    print("Logloss (Train): {}".format(metrics.log_loss(y_train, train_predprob)))

    feat_importance = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_importance.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    return plt
    #plt.show()


def report(results, n_top=3):
    """Print fit results. Designed to work with RandomizedSearchCV.
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


