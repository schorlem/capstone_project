"""Collections of functions that are used for preprocessing the Quora data"""
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df, r):
    """Split the dataset and return two dataframes on for training and one for testing.
    Only the two questons and the labels are kept."""
    # The first three columns are all IDs and are not needed, the last column is the label
    x, y = df.iloc[:, 3:-1].values, df.iloc[:, -1].values
    # Stratified splitting is enable because the data is somewhat imbalanced(37% are duplicates)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=r, stratify=y, random_state=42)
    return x_train, x_test, y_train, y_testt
