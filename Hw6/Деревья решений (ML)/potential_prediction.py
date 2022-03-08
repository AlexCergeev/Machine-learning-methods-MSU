import os

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline

import numpy as np

class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """

    def fit(self, x, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(x)

    def transform(self, x):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        mass = np.zeros((x.shape[0], 8))
        for i, el in enumerate(x):
            mass[i][0] = el.mean()
            mass[i][1] = np.sum(np.absolute(np.diff(np.trapz(el))))
            mass[i][2] = el.mean() / np.trapz(el).mean(where=np.trapz(el) > 0)
            mass[i][3] = np.absolute(np.diff(el, n=2)).sum()
            mass[i][4] = np.sum(np.absolute(np.diff(np.trapz(el)))) / el.sum() / np.trapz(el).mean(where=np.trapz(el) > 0)
            mass[i][5] = np.trapz(el.mean(axis=1)) / el.sum() * np.trapz(el).mean(where=np.trapz(el) > 0)
            mass[i][6] = np.absolute(np.diff(el, n=2)).sum() / el.sum() / np.trapz(el).mean(where=np.trapz(el) > 0)
            mass[i][7] = np.trapz(el.mean(axis=1)) / el.sum() * np.trapz(el).mean(where=np.trapz(el) > 0) * np.absolute(
                np.diff(el, n=2)).sum()
        return np.array(mass)

def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in os.listdir(data_dir):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)

def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    # it's suggested to modify only the following line of this function
    regressor = Pipeline([('vectorizer', PotentialTransformer()),
                          ('decision_tree', ExtraTreesRegressor(n_estimators=3000,
                                                                max_depth=None,
                                                                random_state=13,
                                                                n_jobs=-1))])
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}


