import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict

def train_svm_and_predict(X, y, test_features):
    """
    train_features: np.array, (num_elements_train x num_features) - train data description, the same features and the same order as in train data
    train_target: np.array, (num_elements_train) - train data target
    test_features: np.array, (num_elements_test x num_features) -- some test data, features are in the same order as train features

    return: np.array, (num_elements_test) - test data predicted target, 1d array
    """

    X, y, test_features = X[:, [0, 3, 4]], y, test_features[:, [0, 3, 4]]

    scr_c = []
    for i in np.logspace(-1, 3, 80):
        clf = SVC(C=i, kernel='rbf', class_weight='balanced')
        scr_c.append((cross_val_score(clf, X, y, cv=5).mean(), i))
        if len(scr_c) > 15 and min(scr_c[-15:-1]) > scr_c[-1]: break
    model = SVC(C=max(scr_c)[1], kernel='rbf', class_weight='balanced')
    model.fit(X, y)

    return model.predict(test_features)
