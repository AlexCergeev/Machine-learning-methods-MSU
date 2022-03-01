import numpy as np


class Preprocesser:
    
    def __init__(self):
        pass
    
    def fit(self, X, Y=None):
        pass
    
    def transform(self, X):
        pass
    
    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocesser):

    def __init__(self, dtype=np.float64):
        super(Preprocesser).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        # your code here

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        df = np.array(X).T
        flag = True
        res = np.array(0)
        for col in df:
            unique_elements = np.unique(np.array(col))
            unique_elements.sort()
            array = np.zeros((col.shape[0], len(unique_elements)))
            for i, el in enumerate(col):
                array[i][np.where(unique_elements == el)[0][0]] = 1
            if flag:
                res = array
                flag = False
            else:
                res = np.hstack((res, array))
        return res

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        self.X = np.array(X).T
        self.Y = np.array(Y).T

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        df = np.array(X).T
        flag = True
        res = np.array(0)
        for col in df:
            array = np.zeros((col.shape[0], 3), dtype=self.dtype)
            for i, el in enumerate(col):
                array[i][0] = np.sum((col == el) * self.Y) / np.sum(col == el)  # successes
                array[i][1] = np.sum(col == el) / col.shape[0]  # counters
                array[i][2] = (array[i][0] + a) / (array[i][1] + b)  # relation
            if flag:
                res = array
                flag = False

            else:
                res = np.hstack((res, array))
        return (res)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        self.X = np.array(X).T
        self.Y = np.array(Y).T
        self.seed = seed
        self.group_k_fold = list(group_k_fold(X.shape[0], n_splits=self.n_folds, seed=self.seed))

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        self.X = np.array(X).T
        v = self.X
        flag_g = True
        answer = np.array(0)
        for col in v:
            flag = True
            res = np.array(0)
            for i, j in self.group_k_fold:
                array = np.zeros((col.shape[0], 3), dtype=self.dtype)
                for k, el in zip(i, col[i]):
                    array[k][0] = np.sum((col[j] == el) * self.Y[j]) / np.sum(col[j] == el)
                    array[k][1] = np.sum(col[j] == el) / col[j].shape[0]
                    array[k][2] = (array[k][0] + a) / (array[k][1] + b)
                if flag:
                    res = array
                    flag = False
                else:
                    res = res + array

            if flag_g:
                answer = res
                flag_g = False
            else:
                answer = np.hstack((answer, res))
        return answer

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def logloss(x_onehot, y, w):
    summa = 0
    for xi, yi in zip(x_onehot, y):
        p = np.sum(xi * w)
        f = yi * np.log(p) + (np.array(1) - yi) * np.log(np.array(1) - p)
        if np.isnan(f): continue
        summa += f
    return -summa


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    enc = MyOneHotEncoder(dtype=int)
    enc.fit(x)
    x_onehot = enc.transform(np.array([[i] for i in x]))
    weght = np.array([0.5 for i in range(x_onehot.shape[1])], dtype=np.float64)
    score_min = logloss(x_onehot, y, weght)
    step = 0.5
    k = 0
    while k < 20:
        for i, w in enumerate(weght):
            weght_new = np.array(weght)
            # print(weght)
            for x in np.linspace(w - step, w + step, 6):
                if x < 0 or x > 1: continue
                weght_new[i] = x
                # print(weght_new)
                score = logloss(x_onehot, y, weght_new)
                if score < score_min:
                    # print(score)
                    weght[i] = x
                    score_min = score
        step /= 2
        k += 1
    return weght
