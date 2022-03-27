from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


def list_vectorizer(X_train, X_test):
    vectorizer = TfidfVectorizer(preprocessor=lambda x: ','.join([el for el in x if type(el)==str]).lower() if type(x) == list else x,
                                 tokenizer=lambda x: x.split(','), max_features=113)
    f = vectorizer.fit_transform(X_train[X_train.notna()])
    top_words = sorted(zip(vectorizer.get_feature_names_out(), f.toarray().sum(axis=0)),
                       key=lambda x: x[1], reverse=1)[:4]
    list_words = [i[0] for i in top_words]
    X_train[X_train.isna()] = X_train[X_train.isna()].apply(lambda x: list_words)
    X_test[X_test.isna()] = X_test[X_test.isna()].apply(lambda x: list_words)
    f_train = vectorizer.fit_transform(X_train)
    f_test = vectorizer.transform(X_test)
    return f_train.toarray(), f_test.toarray()


def get_data(df_train, df_test):
    df_train[df_train == 'unknown'] = np.nan
    df_train[df_train == 'unknown'.upper()] = np.nan
    df_test[df_test == 'unknown'] = np.nan
    df_test[df_test == 'unknown'.upper()] = np.nan

    f1_train, f1_test = list_vectorizer(df_train.genres, df_test.genres)
    f2_train, f2_test = list_vectorizer(df_train.directors, df_test.directors)
    f3_train, f3_test = list_vectorizer(df_train.filming_locations, df_test.filming_locations)
    f4_train, f4_test = list_vectorizer(df_train.keywords, df_test.keywords)

    for i in range(3):
        df_train[f'actor_{i}_gender'].fillna('Male', inplace=True)
        df_train[f'actor_{i}_gender'] = df_train[f'actor_{i}_gender'].apply(lambda x: True if x == 'Male' else False)
        df_test[f'actor_{i}_gender'].fillna('Male', inplace=True)
        df_test[f'actor_{i}_gender'] = df_test[f'actor_{i}_gender'].apply(lambda x: True if x == 'Male' else False)

    df_train.drop(['genres', 'directors', 'filming_locations', 'keywords'], axis=1, inplace=True)
    df_test.drop(['genres', 'directors', 'filming_locations', 'keywords'], axis=1, inplace=True)

    return np.hstack([df_train.to_numpy(), f1_train, f2_train, f3_train, f4_train]), np.hstack([df_test.to_numpy(), f1_test, f2_test, f3_test, f4_test])



def train_model_and_predict(train_file: str, test_file: str) -> np.ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """

    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)

    y_train = df_train["awards"]
    del df_train["awards"]
    df_train, df_test = get_data(df_train, df_test)

    reg_1 = LGBMRegressor(**{'learning_rate': 0.01,
                           'max_depth': 12,
                           'n_estimators': 2000,
                           'random_state': 115}).fit(df_train, y_train)

    reg_2 = LGBMRegressor(**{'learning_rate': 0.01,
                           'max_depth': 12,
                           'n_estimators': 2000,
                           'random_state': 27}).fit(df_train, y_train)


    cat_1 = CatBoostRegressor(**{'learning_rate': 0.04,
                               'max_depth': 8,
                               'n_estimators': 1500},
                                train_dir='/tmp/catboost_info',
                            ).fit(df_train, y_train)

    cat_2 = CatBoostRegressor(**{'learning_rate': 0.01,
                               'max_depth': 6,
                               'n_estimators': 4000},
                            train_dir='/tmp/catboost_info',
                            ).fit(df_train, y_train)

    return (reg_1.predict(df_test) + reg_2.predict(df_test) + 2*cat_1.predict(df_test)+2*cat_2.predict(df_test))/6
