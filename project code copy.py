import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

import re
import sklearn

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    FunctionTransformer,
    LabelEncoder,
    TargetEncoder
)

np.random.seed(6954)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
movies = pd.read_csv("movies.csv")

train.drop(labels=['reviewerName'], axis=1, inplace=True)
test.drop(labels=['reviewerName'], axis=1, inplace=True)
movies.drop(labels=['soundType',
                   'distributor',
                   'boxOffice',
                   'originalLanguage',
                   'releaseDateStreaming',
                   'releaseDateTheaters',
                   'ratingContents',
                   'title'], axis=1, inplace=True)

train = train.rename(columns={'isFrequentReviewer':'isTopCritic'})

print(train.dtypes)

def split_genres(x):
    if x is not np.nan:
        return tuple(re.split(", | &",x))
    return tuple()

movies['genre'] = movies.apply({'genre':split_genres})

print(movies['genre'].value_counts())

movies2 = movies.groupby('movieid').agg({
    'audienceScore':np.mean,
    'genre': 'sum',
    'runtimeMinutes':np.max
}).reset_index()

movies2.genre = movies2.genre.apply(lambda x: tuple(i.lower().strip() for i in set(x)))

movies.drop_duplicates(['movieid'],inplace=True)
movies.drop(columns=['audienceScore','genre','runtimeMinutes'],inplace=True)
movies=pd.merge(movies, movies2, on='movieid', how='left')

X_train, y_train = train.loc[:, train.columns != 'sentiment'], train[['sentiment']]

X_train = pd.merge(X_train, movies, on='movieid', how='left')
test = pd.merge(test, movies, on='movieid', how='left')

y_train = LabelEncoder().fit_transform(y_train)

review_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="constant", fill_value="Sample Review Text")),
    ('functionTrans', FunctionTransformer(np.reshape, kw_args={'newshape': -1})),
    ('tfidfVect', TfidfVectorizer(
        strip_accents='unicode',
        stop_words=['english'],
        min_df=5,
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b')
     )
])

preprocessor_review = ColumnTransformer([
    ('review_pipeline',review_pipeline,['reviewText'])
])

class MultiHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, ) -> None:
        self.binarizer_creator = MultiLabelBinarizer
        self.dtype = None

        self.binarizers = []
        self.categories_ = self.classes_ = []
        self.columns = []

    def fit(self, X: pd.DataFrame, y=None):  # noqa
        self.columns = X.columns.to_list()

        for column_name in X:
            binarizer = self.binarizer_creator().fit(X[column_name])
            self.binarizers.append(binarizer)
            self.classes_.append(binarizer.classes_)  # noqa

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if len(self.classes_) != X.shape[1]:
            raise ValueError(f"The fit transformer deals with {len(self.classes_)} columns "
                             f"while the input has {X.shape[1]}.")

        # print(self.classes_)
        return np.concatenate([binarizer.transform(X[c]).astype(self.dtype)
                               for c, binarizer in zip(X, self.binarizers)], axis=1)
    
class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.category_target_means = X.groupby('category')['target'].mean()
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        X_encoded['category'] = X_encoded['category'].map(self.category_target_means)
        return X_encoded
    
    