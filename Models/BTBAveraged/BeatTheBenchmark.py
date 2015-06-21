
"""
Beating the Benchmark
Search Results Relevance @ Kaggle
__author__ : Abhishek

"""

import pandas as pd
import numpy as np
import math
from scipy import sparse
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import RandomizedSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
#====================================================================


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stm = PorterStemmer()

    def __call__(self, doc):
        # Stem each word in the given document and return this
        # as a list of words.
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa

    quadratic_weighted_kappa calculates the quadratic weighted kappa value,
    which is a measure of inter-rater agreement between two raters that
    provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b each
    correspond to a list of integer ratings.  These lists must have the same
    length.

    The ratings should be integers, and it is assumed that they contain the
    complete range of possible ratings.  quadratic_weighted_kappa(X,
    min_rating, max_rating), where min_rating is the minimum possible rating,
    and max_rating is the maximum possible rating.
    """

    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


def build_stacked_model(dist_colnames):
    countvect_char = TfidfVectorizer(
            tokenizer = LemmaTokenizer(),
            analyzer="char_wb",
            binary=False,
            norm = None,
            stop_words = 'english')

    svd = TruncatedSVD()
    scl = StandardScaler()
    clf = SVC()

    p = Pipeline([
        # Use FeatureUnion to combine the features
        # query and title, and distance metrics
        ('features', FeatureUnion(
            transformer_list=[
                ('cvt', Pipeline([
                    ('selectordoc', ItemSelector(key='doc')),
                    ('vect', countvect_char),
                    ('svd', svd),
                ])),
                ('dist', Pipeline([
                    ('selectordist', ItemSelector(key=dist_colnames)),
                    ('distselect', SelectKBest(chi2))
                ]))
            ]
        )),
        # Use a SVC classifier on the combined features
        ('scl', scl),
        ('clf', clf),
    ])
    return p

def report(grid_scores, n_top=5):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

def combine_factors(x):
    s = "{} {}".format(x['query'], x['product_title'])
    return s

def grid_search_parameters(traindata, y, dist_colnames):
    # Create the pipeline
    clf = build_stacked_model(dist_colnames)

    # Create a parameter grid to search for
    # best parameters for everything in the pipeline
    param_grid = {
            'features__cvt__vect__ngram_range' : [(1, 4), (1, 5), (1, 6)],
            'features__cvt__vect__min_df' : list(range(3, 10, 1)),
            'features__cvt__svd__n_components' : list(range(100, 250, 5)),
            'features__dist__distselect__k' : list(range(10, len(dist_colnames), 5)),
            'clf__degree' : list(range(2, 6, 1)),
            'clf__C' : list(range(4, 10, 1)),
        }

    # Kappa Scorer
    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)

    # Cross validation
    cv = StratifiedKFold(y, n_folds=3, shuffle=True, random_state=42)

    # Initialize Grid Search Model
    # Try many different parameters to find the best fitting model
    model = grid_search.RandomizedSearchCV(
            n_iter=250,  # number of setting to try
            estimator=clf,  # Pipeline
            param_distributions=param_grid,
            scoring=kappa_scorer,
            verbose=10,
            n_jobs=8,  # Number of jobs to run in parallel
            cv=cv,
            iid=True,
            refit=False)

    # Fit Grid Search Model
    print("Fitting training data .")
    model.fit(traindata, y)

    report(model.grid_scores_)

def submit(traindata, y, testdata, idx, dist_colnames):
    params = []
    params.append({
            #0.655 cv score
            'features__cvt__vect__ngram_range' : (1, 6),
            'features__cvt__vect__min_df' : 3,
            'features__cvt__svd__n_components' : 200,
            'features__dist__distselect__k' : 30,
            'clf__degree' : 5,
            'clf__C' : 9,
        })

    predictions = []
    for param in params:
        print("Fitting: {0}".format(params))
        clf = build_stacked_model(dist_colnames)
        clf.set_params(**param)
        clf.fit(traindata, y)
        predictions.append(clf.predict(testdata))

    # add rudis predictions here too
    rudisPreds = pd.read_csv('../srkruger1_61897_submit.csv')
    predictions.append(rudisPreds['prediction'].tolist())

    print("Creating submission")
    num_examples = len(testdata)
    num_models = len(predictions)
    preds = []
    for i in range(num_examples):
        sum = 0
        for j in range(num_models):
            sum += predictions[j][i]
        preds.append(int(math.floor(sum / num_models)))

    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv("Submission/sub.csv", index=False)
    print("done")

if __name__ == '__main__':
    # Load the training file
    print("Loading data.")
    train = pd.read_csv('../../Raw/train.csv')
    train_dist = pd.read_csv('../../Processed/distances_features.csv')
    test = pd.read_csv('../../Raw/test.csv')
    test_dist = pd.read_csv('../../Processed/distances_features_test.csv')

    print("Preprocessing")
    # Make of copy of the query ids so we
    # can build a submission later on.
    idx = test.id.values.astype(int)
    # we dont need ID columns
    train = train.drop('id', axis=1)
    train_dist = train_dist.drop('id', axis=1)
    test  = test.drop('id', axis=1)
    test_dist = test_dist.drop('id', axis=1)

    # create labels. drop useless columns
    y = train.median_relevance.values
    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)
    train_dist = train_dist.drop(['outcome'], axis=1)

    dist_colnames = list(train_dist.columns.values)

    trainX = pd.DataFrame({
                'doc':list(train.apply(combine_factors, axis=1)),
             })
    trainX = pd.concat([trainX, train_dist], axis = 1, join_axes=[trainX.index])
    testX =  pd.DataFrame({
                'doc':list(test.apply(combine_factors, axis=1))
             })
    testX = pd.concat([testX, test_dist], axis = 1, join_axes=[testX.index])

    #grid_search_parameters(trainX, y, dist_colnames)

    submit(trainX, y, testX, idx, dist_colnames)

