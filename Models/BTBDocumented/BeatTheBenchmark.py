
"""
Beating the Benchmark 
Search Results Relevance @ Kaggle
__author__ : Abhishek

"""

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import RandomizedSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

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


def build_stacked_model():
    countvect_char = TfidfVectorizer(
            tokenizer = LemmaTokenizer(),
            analyzer="char_wb", 
            binary=False, 
            norm = None,
            stop_words = 'english')

    svd = TruncatedSVD()
    scl = StandardScaler()
    clf = SVC()

    p = pipeline.Pipeline([
        ('vect', countvect_char),
        ('svd', svd),
        ('scl', scl),
        ('clf', clf)]
    )

    return p


if __name__ == '__main__':
    # Load the training file
    print("Loading data.")
    if 1:
        train = pd.read_csv('../../Raw/train.csv')
        test  = pd.read_csv('../../Raw/test.csv')
    else:
        train = pd.read_csv('../../Processed/train_scrubbed.csv')
        test  = pd.read_csv('../../Processed/test_scrubbed.csv')

    print("Preprocessing")
    # Make of copy of the query ids so we
    # can build a submission later on.
    idx = test.id.values.astype(int)
    # we dont need ID columns
    train = train.drop('id', axis=1)
    test  = test.drop('id', axis=1)

    # create labels. drop useless columns
    y = train.median_relevance.values
    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

    # do some lambda magic on text columns
    # Combine query and product_title into one string
    traindata = list(train.apply(lambda x:'%s %s' % 
        (x['query'],x['product_title']),axis=1))
    testdata = list(test.apply(lambda x:'%s %s' % 
        (x['query'],x['product_title']),axis=1))

    #traindata = list(train.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'], x['product_description']),axis=1))
    #testdata = list(test.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'], x['product_description']),axis=1))

    # Create the pipeline 
    clf = build_stacked_model()

    # Create a parameter grid to search for 
    # best parameters for everything in the pipeline
    param_grid = {
            'vect__ngram_range' : [(1, 5)],
            'vect__min_df' : [6], # list(range(1, 20, 10)),
            'svd__n_components' : [220], # list(range(100, 700, 500)),
            'clf__degree' :[4], # list(range(2, 8, 4)),
            'clf__C' : [6], # list(range(5, 10, 5)),
            }

    # Kappa Scorer 
    kappa_scorer = metrics.make_scorer(
            quadratic_weighted_kappa, greater_is_better = True)

    # Cross validation
    cv = StratifiedKFold(y, n_folds=2, shuffle=True, random_state=42)

    # Initialize Grid Search Model
    # Try many different parameters to find the best fitting model
    model = grid_search.RandomizedSearchCV(
            n_iter=1,  # number of setting to try
            estimator=clf,  # Pipeline
            param_distributions=param_grid,
            scoring=kappa_scorer,
            verbose=10,
            n_jobs=1,  # Number of jobs to run in parallel
            cv=cv,
            iid=True,
            refit=True)


    # Fit Grid Search Model
    print("Fitting training data .")
    model.fit(traindata, y)
    exit(0)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # Get best model
    best_model = model.best_estimator_
    # Fit model with best parameters optimized for quadratic_weighted_kappa
    best_model.fit(traindata, y)
    preds = best_model.predict(testdata)

    print("Creating submission")
    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv("Submission/sub.csv", index=False)
    print("done")

