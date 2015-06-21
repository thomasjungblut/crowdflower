"""
Based on Beating the Benchmark by Abhishek

"""

import pandas as pd
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import decomposition, metrics, grid_search
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.cross_validation import StratifiedKFold
import numpy as np

#====================================================================


# From: http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to sklearn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of
    `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
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


class IdentityTransform(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, values):
        # Needs to be a numpy array for later operations like scaling
        return np.array([[v] for v in values], dtype=float)


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

    numerator   = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

def combine_factors(x):
    s = "{} {}".format(x['query'], x['product_title'])
    return s


def build_pipe_line():
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
        # query and title, and rating range
        ('features', FeatureUnion(
            transformer_list=[
                ('cvt', Pipeline([
                    ('selector', ItemSelector(key='doc')),
                    ('vect', countvect_char),
                    ('svd', svd),
                    ('scl', scl)
                ])),
                ('minr', Pipeline([
                    ('selector', ItemSelector(key='maxr')),
                    ('transf', IdentityTransform()),
                    ('scl', scl)
                ])),
                ('maxr', Pipeline([
                    ('selector', ItemSelector(key='maxr')),
                    ('transf', IdentityTransform()),
                    ('scl', scl)
                ])),
            ],
            # weight components in FeatureUnion
            transformer_weights={
                'doc':  1.0,
                'minr': 0.1,
                'maxr': 0.1
            },
        )),
        # Use a SVC classifier on the combined features
        ('clf', clf),
    ])
    return p



if __name__ == '__main__':
    train     = pd.read_csv('../../Raw/train.csv')
    test      = pd.read_csv('../../Raw/test.csv')
    train_mmr = pd.read_csv('../../Processed/train_minmaxr.csv')
    test_mmr  = pd.read_csv('../../Processed/test_minmaxr.csv')

    # Make of copy of the query ids so we
    # can build a submission later on.
    idx = test.id.values.astype(int)
    # we dont need ID columns anymore
    train = train.drop('id', axis=1)
    test  =  test.drop('id', axis=1)

    # create labels.
    y = train.median_relevance.values
    # Now we can drop them from the training set
    # and we can drop the related info too.
    train = train.drop(['median_relevance', 
                        'relevance_variance'], axis=1)

    trainX = pd.DataFrame({'doc':list(train.apply(combine_factors,axis=1)),
                           'minr':train_mmr['minr'],
                           'maxr':train_mmr['maxr']
                          })
    testX = pd.DataFrame({'doc':list(test.apply(combine_factors, axis=1)),
                          'minr':test_mmr['minr'],
                          'maxr':test_mmr['maxr']
                         })
    clf = build_pipe_line()

    print("pipeline:", [name for name, _ in clf.steps])

    if True:
        # Create a parameter grid to search for 
        # best parameters for everything in the pipeline
        if False:
            param_grid = {
                    'vect__ngram_range' : [(2,7), (1, 6), (2,6), (3,6), (2,5) ],
                    'vect__min_df' : list(range(3, 12, 1)), #[6]
                    'svd__n_components' : list(range(100, 350, 5)),  #220
                    'clf__degree' : list(range(1, 10, 1)),  #4
                    'clf__C' : list(range(5, 10, 1)),
            }
        else:
            param_grid = {
                    'features__cvt__vect__ngram_range' : [(1, 6)],
                    'features__cvt__vect__min_df' :  [3],
                    'features__cvt__svd__n_components' : list(range(120, 300, 2)),  #220
                    'features__cvt__svd__n_iter' : [2,4,8],
                    'clf__degree' : [5],
                    'clf__C' : [9]
            }

        # Kappa Scorer 
        kappa_scorer = metrics.make_scorer(
                quadratic_weighted_kappa, greater_is_better = True)

        # Cross validation
        cv = StratifiedKFold(y, n_folds=3, shuffle=True, random_state=42)

        # Initialize Grid Search Model
        # Try many different parameters to find the best fitting model
        model = grid_search.RandomizedSearchCV(
                n_iter=60,  # number of setting to try
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
        model.fit(trainX, y)

        print("Best score: %0.3f" % model.best_score_)
        print("Best parameters set:")
        best_parameters = model.best_estimator_.get_params()
        for param_name in sorted(param_grid.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        # Get best model
        best_model = model.best_estimator_
        # Fit model with best parameters optimized for 
        # quadratic_weighted_kappa
        best_model.fit(trainX, y)
        preds = best_model.predict(testX)

        print("Creating submission")
        # Create your first submission file
        submission = pd.DataFrame({"id": idx, "prediction": preds})
        submission.to_csv("Submission/sub.csv", index=False)
        print("done")


