
import pandas as pd
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer


#====================================================================
class LemmaTokenizer(object):

    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stm = PorterStemmer()

    def __call__(self, doc):
        # Stem each word in the given document and return this
        # as a list of words.
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]



def combine_factors(x):
    z = x['product_description']
    if pd.isnull(z):
        z = ""
    else:
        z = z[:80]
    s = "{} {} {}".format(x['query'], x['product_title'], z)
    return s


#if __name__ == '__main__':    

train = pd.read_csv('../../Raw/train.csv')
test  = pd.read_csv('../../Raw/test.csv')

train_mmr = pd.read_csv('../../Processed/train_minmaxr.csv')
test_mmr  = pd.read_csv('../../Processed/test_minmaxr.csv')


# Make of copy of the query ids so we
# can build a submission later on.
idx = test.id.values.astype(int)
# we dont need ID columns anymore
train = train.drop('id', axis=1)
test  =  test.drop('id', axis=1)

print(idx)

# create labels.
y = train.median_relevance.values
# Now we can drop them from the training set
# and we can drop the related info too.
train = train.drop(['median_relevance', 'relevance_variance'], axis=1)
print(y)

# concat query and product_title
traindata = list(train.apply(combine_factors,axis=1))

print(traindata[1])

print(train_mmr)

