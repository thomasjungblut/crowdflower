
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




#if __name__ == '__main__':    

train = pd.read_csv('../../Raw/train.csv')
test  = pd.read_csv('../../Raw/test.csv')

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
traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))

#=====================================================================
vid = 1
if vid == 1:
    vectorizer = TfidfVectorizer(
            tokenizer = LemmaTokenizer(),
            analyzer="char_wb", 
            binary=False, 
            norm = None,
            stop_words = 'english')
elif vid == 2:
    countvect_word = TfidfVectorizer(
            ngram_range=(1, 3),
            tokenizer = LemmaTokenizer(),
            token_pattern=r'\w{1,}',
            analyzer="word", 
            binary=False, 
            norm = None,
            min_df=2,
            stop_words = 'english')
else:
    vectorizer = TfidfVectorizer(
            sublinear_tf=True, 
            max_df=0.5,
            stop_words='english')

vectorizer.fit(traindata)
train_X = vectorizer.transform(traindata)
test_X  = vectorizer.transform(testdata)

print(train_X.shape)


