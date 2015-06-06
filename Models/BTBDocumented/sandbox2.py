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


traindata = [
"a large dog",
"a dog house",
"a small cat",
"a small dog",
"a furry cat",
"dog bites cat" ]

testdata = [
    "a furry dog in a house",
    "a dog",
    "a small dog"
]

vid = 1
if vid ==1:
    vectorizer = TfidfVectorizer(
            ngram_range=(1, 1),
            tokenizer = LemmaTokenizer(),
            analyzer="char_wb", 
            binary=False, 
            norm = None,
            stop_words = 'english')
elif vid == 2:
    vectorizer = TfidfVectorizer(
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

print(train_X)
print("-----------")
print(test_X.shape)
print(test_X)
print(vectorizer.get_feature_names())

