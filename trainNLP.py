import pandas as pd
import nltk
import math
import numpy as np
import unicodedata
import unidecode
import re
import ast
from nltk.tokenize import RegexpTokenizer,word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.metrics import TrigramAssocMeasures
import argparse
from io import StringIO
import io

'''
#Data prep
df1 = pd.read_csv('Full-Economic-News-DFE-839861-3.csv',encoding = 'iso-8859-1')
df2 = pd.read_csv('us-economic-newspaper-1.csv',encoding = 'iso-8859-1')
df3 = pd.concat([df1.loc[df1['relevance']=='yes'].loc[:,['positivity','relevance','headline','text']],df2.loc[df2['relevance']=='yes'].loc[:,['positivity','releance','headline','text']]])
#print(df3)
'''


print(" 1- Text")
print(" 2 - File")
choose =  input("Please choose:")
choose = int(choose)

if choose == 1:
	df1 = input("Please input more than 2 sentences: ")
	texts1 = nltk.sent_tokenize(df1)
	df3= pd.DataFrame(texts1)
	df3.columns = ['text']
	print(df3)

else:
	texts = input("Input filename:  ")
	#for open and read content
	'''
	lines4 = io.StringIO()   #file like object to store all lines
	with open(texts, 'r', encoding='utf8') as file4:
		lines4.write(file4.read())
		lines4.write('\n')

	lines4.seek(0)        # now you can treat this like a file like object
	df1 = lines4.read()
	texts1 = nltk.sent_tokenize(df1)
	df3= pd.DataFrame(texts1)
	df3.columns = ['text']
	print(df3)
	'''
	df3 = pd.read_csv(texts, encoding = 'iso-8859-1')
	print(df3)



tokenizer = RegexpTokenizer("[\w']+")
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()


def bag_of_words(words):
	return dict([(word,True) for word in words])

def bag_of_ngram_words(words,bscore_fn=BigramAssocMeasures.chi_sq,tscore_fn=TrigramAssocMeasures.chi_sq,n=200):
	bigram_finder = BigramCollocationFinder.from_words(words)
	bigrams = bigram_finder.nbest(bscore_fn,n)
	trigram_finder = TrigramCollocationFinder.from_words(words)
	trigrams = trigram_finder.nbest(tscore_fn,n)	
	return bag_of_words(words+bigrams+trigrams)
	
def bag_of_bigram_words(words,score_fn=BigramAssocMeasures.chi_sq,n=100):
	bigram_finder = BigramCollocationFinder.from_words(words)
	bigrams = bigram_finder.nbest(score_fn,n)
	
	return bag_of_words(words+bigrams)


features = []
for i in range(len(df3['text'].values)):
	features.append(str([word for word in bag_of_bigram_words(tokenizer.tokenize(df3['text'].values[i])) if word not in stopwords.words('english')]))

for i in range(len(df3['text'].values)):
	features[i] = stemmer.stem(lemmatizer.lemmatize(features[i]))


labels3 = df3['label'].fillna(1).values
lab = [1 if v == 'positive' else 0 for v in labels3]
print(lab)


train_feats=[]
test_feats=[]
train_labels=[]
test_labels=[]


#Split value in red
split = 0.75

for i in range(len(features)):
	if i < math.floor(split*len(features)):
		train_feats.append(features[i])
		train_labels.append(labels3)
	else:
		test_feats.append(features[i])
		test_labels.append(labels3)



count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_feats)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, train_labels)
y_pred_class = clf.predict(X_train_tfidf)
#find accuracy
#from sklearn.naive_bayes import MultinomialNB
#nb = MultinomialNB()
#nb.fit(X_train_tfidf, train_labels)
#y_pred_class = clf.predict(X_train_tfidf)\







from sklearn import metrics
accuracy = metrics.accuracy_score(y_pred_class, train_labels)
print("Accuracy: ", round(accuracy*100,2),"%")



'''
import pickle
filename = 'finalized_model1.sav'
pickle.dump(clf, open(filename, 'wb'))
pickle.dump(count_vect, open("vectorizer1.pickle", "wb"))
'''



