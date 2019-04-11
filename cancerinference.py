import pandas as pd
import nltk
import math
import numpy as np
import unicodedata
import unidecode
import re
import ast
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer,word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import pickle
import os
import io
import glob
from pandas.compat import StringIO
import sys

filename = 'finalized_model_cancer1.sav'
clf = pickle.load(open(filename, 'rb'))


print(" 1- File")
print(" 2 - Text")
choose =  input("Please choose:")
choose = int(choose)

if choose == 1:
	direct = os.listdir()
	print(direct)
	texts = input("Filename:  ")
	if '.csv' in texts:
		df3 = pd.read_csv(texts, encoding = 'iso-8859-1')
		df4 = pd.read_csv(texts, encoding = 'iso-8859-1')
		del df4['label']
		print(df3)
		

	else :
		
		fileDir = os.path.dirname(os.path.realpath('__file__'))
		path1 = os.path.join(fileDir, texts)
		path = os.listdir(path1)
		allFiles = glob.glob(path1 + "/*.csv")
		frame = pd.DataFrame()
		list_ = []

		for file_ in allFiles:
			df = pd.read_csv(file_,index_col=None, header=0)
			list_.append(df)
		df3 = pd.concat(list_)
		df4 = pd.concat(list_)
		del df4['label']
		print(df3)


else:
	texts = input("Please enter your text: ")
	df1 = nltk.sent_tokenize(texts)
	df2 = nltk.sent_tokenize(texts)	
	df3 = pd.DataFrame(df1)
	df4 = pd.DataFrame(df2)
	df3.columns = ['text']
	df4.columns = ['text']
	print(df3)

	

tokenizer = RegexpTokenizer("[\w']+")
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()


def bag_of_words(words):
	return dict([(word,True) for word in words])
	
def bag_of_bigram_words(words,score_fn=BigramAssocMeasures.chi_sq,n=100):
	bigram_finder = BigramCollocationFinder.from_words(words)
	bigrams = bigram_finder.nbest(score_fn,n)
	
	return bag_of_words(words+bigrams)
	
nodiacriticsfeats = []

for i in range(len(df3['text'].values)):
	for j in range(len(tokenizer.tokenize(df3['text'].values[i]))):
		if all(ord(char) < 128 for char in tokenizer.tokenize(df3['text'].values[i])[j]) == False:
			nodiacriticsfeats.extend(tokenizer.tokenize(df3['text'].values[i].replace(tokenizer.tokenize(df3['text'].values[i])[j],'and')))
			df3['text'].values[i] = str(nodiacriticsfeats)
			nodiacriticsfeats = []

	
features = []
	
for i in range(len(df3['text'].values)):
	features.append(str([word for word in bag_of_bigram_words(tokenizer.tokenize(df3['text'].values[i])) if word not in stopwords.words('english')]))

for i in range(len(features)):
	for j in range(len(tokenizer.tokenize(features[i]))):
		tokenizer.tokenize(features[i])[j] = stemmer.stem(lemmatizer.lemmatize((tokenizer.tokenize(features[i])[j])))
		
count_vect = pickle.load(open("vectorizer_cancer1.pickle","rb"))

m = 0
n = 0

for i in range(len(df4.values)):
	if clf.predict(count_vect.transform([features[i]])) == 0:
		for j in range(len(df4.values[i])):
			print("NEGATIVE",df4.values[i][j])
			n = n + 1

			
	elif clf.predict(count_vect.transform([features[i]])) == 1:
		for j in range(len(df4.values[i])):
			print("POSITIVE",df4.values[i][j])
			m = m + 1


t = len(df4)
neg = round(((n/t) * 100),2)
pos = round(((m/t) * 100),2)
	
print("Positivity: ", pos, "%")
print("Negativity: ", neg, "%")



