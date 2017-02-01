import re
import os
import string
import numpy as np
import nltk
from os import listdir
import itertools
from collections import defaultdict
import boto.s3.connection
from nltk import PorterStemmer
from pandas import DataFrame
from cStringIO import StringIO
from os.path import isfile, join
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TreebankWordTokenizer
from pdfminer.converter import TextConverter
from nltk.stem.wordnet import WordNetLemmatizer


from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
pdfdocs={}
doc={}

def remove_punctuation(s):
	table = string.maketrans("","")
	return s.translate(table, string.punctuation)

def tokenize(text):
    text = remove_punctuation(text)
    text = text.lower()
    texts=re.split("\W+", text)
    lmtzr = WordNetLemmatizer()
    x=nltk.pos_tag([lmtzr.lemmatize(i) for i in texts])
    return x

def count_words(words):
    wc = {}
    for word in words:
        wc[word] = wc.get(word, 0.0) + 1.0
    return wc

def gettags(x):
	
	st = PorterStemmer()
	x= unicode(x, errors='ignore')
	x=x.lower()
	vectorizer = CountVectorizer(ngram_range=(1,1), stop_words=None, tokenizer=TreebankWordTokenizer().tokenize)
	#^([a-zA-Z]*|\d+|\W)$
	tokenize = vectorizer.build_tokenizer()
	tokenList = tokenize(x)
	tokenList = [token for token in tokenList if re.match('[a-zA-Z]+', token)]
	lmtzr = WordNetLemmatizer()
	#x=[ for i in tokenList]
	#print [i for i in tokenList]
	tags=nltk.pos_tag([lmtzr.lemmatize(i) for i in tokenList])
	return tags

def pdfreader(doc,pages=None):
	if not pages:
		pagenums = set()
	else:
		pagenums = set(pages)
	output = StringIO()
	manager = PDFResourceManager()
	converter = TextConverter(manager, output, laparams=LAParams())
	interpreter = PDFPageInterpreter(manager, converter)
	infile = file(doc, 'rb')
	for page in PDFPage.get_pages(infile, pagenums):
		interpreter.process_page(page)
	infile.close()
	converter.close()
	text = output.getvalue()
	output.close
	return text

def collectivetags(x):
    x=list(x)
    if x[0].startswith('N') :
        x[0]='NN'
        x=tuple(x)
        return x
    elif x[0].startswith('VB') :
        x[0]='VB'
        x=tuple(x)
        return x
    elif not x[0] is "''":
        return x

def __setitem__(self, key, value):
	if key in self:
		super().__setitem__(key, self[key] + value)
	else:
		super().__setitem__(key, value)

def SparkConnection():
	conf = SparkConf().setAppName("Words count").setMaster("local")
	sc = SparkContext(conf=conf)
	return sc

def build_data_frame(path, classification):
    rows = []
    index = []
    with open(path,'r') as files:
        rows.append({'text': files.read(), 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame

if __name__ == '__main__':
	textfile="/Users/radhikamanivannan/Desktop/1850/"
	pdfs_inpath = [f for f in listdir(textfile) if isfile(join(textfile, f))]
	
	# no_extn_file = [f.split(".")[0] for f in pdfs_inpath]

	# for pdffile in no_extn_file:
	# 	if not os.path.exists(textfile+pdffile+".txt") and pdffile!="":
	# 		converted_textfile=pdfreader(textfile+pdffile+".pdf")
	# 		with open(textfile+pdffile+".txt" , "w") as f:
	# 			f.write(converted_textfile)
	for pdffile in pdfs_inpath:
		if isfile(join(textfile, pdffile)):
			f = open(textfile+pdffile, 'r')
			datainfile=f.read()
			tags=gettags(datainfile)
			pdfdocs[pdffile]=tags
			
	print pdfdocs
	for i in pdfdocs:
		pdfdocs[i] =  dict((x,y) for x, y in pdfdocs[i])
	for i in pdfdocs:
		pdfdocs[i] = pdfdocs[i].values()
		pdfdocs[i] = Counter(pdfdocs[i])
	print pdfdocs

	val={}

	test = "Hello to this radhika to my name and yes to to"
	test=gettags(test)
	
	val=[x[1] for x in test]
	print Counter(val)


