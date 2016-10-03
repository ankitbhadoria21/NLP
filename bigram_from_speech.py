#!/usr/bin
import re,nltk


def clean(filename):
	global lt
	LB=[lines.rstrip() for lines in open(filename,"r").readlines() 
		if lines!="\n" and lines.startswith("Source:") == False] 

	LB = [re.sub(r'[^\w ]','',lines) for lines in LB]
	LB=[lines.split() for lines in LB if not lines.isdigit()]
	lt=LB
	LB=[item.lower() for sub in LB for item in sub if not item.isdigit()]
	return LB

LB_train=clean("gettys.txt")

from nltk.model import build_vocabulary
vocab = build_vocabulary(2, LB_train)
from nltk.model import count_ngrams
bigram_counts = count_ngrams(2, vocab,lt)


from nltk.model import LaplaceNgramModel
LB_Model=LaplaceNgramModel(bigram_counts)

LB_test=clean("test.txt")

MB_train=clean("test2.txt")

vocab1=build_vocabulary(2, MB_train)

bigram_counts1 = count_ngrams(2, vocab1,lt)

MB_Model=LaplaceNgramModel(bigram_counts1)

MB_test=clean("test3.txt")
print(LB_Model.perplexity(LB_test))
print(MB_Model.perplexity(MB_test))

print(LB_Model.perplexity(LB_train))
print(MB_Model.perplexity(MB_train))

print(MB_Model.perplexity(LB_train))
print(LB_Model.perplexity(MB_train))
