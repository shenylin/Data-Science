# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 14:48:32 2022

@author: reneeh2
"""

import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('stopwords')
from wordcloud import WordCloud
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

sentence = """Humpty Dumpty sat on a wall. 
Humpty Dumpty had a great fall. 
All the king’s horses and all the king’s men could not put Humpty together again.""" 

wordcloud = WordCloud(collocations = False, background_color = 'white').generate(sentence)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

tokens = nltk.word_tokenize(sentence)
print(tokens)

fdist = FreqDist(tokens)
print(fdist.most_common(10))
fdist.plot()

tagged = nltk.pos_tag(tokens)
#print (tagged[0:6])

entities = nltk.chunk.ne_chunk(tagged)
print(entities) 

tokens_no_punc = []
for t in tokens:
    if t.isalpha():
        tokens_no_punc.append(t.lower())
        #print(tokens_no_punc)

fdist = FreqDist(tokens_no_punc)
print(fdist.most_common(10))
fdist.plot()

stopwords = stopwords.words("english")
tokens_clean = []
for t in tokens_no_punc:
    if t not in stopwords:
        tokens_clean.append(t)
        #print(tokens_clean)
        #print (stopwords)
        
fdist = FreqDist(tokens_clean)
print(fdist.most_common(10))
fdist.plot()    