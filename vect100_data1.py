import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


dataset=pd.read_csv('ic.csv', sep=';', engine='python',
    na_values=['NA','?'])

SmilesCanonical=dataset.iloc[:,7].values
Observed=dataset.iloc[:,1].values
#print(SmilesCanonical[:20])


print('model de transformation')
modelV = Word2Vec.load("m.model")
print(modelV.corpus_count)
print(modelV)
SmilesCanonical2 = modelV[modelV.wv.vocab]
print(SmilesCanonical2)
print(SmilesCanonical2.shape)


#print('predict------------------------------------------')
#answer_vector = modelV.corpus_total_words
#print(answer_vector )


print('List des molécule  ------------------------------------------------------')

list = []
vec=[]
vec2=[]
i=0


for str in SmilesCanonical:
    print(str)

    for i in range(len(str)):
      #print(i)
      #print(str[i])
      vec = modelV.wv[str[i]]
      #vec=sum(SmilesCanonical2[i])
      vec = vec + vec
      #print(list)

    vec2.append(vec / len(str))
    vec = []

vec2
print('Vecteur final  ------------------------------------------------------------')
print(vec2)
print(len(vec2))
#input = vec2.reshape(356,1)
#print(input)
#print(modelV.predict_output_word(str[0]))
#vec=modelV.predict_output_word(str[0])
