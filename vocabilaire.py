
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import *
import pandas as pd


dataset=pd.read_csv('IGC50.csv', sep=';', engine='python',
    na_values=['NA','?'])
SmilesCanonical=dataset.iloc[:,0].values

print(SmilesCanonical[0])

model_1 = Word2Vec(size=100, min_count=1)

model_1.build_vocab(SmilesCanonical)

total_examples = model_1.corpus_count
print(total_examples)
model_1.save("m.model2")
model = Word2Vec.load("m.model2")

SmilesCanonical2 = model[model.wv.vocab]
print(SmilesCanonical2)



