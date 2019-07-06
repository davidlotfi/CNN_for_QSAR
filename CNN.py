import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from keras.models import Sequential
from keras.layers import *


#load data
dataset=pd.read_csv('ic.csv', sep=';', engine='python',
    na_values=['NA','?'])



SmilesCanonical=dataset.iloc[:,7].values
Observed=dataset.iloc[:,1].values

print('-------------- data avant transformation --------------------------')
print(SmilesCanonical[:20])


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

      #print(str[i])
      vec = modelV.wv[str[i]]
      #vec=sum(SmilesCanonical2[i])
      vec = vec + vec


    vec2.append(vec / len(str))
    vec = []

vec2
print('---------------------data aprés transformation------------------------------')
print('Vecteur final  ------------------------------------------------------------')
print(vec2)
print(len(vec2))




#creation de modèle
#initialising the CNN
model= Sequential()

#Step 1 - Convolution
model.add(Conv2D(kernel_size = (2,1), filters = 1, input_shape=(50,2,1), activation='relu'))

#Step 2 - Pooling : consiste a réduir la taille de " matrice " ou " feature map "
model.add(MaxPooling2D(pool_size = (2,1), strides=(2,1)))
#model.add(Dropout(0.5))

#Step 3 - Flattening : consiste transformer matrice ou feature map a 1 seul vecteur( 100 vecteur >>> 1 vecteur )

model.add(Flatten())

#Step 4 -Connection

model.add(Dense(16, activation='relu'))
model.add(Dense(1))

print(model.summary())

print('compile model')
model.compile(loss='mse', optimizer='adam', metrics=['mse'])# metrics la zam charh

print('fit model in training')

# trés important preparation data et structure
X=(np.array(vec2)).reshape(356,50,2,1)

model.fit(X,Observed, batch_size=len(vec2),epochs=1000)
