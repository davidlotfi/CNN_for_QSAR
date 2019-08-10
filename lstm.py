import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn import metrics



datasett=pd.read_csv('IGC50.csv', sep=';', engine='python',
    na_values=['NA','?'])
dataset=pd.read_csv('result', delimiter=',', engine='python',
    na_values=['NA','?'])
vec2=dataset.values
Observed=dataset.iloc[:,1].values



def coeff_determination(y_true, y_pred):  #calculer la precision de modele
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


model= Sequential()
#initialisation de RNN
model.add(LSTM(units=4, input_shape=(None,1),activation='relu')) # Hidden 1
#model.add(Dense(10, activation='relu')) # Hidden 2
model.add(Dense(1)) 

model.compile(loss='mse',optimizer='adam', metrics=[coeff_determination])

X=np.array(vec2).reshape(len(vec2),100,1)
print(vec2[0])
print(X[0])
print('training and evaluation --------------------')


# split into train and test
x_train, x_test, y_train, y_test = train_test_split(
    X, Observed, test_size=0.25, random_state=42)

print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))

model.fit(x_train,y_train, batch_size=len(vec2),epochs=1000, validation_data=(x_test,y_test))


# Prediction
y_pred=model.predict(x_test)
print(y_pred)
score = metrics.mean_squared_error(y_pred,y_test)
print("Final score (MSE): {}".format(score))
plt.plot(y_pred,y_test,label='linear')
plt.scatter(y_pred,y_test)
plt.show()




