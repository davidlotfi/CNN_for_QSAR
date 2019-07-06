# CNN_for_QSAR
Application de l’apprentissage en profondeur dans la modélisation QSAR pour la prédiction de l’activité biologique 

# installation
* Anaconda ( https://docs.conda.io/en/latest/miniconda.html  )
* Tensorflow ( https://www.tensorflow.org  )
* Kearas  ( https://www.keras.io  )
* IDE python (pycharm) (https://www.pycharm.org  )

# Datasets
Les données utilisées pour l'évaluation du modèle peuvent être téléchargées à partir de ce projet, 
(ic.csv) pour 356 elemant.
## Préparer les données pour la formation
dataset=pd.read_csv('ic.csv', sep=';', engine='python',
    na_values=['NA','?'])
# Mettre le modele a l'entrainement
X=(np.array(vec2)).reshape(356,50,2,1)

model.fit(X,Observed, batch_size=len(vec2),epochs=1000)



