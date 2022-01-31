from flask import Flask,render_template,request , jsonify

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
import pickle

#Load Data
dataframe = pd.read_csv("data_modelisation.csv")
all_id_client = list(dataframe['SK_ID_CURR'].unique())

y = dataframe['TARGET']
dataframe.drop(columns='TARGET', inplace=True)

# Chargement du modèle :

model = pickle.load(open('credit_final.pkl','rb'))


app=Flask(__name__)



@app.route('/')
def home():
    return render_template("index.html")
    #return "Prédiction rapide de l'acceptation ou non d'un prêt pour l'entreprise 'Prêt à dépenser' "
   



@app.route('/credit/<id_client>', methods=['GET'])
def predict(id_client):
    '''
    For rendering results on HTML GUI
    '''

    #ID = request.args.get('id_client')
    id_client = int(id_client)
    if id_client not in all_id_client:
        prediction="Ce client n'est pas répertorié"
    else :
        X = dataframe[dataframe['SK_ID_CURR'] == id_client]
        X = X.drop(['SK_ID_CURR'], axis=1)

        pred_prob = model.predict_proba(X)[:, 1][0]
        percent = round(pred_prob * 100, 0)

        if pred_prob < 0.5:
            classification = 'Rejet de la demande de credit'


        else:
            classification = 'Acceptation de la demande de credit'
    dict_final = {'prediction' : str(classification),
                  'proba' : pred_prob }

    
    return render_template('index.html', valeur=percent, prediction=classification)





# Run app.
#lancement de l'application
if __name__ == "__main__":
    app.run(debug=True)
