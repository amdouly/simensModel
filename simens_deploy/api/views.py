#Import necessary libraries
import pickle
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render
import numpy as np

# Create your views here.
@api_view(['GET'])
def index_page(request):
    return_data = {
        "error" : "0",
        "message" : "Successful",
    }
    return Response(return_data)

@api_view(["POST"])
def predict_typediabetique(request):

    try:

        AGE = request.data.get('AGE',None)
        SEXE = request.data.get('SEXE',None)
        deshydratation = request.data.get('deshydratation',None)
        tension_arterielle_max = request.data.get('tension_arterielle_max',None)
        tension_arterielle_min = request.data.get('tension_arterielle_min',None)
        sucre = request.data.get('sucre',None)
        taille = request.data.get('taille',None)
        poids = request.data.get('poids',None)
        poulsBatMin = request.data.get('poulsBatMin',None)
        corps_cetonique = request.data.get('corps_cetonique',None)
        fields = [AGE, SEXE, deshydratation, tension_arterielle_max,
        tension_arterielle_min, sucre, taille, poids, poulsBatMin,
        corps_cetonique]
        if not None in fields:
            #Datapreprocessing Convert the values to float
            AGE = float(AGE)
            SEXE = float(SEXE)
            deshydratation = float(deshydratation)
            tension_arterielle_max = float(tension_arterielle_max)
            tension_arterielle_min = float(tension_arterielle_min)
            sucre = float(sucre)
            taille = float(taille)
            poids = float(poids)
            poulsBatMin = float(poulsBatMin)
            corps_cetonique = float(corps_cetonique)
            result = [AGE, SEXE, deshydratation, tension_arterielle_max,
                  tension_arterielle_min, sucre, taille, poids, poulsBatMin,
                  corps_cetonique]
            #Passing data to model & loading the model from disks
            model_path = 'log_reg.pkl'
            regressionlog = pickle.load(open(model_path,'rb'))
            prediction = regressionlog.predict([result])[0]
            conf_score =  np.max(regressionlog.predict_proba([result]))*100
            predictions = {
                'prediction Diabete Type' : prediction,
                'confidence_score' : conf_score
            }
        else:
            predictions = {
                'error' : '1',
                'message': 'Invalid Parameters'                
            }
    except Exception as e:
        predictions = {
            'error' : '2',
            "message": str(e)
        }
    
    return Response(predictions)