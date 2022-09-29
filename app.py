from fastapi import FastAPI
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from yaml import parse
from model import NLPModel

app = FastAPI()

model = NLPModel()

model_path = 'lib/models/SentimentClassifier.pkl'

with open(model_path, 'rb') as f:
    model.clf = pickle.load(f)

vec_path = 'lib/models/TFIDFVectorizer.pkl'

with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

parser = reqparse.RequestParser()
parser.add_argument('query')

class PredictSentiment(Resource):
    @app.get('/')
    async def get(self):
        args = parser.parse_args()
        user_query = args['query']

        uq_vectorize = model.vectorizer_transform(np.array([user_query]))
        prediction = model.predict(uq_vectorize)
        pred_proba = model.predict_proba(uq_vectorize)

        if prediction == 0:
            pred_text = 'negative'
        else:
            pred_text = 'positive'

        confidence = round(pred_proba[0], 3)

        output = {'prediction': pred_text, 'confidence': confidence}

        return output

if __name__=='__main__':
    app.run(debug=True)