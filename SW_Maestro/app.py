from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np

app = Flask(__name__)
api = Api(app)

clf_path = 'pk_model/clf.pk'
with open(clf_path, 'rb') as f:
    clf = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('emg')

class PredictGesture(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['emg']
        user_query = [float(s) for s in user_query]

        # make a prediction
        go = [user_query,]
       
        prediction = clf.predict(np.array(go))

        # Output 1 to 5, hand gesture number
        prediction = prediction[0]

        # create JSON object
        output = {'prediction': prediction}
        
        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictGesture, '/')


if __name__ == '__main__':
    app.run(debug=True)
