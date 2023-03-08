import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Creer l'application
app = Flask(__name__)
# load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    #convertir les features a une numpyarray
    data = [np.array(features)]
    prediction = model.predict(data)
    return render_template("index.html", prediction = "L'espèce de la fleur est :  {}".format(prediction))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port='5000',debug=True)