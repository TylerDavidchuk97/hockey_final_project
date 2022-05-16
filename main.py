import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, jsonify, request
from project.functions import compareable, players, shape_data, scale_data, predict, get_result

#create flask app
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/')
def index_view():
    return render_template('index.html')

@app.route('/predict_lstm.html', methods=['POST', 'GET'])
def predict():

    #grab from our webpage to python
    if request.method == 'POST':
            player = request.form['playername']

    #return result from python to webpage
    return render_template('predict_lstm.html', n = player)

@app.route('/predict_knn.html', methods=['POST', 'GET'])
def closest():

    return render_template('predict_knn.html')
            

if __name__ == '__main__':
    app.run(debug=True, port=5000)