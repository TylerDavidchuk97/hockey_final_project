from unittest import result
from matplotlib.pyplot import title
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from flask import Flask, render_template, jsonify, request
from project.functions import compareable, get_result

#create flask app
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/')
def index_view():
    return render_template('index.html')

@app.route('/predict_lstm', methods=['POST', 'GET'])
def predict():

    #grab from our webpage to python
    if request.method == 'POST':
            player = request.form['playername_lstm']

            lstm_results = get_result(str(player))
            
            #return result from python to webpage
            return render_template('predict_lstm.html', tables=[lstm_results.to_html(classes='data', header=True, index=False)])

        #render webpage for get requests, as we won't actually input data as we're clicking the button to load the page
    if request.method == 'GET':
        return render_template('predict_lstm.html')

@app.route('/predict_knn', methods=['POST', 'GET'])
def closest():

    #grab from our webpage to python
    if request.method == 'POST':
            player = request.form['playername_knn']

            knn_results = compareable(str(player))
            
            #return result from python to webpage
            return render_template('predict_knn.html', tables=[knn_results.to_html(classes='data', header=True, index=True)])

        #render webpage for get requests, as we won't actually input data as we're clicking the button to load the page
    if request.method == 'GET':
        return render_template('predict_knn.html')
            

if __name__ == '__main__':

    app.run(debug=True, port=5000)