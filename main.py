import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, jsonify, request
from project.functions import compareable, players, shape_data, scale_data, predict, get_result

#create flask app
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

#load our two datasets we need to use to predict with
lstm_df = pd.read_csv('../hockey_final_project/LSTM_Dataset.csv')


#load our LSTM_Hockey_Model in using keras
LSTM_model = load_model('Hockey_LSTM_Model.h5')


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