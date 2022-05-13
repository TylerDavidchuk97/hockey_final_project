import pickle
import import_ipynb
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, jsonify, request
from keras.models import load_model
from project import players



#create flask app
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

#load our two datasets we need to use to predict with
lstm_df = pd.read_csv('../hockey_final_project/LSTM_Dataset.csv')
knn_df = pd.read_csv('../hockey_final_project/KNN_Dataset.csv')

#load the model from disk
with open('Hockey_KNN_model.pkl', 'rb') as f:
    loaded_model_knn = pickle.load(f)

#load our LSTM_Hockey_Model in using keras
LSTM_model = load_model('Hockey_LSTM_Model.h5')


@app.route('/')
def index_view():
    return render_template('index.html')

@app.route('/predict_lstm.html', methods=['POST'])
def predict():

    player = request.json(force=True)
    

    return render_template('predict_lstm.html')

@app.route('/predict_knn.html', methods=['POST'])
def closest():

    return render_template('predict_knn.html')
            

if __name__ == '__main__':
    app.run(debug=True, port=5000)