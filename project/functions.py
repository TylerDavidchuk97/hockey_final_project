
#this is a file that contains all functions required to generate predictions from the final_project.ipynb file
#this file was created as there seems to be an issue importing functions from .ipynb files into .py files, so the functions.py files will be imported instead-
#-as it will follow standard python machinery
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from tensorflow import keras
from keras.models import load_model
pd.options.mode.chained_assignment = None  # default='warn'


#create function to return our closest compareable player along with their stats which is pulled directly from the knn_dataframe
def compareable(player):

    df_model = pd.read_csv('../hockey_final_project/Prediction_Dataset.csv')

    #create a new df for the KNN
    df_knn = df_model.drop(['season', 'link'], axis=1)

    #create a groupby aggregating all the stats to get career averages
    group = df_knn.groupby('playername').agg({'gp': ['sum'], 'tp': ['sum'], 'ppg': ['mean'], 'gpg': ['mean'],
                                    'apg': ['mean'], 'pmpg': ['mean'], '+/-pg': ['mean'],
                                    'position_C': ['mean'], 'position_D': ['mean'], 'position_F': ['mean'], 
                                    'position_L': ['mean'], 'position_R': ['mean'], 'position_W': ['mean']})

    #drop column level 1 which is the mean and sum as we have an multiindex and it's a bit of a pain to work with it
    group.columns = group.columns.droplevel(1)

    #create a nearest neighbours for our dataset and fit it
    nn= NearestNeighbors(radius=0.5, algorithm='auto')
    KNN_model = nn.fit(group)


    #get the closest compareable player for our asked about player
    comparable = KNN_model.kneighbors([group.loc[player,:]], 2, False)

    #convert array to list which is the index in our group dataframe of which player is closest to them
    comparable = list(comparable[0])

    return group.iloc[comparable]


def shape_data(df, lags):
    shape_of_dataset = df.shape

    #create dimensions force int otherwise reshape will throw error
    dim1 = int(shape_of_dataset[0] / lags)
    dim2 = int(lags)
    dim3 = int(shape_of_dataset[1])

    #reshape and remove value we want to predict for which is ppg which is in position [13] -1 will start from but not include
    reshaped = np.array(df).reshape((dim1, dim2, dim3))

    #remove ppg from indpendants 
    x = reshaped[:,:,:-1]

    #remove our responding variable ppg from the reshaped data
    y = reshaped[:,:,-1]

    return x , y

def scale_data(train, test):

    #inintalize min max scaler
    scaler = MinMaxScaler()
    
    #because the data is in a 3d shape we need to reshape it to 2D form which is what MinMaxscaler expects to transform the data
    #we add another reshape afterwards to reshape the dataset back to the original 3D shape we had before transformation
    #fit_transform the x_train data
    train = scaler.fit_transform(train.reshape(-1, train.shape[-1])).reshape(train.shape)

    #fit transform the x_test data
    test = scaler.transform(test.reshape(-1, test.shape[-1])).reshape(test.shape)

    return train, test


def players(df, seasons):

    #group the data so that we have two rows for every player to predict from
    grouped = df.sort_values(by=['playername', 'season']).groupby('link', dropna=False).tail(seasons)

    #filter the data if we don't have three rows we've got an issue predicting so we'll need to exclude those players unfortunately as it throws the prediction off pretty bad
    counts = grouped['link'].value_counts()
    filtered_data = grouped[~grouped['link'].isin(counts[counts < seasons].index)]
    
    filtered_data

    return filtered_data


def predict(df, lstm_model):

    with open('../hockey_final_project/x_train.pkl', 'rb') as f:
        x_train = pickle.load(f)

    eligible_players = players(df, 3).reset_index(drop=True)
    
    #drop the columns we don't want
    player_data_to_predict = eligible_players.drop(['season', 'playername', 'link'], axis=1)

    #reorder the columns so we can isolate ppg easier
    player_data_to_predict = player_data_to_predict[['gp','tp','gpg','apg','pmpg','+/-pg','position_C','position_D','position_F','position_L','position_R','position_W','ppg']]

    #now we need to reshape the data
    player_data_to_predict_x, player_data_to_predict_y = shape_data(player_data_to_predict, 3)

    #now we scale the data
    #inintalize min max scaler
    mm_scaler = MinMaxScaler()
    
    #because the data is in a 3d shape we need to reshape it to 2D form which is what MinMaxscaler expects to transform the data
    #we add another reshape afterwards to reshape the dataset back to the original 3D shape we had before transformation
    #fit_transform the x_train data
    train = mm_scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)

    scaled_x = mm_scaler.transform(player_data_to_predict_x.reshape(-1, player_data_to_predict_x.shape[-1])).reshape(player_data_to_predict_x.shape)

    #get our ppg predictions
    ppg_results = lstm_model.predict(scaled_x)

    #create empty table with 12 fields matching our features without responding variable
    trainPredict_dataset_like = np.zeros(shape=(len(player_data_to_predict_x), 12))
    #put the predicted values in the right field
    trainPredict_dataset_like[:,0] = ppg_results[:,0]
    #inverse transform and then select the right field
    results = mm_scaler.inverse_transform(trainPredict_dataset_like)[:,0]

    return results, eligible_players



def get_result(lstm_model, player):
    
    data = pd.read_csv('../hockey_final_project/Prediction_Dataset.csv')

    results, eligible_players = predict(data, lstm_model)

    last_season = players(eligible_players, 1).reset_index(drop=True)

    #add the results to the df
    last_season['LSTM_predicted_next_season_ppg'] = results

    #grab only information we would like to see
    last_season = last_season[['playername', 'link', 'ppg', 'LSTM_predicted_next_season_ppg']]

    #rename ppg to current season ppg, and link to Elite prospects link
    last_season.rename(columns={'ppg':'current_season_ppg', 'link':'elite_prospects_link'}, inplace=True)
    result = last_season.loc[last_season['playername']==player]

    return result
