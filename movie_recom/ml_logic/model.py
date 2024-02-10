import numpy as np
import pandas as pd
from colorama import Fore, Style
import pickle
from pathlib import Path
from movie_recom.params import *

from sklearn.neighbors import NearestNeighbors

def fit_n_nearest_neighbors(n: int, df_embedded: pd.DataFrame):
    '''
    finds n nearest neighbors using the NearestNeighbors algorithm
    '''
    #instatiate model
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(df_embedded)
    # save in pickle
    # Define the path and foldername to save the data
    model_path = Path(PARENT_FOLDER_PATH).joinpath("saved_models/nbrs.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(nbrs, f)


def predict_n_nearest_neighbors(n_neighbors: int ,prompt_embedded: pd.DataFrame):
    '''
    finds n nearest neighbors using the NearestNeighbors algorithm
    '''
    #load pickle model
    # Define the path and foldername to save the data
    model_path = Path(PARENT_FOLDER_PATH).joinpath("saved_models/nbrs.pkl")
    with open(model_path, 'rb') as f:
        loaded_nbrs = pickle.load(f)
    #find the n nearest neighbors
    distances, indices = loaded_nbrs.kneighbors(prompt_embedded, n_neighbors=n_neighbors)
    # generate output list
    list_of_titles = []
    # load list of titles
    filepath = Path(PARENT_FOLDER_PATH).joinpath("processed_data/data_titlenames.csv")
    df_titles = pd.read_csv(filepath, index_col=0)
    for index in indices[0][0:]:
        list_of_titles.append(df_titles.iloc[index][0])
    return list_of_titles
