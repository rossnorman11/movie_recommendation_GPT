import numpy as np
import pandas as pd
import time
from colorama import Fore, Style

from sklearn.neighbors import NearestNeighbors

def find_n_nearest_neighbors(n: int, prompt_embedded: pd.DataFrame, df_embedded: pd.DataFrame):
    '''
    finds n nearest neighbors using the NearestNeighbors algorithm
    '''
    #instatiate model
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(df_embedded)
    #find the n nearest neighbors
    distances, indices = nbrs.kneighbors(prompt_embedded)
    # generate output list
    list_of_titles = []
    for index in indices[0][0:]:
        list_of_titles.append(df_embedded.iloc[index].name)
    return list_of_titles
