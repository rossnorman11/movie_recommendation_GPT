import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style

from movie_recom.params import *
from movie_recom.ml_logic.encoders import mini_lm_encode, bert_encode
from movie_recom.ml_logic.data import get_data, save_data
from movie_recom.ml_logic.model import fit_n_nearest_neighbors, predict_n_nearest_neighbors, compute_cosine_sim
from movie_recom.ml_logic.preprocessor import shorten_synopsis
import requests


def embed_data_with_mini():
    """
    load the data and shorten the synopsis
    embed the data
    """
    # get the data from data.get_raw_data
    df = get_data("raw_data/mpst_full_data.csv")
    # Process data
    # shorten the synopsis with preprocessor.shorten_synopsis
    df = shorten_synopsis(max_len=500, df=df)
    # embed the synopsis with encoders and saves it

    #if EMBEDDING_TYPE == 'mini':
    df_encoded, df_index = mini_lm_encode(df)
    save_data(df_encoded, 'processed_data/data_mini_embedded.csv')
    save_data(df_index, 'processed_data/data_titlenames.csv')
    # elif EMBEDDING_TYPE == 'bert':
    #     df_encoded, df_index = bert_encode(df)
    #     save_data(df_encoded, 'processed_data/data_bert_embedded.csv')
    #     save_data(df_index, 'processed_data/data_titlenames.csv')


def embed_prompt(prompt: str) -> pd.DataFrame:
    """
    embed the prompt
    """
    #put it into a dataframe
    prompt_df = pd.DataFrame({'title': ['prompt'], 'plot_synopsis': [prompt]})


    #embed the prompt with encoders
    if EMBEDDING_TYPE == 'mini':
        prompt_embedded, df_index = mini_lm_encode(prompt_df)
        return prompt_embedded
    if EMBEDDING_TYPE == 'bert':
        prompt_embedded, df_index = bert_encode(prompt_df)
        return prompt_embedded


def merge_promt_with_favorits(prompt_embedded: pd.DataFrame, favs: list) -> pd.DataFrame:
    # get the embedded data
    if EMBEDDING_TYPE == 'mini':
        df_embedded = get_data('processed_data/data_mini_embedded.csv')
        df_filtered = df_embedded[df_embedded.index.isin(favs)] # embedded dataframe with just the favorites
        series = prompt_embedded.iloc[0,:] # convert the prompt dataframe to a series
        df_filtered.loc['prompt'] = series.to_list() # add the prompt to the dataframe (concat didnt work well)
        mean_df = df_filtered.mean(axis=0).to_frame().T # get the mean of the dataframe, keep it as dataframe
        mean_df.index = ['prompt'] # set the index to 'prompt'
        return mean_df
    return prompt_embedded


def fit_nearest_neighbors(n_neighbors: int = 10):
    '''
    fit the model
    '''

    if EMBEDDING_TYPE == 'mini':
        # get the embedded data
        df_embedded = get_data("processed_data/data_mini_embedded.csv")
        fit_n_nearest_neighbors(n=n_neighbors, df_embedded=df_embedded)
    if EMBEDDING_TYPE == 'bert':
        # get the embedded data
        df_embedded = get_data("processed_data/data_bert_embedded.csv")
        # fit the model with model.fit_n_nearest_neighbors
        fit_n_nearest_neighbors(n=n_neighbors, df_embedded=df_embedded)

def predict(prompt: str = 'godfather movie with a lot of action', n_neighbors: int = 5) -> list:

    '''
    get the prompt and recommend movies based on it
    '''
    # get the embedded prompt
    prompt_embedded = embed_prompt(prompt)
    #import ipdb; ipdb.set_trace()

    if SEARCH_TYPE == 'knn':
        # find the nearest neighbors with model.find_n_nearest_neighbors
        recom_list = predict_n_nearest_neighbors(n_neighbors=n_neighbors, prompt_embedded=prompt_embedded)
        print(recom_list)
        return recom_list

    elif SEARCH_TYPE == 'cosine':
        # recommend with cosine similarity
        recom_list =  compute_cosine_sim(prompt_embedded)
        print(recom_list)
        return recom_list
    else: print("hi")

def call_api():
    url = 'http://localhost:8000/predict'

    params = {
        'prompt': 'Love story in England without happy ending', # 0 for Sunday, 1 for Monday, ...
        'n_recom': 7
    }

    response = requests.get(url, params=params)
    response.json() #=> {wait: 64}
    # print(response.json())

def test():
    pass

if __name__ == '__main__':
    pass
    # test()
