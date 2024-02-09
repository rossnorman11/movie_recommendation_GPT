import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style

from movie_recom.params import *
from movie_recom.ml_logic.encoders import mini_lm_encode, bert_encode
from movie_recom.ml_logic.data import get_raw_data, get_embedded_data, save_embedded_data
from movie_recom.ml_logic.model import find_n_nearest_neighbors
from movie_recom.ml_logic.preprocessor import shorten_synopsis
import requests


def embed_data():
    """
    load the data and shorten the synopsis
    embed the data
    """
    # get the data from data.get_raw_data
    df = get_raw_data()
    # Process data
    # shorten the synopsis with preprocessor.shorten_synopsis
    df = shorten_synopsis(max_len=500, df=df)

    # embed the synopsis with encoders.mini_lm_encode
    #df_embedded = mini_lm_encode(df)

    # embed the synopsis with encoders.bert_encode
    df_embedded = bert_encode(df)

    save_embedded_data(df_embedded)

def embed_prompt(prompt: str) -> pd.DataFrame:
    """
    embed the prompt
    """
    #put it into a dataframe
    prompt_df = pd.DataFrame({'title': ['prompt'], 'plot_synopsis': [prompt]})
    #embed the prompt with encoders.mini_lm_encode
    #prompt_embedded = mini_lm_encode(prompt_df)

    #embed the prompt with encoders.bert_encode
    prompt_embedded = bert_encode(prompt_df)

    return prompt_embedded

def recommend(prompt: str="love story in London", n_neighbors: int = 5) -> list:
    '''
    get the prompt and recommend movies based on it
    '''
    # get the embedded prompt
    prompt_embedded = embed_prompt(prompt)

    # get the embedded data
    df_embedded = get_embedded_data()

    # find the nearest neighbors with model.find_n_nearest_neighbors
    recom_list = find_n_nearest_neighbors(n=n_neighbors, prompt_embedded=prompt_embedded, df_embedded=df_embedded)
    print(recom_list)
    return recom_list

def call_api():
    url = 'http://localhost:8000/predict'

    params = {
        'prompt': 'Love story in England without happy ending', # 0 for Sunday, 1 for Monday, ...
        'n_recom': 7
    }

    response = requests.get(url, params=params)
    response.json() #=> {wait: 64}
    print(response.json())

if __name__ == '__main__':
    pass
