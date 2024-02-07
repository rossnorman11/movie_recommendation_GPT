import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style

from movie_recom.params import *
from movie_recom.ml_logic.encoders import mini_lm_encode
from movie_recom.ml_logic.data import get_data
from movie_recom.ml_logic.model import find_n_nearest_neighbors
from movie_recom.ml_logic.preprocessor import shorten_synopsis

def preprocess() -> pd.DataFrame:
    """
    load the data and shorten the synopsis
    """
    # Retrieve data
    df = get_data()
    # Process data
    df = shorten_synopsis(max_len=500, df=df)

def embed_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    embed the data
    """
    df_embedded = mini_lm_encode(df)
    return df_embedded

def embed_prompt(prompt: str) -> pd.DataFrame:
    """
    embed the prompt
    """
    prompt_df = pd.DataFrame({'title': 'prompt' ,'plot_synopsis': prompt})
    prompt_embedded = mini_lm_encode(prompt_df)
    return prompt_embedded

def recommend(df_embedded: pd.DataFrame, prompt_embedded: pd.DataFrame) -> list:
    recom_list = find_n_nearest_neighbors(n=5, prompt_embedded=prompt_embedded, df_embedded=df_embedded)
    print(recom_list)
    return recom_list

if __name__ == '__main__':
    data = preprocess()
    data_embedded = embed_data(data)
    prompt = "drug addict in america looking for work"
    prompt_embedded = embed_prompt(prompt)
    recommend(data_embedded, prompt_embedded)
