import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style

from movie_recom.params import *
from movie_recom.ml_logic.encoders import mini_lm_encode
from movie_recom.ml_logic.data import get_raw_data, get_embedded_data, save_embedded_data
from movie_recom.ml_logic.model import find_n_nearest_neighbors
from movie_recom.ml_logic.preprocessor import shorten_synopsis


def embed_data():
    """
    load the data and shorten the synopsis
    embed the data
    """
    df = get_raw_data()
    # Process data
    df = shorten_synopsis(max_len=500, df=df)
    df_embedded = mini_lm_encode(df)
    save_embedded_data(df_embedded)

def embed_prompt() -> pd.DataFrame:
    """
    embed the prompt
    """
    prompt = "drug addict in america looking for work"
    prompt_df = pd.DataFrame({'title': ['prompt'], 'plot_synopsis': [prompt]})
    prompt_embedded = mini_lm_encode(prompt_df)
    return prompt_embedded

def recommend() -> list:
    # get the embedded prompt
    prompt_embedded = embed_prompt()

    # get the embedded data
    df_embedded = get_embedded_data()

    recom_list = find_n_nearest_neighbors(n=5, prompt_embedded=prompt_embedded, df_embedded=df_embedded)
    print(recom_list)
    return recom_list

if __name__ == '__main__':
    pass
