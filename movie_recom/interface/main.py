import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style

from movie_recom.params import *
from movie_recom.ml_logic.encoders import bert_encode, tf_vectorize
from movie_recom.ml_logic.data import get_data
from movie_recom.ml_logic.model import predict_NN, vector_cosine
from movie_recom.ml_logic.preprocessor import create_output_NN
import requests

def embed_prompt(prompt: str) -> pd.DataFrame:
    """
    embed the prompt
    """
    #put it into a dataframe for NN
    prompt_embedded = bert_encode(prompt)
    return prompt_embedded

def merge_promt_with_favorits(prompt_embedded: pd.DataFrame, favs: list) -> pd.DataFrame:
    # get the embedded data
    # TODO: Adjust to latest models
    if EMBEDDING_TYPE == 'mini':
        df_embedded = get_data('processed_data/data_mini_embedded.csv')
        df_filtered = df_embedded[df_embedded.index.isin(favs)] # embedded dataframe with just the favorites
        series = prompt_embedded.iloc[0,:] # convert the prompt dataframe to a series
        df_filtered.loc['prompt'] = series.to_list() # add the prompt to the dataframe (concat didnt work well)
        mean_df = df_filtered.mean(axis=0).to_frame().T # get the mean of the dataframe, keep it as dataframe
        mean_df.index = ['prompt'] # set the index to 'prompt'
        return mean_df
    return prompt_embedded

def find_recommendation_vector(text):
    # Vectorise user input
    vectorized_prompt = tf_vectorize(text)
    #return dataframe with movie recommendations and similarity score
    return vector_cosine(vectorized_prompt)

def predict(prompt: str = 'drug addict getting his life back on track') -> list:

    '''
    get the prompt and recommend movies based on it
    '''
    # get the embedded prompt

    # recommend with cosine similarity
    recom_list =  find_recommendation_vector(prompt)

    prompt_embedded = embed_prompt(prompt)
    pred_ratings = predict_NN(prompt_embedded)
    pred_recommendations = create_output_NN(pred_ratings)

    combined = pd.merge(left=pred_recommendations, right=recom_list, left_index=True, right_on='title', how='left')
    combined['sum'] = combined['rating'] + 3*combined['similarity']

    recommendations = combined.sort_values(by='sum', ascending=False)[0:5]

    print(recommendations)

    return recommendations

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
