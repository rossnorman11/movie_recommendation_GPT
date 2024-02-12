import numpy as np
import pandas as pd
from colorama import Fore, Style
import pickle
from pathlib import Path
from movie_recom.params import *
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

####################################
###     K NEAREST NEIGHBORS      ###
####################################

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

####################################
###     COSINE SIMILIRATY        ###
####################################

from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_sim(prompt):
    '''
    computes cosine similarity between an embedded prompt and embedded plots
    '''
    model_name = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)


    # load embedded plots
    filepath_plot = Path(PARENT_FOLDER_PATH).joinpath("processed_data/embeddings_plot.npy")
    plot_embedded = np.load(filepath_plot)
    #print(plot_embedded)
    # load movie df with titles
    filepath_title = Path(PARENT_FOLDER_PATH).joinpath("raw_data/movie_with_summary.csv")
    df = pd.read_csv(filepath_title)

    # compute cosine similarity
    user_token = tokenizer(prompt, return_tensors="pt")
    user_outputs = model(**user_token)
    user_embedded = user_outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    cos_similarities_emb = cosine_similarity([user_embedded], plot_embedded).flatten()
    similar_movies_emb = pd.DataFrame({'title': df['title'], 'similarity': cos_similarities_emb})
    similar_movies_emb = similar_movies_emb.sort_values(by='similarity', ascending=False)
    top_10_recommendations_emb = similar_movies_emb.head(10)[['title', 'similarity']]
    emb_recommendations = f"Top 10 recommendations:\n{top_10_recommendations_emb.to_string(index=False)}"
    return emb_recommendations

    # select top 10 recommended movies
    top_10_recommendations_emb = similar_movies_emb.head(10)[['title', 'similarity']]
    #emb_recommendations = f"Top 10 recommendations:\n{top_10_recommendations_emb.to_string(index=False)}"

    return top_10_recommendations_emb
