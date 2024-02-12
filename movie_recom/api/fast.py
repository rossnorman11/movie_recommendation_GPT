import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from movie_recom.interface.main import embed_prompt, merge_promt_with_favorits
from pathlib import Path
from movie_recom.params import *
import pickle

app = FastAPI()
#load pickle model
# Get the parent folder of the current file (goes up 2 levels)
parent_folder_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# Define the path and foldername to save the data
model_path = Path(parent_folder_path).joinpath("saved_models/nbrs.pkl")
with open(model_path, 'rb') as f:
    app.state.model = pickle.load(f)


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict(
        prompt: str = "drug addict in america looking for work", # prompt
        n_recom: int = 5, # number of recommendations
        fav_list: list = [] # list of favorite movies
    ):
    """
    gives a list of n_recom recommendations based on the prompt
    """
    prompt_embedded = embed_prompt(prompt)
    final_prompt_embedded = prompt_embedded
    if len(fav_list) > 0:
        final_prompt_embedded = merge_promt_with_favorits(prompt_embedded, fav_list)
    distances, indices = app.state.model.kneighbors(final_prompt_embedded, n_neighbors=n_recom)

    # generate output list
    # load list of titles
    filepath = Path(PARENT_FOLDER_PATH).joinpath("processed_data/data_titlenames.csv")
    df_titles = pd.read_csv(filepath, index_col=0)
    list_of_titles = []
    for index in indices[0][0:]:
        list_of_titles.append(df_titles.iloc[index][0])
    return {"Our recommendation is": list_of_titles}


@app.get("/")
def root():
    return {"message": "Welcome to the MovieRecommendation API"}

# # Define a root `/` endpoint
# @app.get('/')
# def index():
#     return {'ok': True}
