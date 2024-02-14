import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from movie_recom.interface.main import embed_prompt, merge_prompt_with_favorites, predict
from pathlib import Path
from movie_recom.params import *
import pickle

app = FastAPI()
#load pickle model



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
        fav_list: list=[],
        weight_n: float=0.5
    ):
    """
    gives a list of n_recom recommendations based on the prompt
    """
    # generate output list

    movie_list = predict(prompt, fav_list, weight_n)
    # load list of titles

    return {"Our recommendation is": movie_list}


@app.get("/")
def root():
    return {"message": "Welcome to the MovieRecommendation API"}

# # Define a root `/` endpoint
# @app.get('/')
# def index():
#     return {'ok': True}
