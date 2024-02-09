import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from movie_recom.interface.main import recommend

app = FastAPI()

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
        n_recom: int = 5 # number of recommendations
    ):
    """
    gives a list of n_recom recommendations based on the prompt
    """
    recom_list = recommend(prompt, n_recom)
    return {"Our recommendation is": recom_list}


@app.get("/")
def root():
    return {"message": "Welcome to the MovieRecommendation API"}

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}
