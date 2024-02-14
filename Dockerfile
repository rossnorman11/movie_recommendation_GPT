FROM python:3.10.6-buster
WORKDIR workdir_docker

COPY requirements_docker.txt requirements_docker.txt
RUN pip install --upgrade pip
RUN pip install -r requirements_docker.txt
COPY movie_recom movie_recom
COPY raw_data/movie_title.pkl raw_data/movie_title.pkl
COPY processed_data processed_data
COPY saved_models saved_models
COPY setup.py setup.py
RUN pip install .
CMD uvicorn movie_recom.api.fast:app --host 0.0.0.0 --port $PORT
