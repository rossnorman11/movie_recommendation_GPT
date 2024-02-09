FROM python:3.10.6-buster
WORKDIR workdir_docker
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY movie_recom movie_recom
COPY setup.py setup.py
RUN pip install .
CMD uvicorn movie_recom.api.fast:app --host 0.0.0.0 --port $PORT
