.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y movie_recom || :
	@pip install -e .

data_embedding_with_mini:
	python -c 'from movie_recom.interface.main import embed_data_with_mini; embed_data_with_mini()'

fit_nearest_neighbors:
	python -c 'from movie_recom.interface.main import fit_nearest_neighbors; fit_nearest_neighbors()'

prediction:
	python -c 'from movie_recom.interface.main import predict; predict()'

run_api:
	uvicorn movie_recom.api.fast:app --reload

call_api:
	python -c 'from movie_recom.interface.main import call_api; call_api()'

run_all: prediction
