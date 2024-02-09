.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y movie_recom || :
	@pip install -e .

embedding:
	python -c 'from movie_recom.interface.main import embed_data; embed_data()'

recommendation:
	python -c 'from movie_recom.interface.main import recommend; recommend()'

run_api:
	uvicorn movie_recom.api.fast:app --reload

call_api:
	python -c 'from movie_recom.interface.main import call_api; call_api()'

run_all: embedding recommendation
