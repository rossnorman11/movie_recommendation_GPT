.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y movie_recom || :
	@pip install -e .

run_embed:
	python -c 'from movie_recom.interface.main import embed_data; embed_data()'

run_recommendation:
	python -c 'from movie_recom.interface.main import recommend; recommend()'

run_all: run_embed run_recommendation
