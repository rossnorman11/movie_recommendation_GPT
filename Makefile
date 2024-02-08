.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y movie_recom || :
	@pip install -e .

embedding:
	python -c 'from movie_recom.interface.main import embed_data; embed_data()'

recommendation:
	python -c 'from movie_recom.interface.main import recommend; recommend()'

run_all: embedding recommendation
