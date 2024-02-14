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
	python -c 'from movie_recom.interface.main import predict_movie; predict_movie()'

run_api:
	uvicorn movie_recom.api.fast:app --reload

call_api:
	python -c 'from movie_recom.interface.main import call_api; call_api()'

run_all:
	prediction

docker_build:
	docker build --tag=$GAR_IMAGE:dev .


docker_push:
	docker push europe-west1-docker.pkg.dev/${GCP_PROJECT}/movierecom/${GAR_IMAGE}:prod

docker_deploy:
	gcloud run deploy --image europe-west1-docker.pkg.dev/${GCP_PROJECT}/taxifare/${GAR_IMAGE}:prod --cpus ${GAR_CPU} --memory ${GAR_MEMORY} --region europe-west1--env-vars-file .env.yaml
