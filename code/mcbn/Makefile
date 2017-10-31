IMG_BASE_TAG = mcbn/evaluator

build:
	sudo docker build --no-cache -t $(IMG_BASE_TAG):local .

run:
	sudo docker run -d -v $(shell pwd)/:/mcbn \
	$(IMG_BASE_TAG):local \
	/bin/bash -c "virtualenv docker_env; \
	source docker_env/bin/activate; \
	pip install -r requirements.txt; \
	pip install scikit-optimize==0.4; \
	python run_evaluation.py $(dataset) $(seed); \
	tail -f /dev/null"
