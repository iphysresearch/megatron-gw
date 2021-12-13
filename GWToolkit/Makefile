# Oneshell means I can run multiple lines in a recipe in the same shell, so I don't have to
# chain commands together with semicolon
.ONESHELL:
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash

.PHONY: help setup mypy lint test clean docs heck lab notebook add-to-jupyter remove-from-jupyter
#######
# Help
#######

.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

CONDA = conda
CONDA_REQUIREMENTS = environment.yml
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

ENV_NAME = gwtoolkit
#ENV_NAME = $$(python -c "import yaml;print(yaml.safe_load(open('$(CONDA_REQUIREMENTS)'))['name'])")

###################
# Conda Enviroment
###################

setup: ## Create the conda environment
	@echo 'creating the '$(ENV_NAME)' environment'
	@echo
	$(CONDA) env create --file $(CONDA_REQUIREMENTS)
	@echo

clean-pyc:
	@echo 'cleaning up temporary files'
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*

clean: clean-pyc clean-test ## Clean up the existing pyc files and coverage report
	@echo 'NOTE: you should clean up the following occasionally (by hand)'
	git clean -fdn

test: clean ## 1. Unit tests 
	$(CONDA_ACTIVATE) $(ENV_NAME) && py.test tests --cov=$(ENV_NAME) --cov-report=term-missing --cov-fail-under 95
test_PCL: clean 
	$(CONDA_ACTIVATE) && py.test tests --cov=$(ENV_NAME) --cov-report=term-missing --cov-fail-under 95

lint: ## 2. Lint to Ensure Consistency
	$(CONDA_ACTIVATE) $(ENV_NAME) && pylint $(ENV_NAME) -j 4 --reports=y --max-line-length=120 --max-attributes=15 --fail-under=9
lint_PCL: 
	$(CONDA_ACTIVATE) && pylint $(ENV_NAME) -j 4 --reports=y --max-line-length=120 --max-attributes=15 --fail-under=9

mypy: ## 3. Check For Type Errors (mypy)
	$(CONDA_ACTIVATE) $(ENV_NAME) && mypy $(ENV_NAME) --ignore-missing-imports
mypy_PCL: 
	$(CONDA_ACTIVATE) && mypy $(ENV_NAME) --ignore-missing-imports
# > Why we use `--ignore-missing-imports`? See [this](https://mypy.readthedocs.io/en/stable/running_mypy.html#missing-imports).

docs: FORCE
	cd docs; $(CONDA_ACTIVATE) $(ENV_NAME) && sphinx-apidoc -o ./source ./$(ENV_NAME)
	cd docs; $(CONDA_ACTIVATE) $(ENV_NAME) && sphinx-build -b html ./source ./build
FORCE:

check: test lint mypy ## Run the full suite of checks [1+2+3]
check_PCL: test_PCL lint_PCL mypy_PCL ## Run the full suite of checks for PCL [1+2+3]

###################
# Jupyter Enviroment
###################

lab: ## Runing jupyter lab in conda environment
	$(CONDA_ACTIVATE) $(ENV_NAME) && jupyter lab --allow-root

notebook: ## Runing jupyter notebook in conda environment
	$(CONDA_ACTIVATE) $(ENV_NAME) && jupyter notebook --allow-root	

add-to-jupyter: ## Register the conda environment to Jupyter
	$(CONDA_ACTIVATE) $(ENV_NAME) && python -s -m ipykernel install --user --name $(ENV_NAME)

remove-from-jupyter: ## Remove the conda environment from Jupyter
	jupyter kernelspec uninstall $(ENV_NAME)
