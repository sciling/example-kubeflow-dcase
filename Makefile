finalize:
	poetry update

pre-commit:
	poetry run pre-commit install

install: finalize
	poetry install

tests: install
	poetry run coverage run -m pytest -p no:sugar -s -q tests/

lint: install
	poetry run pre-commit run -a
