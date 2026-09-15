.PHONY: install test demo
install:
	python -m pip install -e ".[dev]"
test:
	python -m pytest -q
demo:
	python -m acie demo --out outputs/demo
