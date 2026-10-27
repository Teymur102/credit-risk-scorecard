PY := .venv/bin/python

.PHONY: all setup run test clean

all: setup run test

setup:
	test -d .venv || python3 -m venv .venv
	$(PY) -m pip install -q -r requirements.txt

run:
	$(PY) run.py

test:
	$(PY) -m pytest -q

clean:
	rm -rf reports figures
