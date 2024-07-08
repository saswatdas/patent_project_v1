install:
		pip install --upgrade pip &&\
			pip install -r requirements.txt

format:
		black pages/*.py

lint:
		pylint --disable=R,C pages/*.py  || true



all: install format lint 
