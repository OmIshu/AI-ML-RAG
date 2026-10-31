install:
	pip install -e ".[dev]"

ingest:
	python scripts/ingest.py data/pdfs

api:
	uvicorn ragbot.api.main:app --port 8000

eval-data:
	python -m ragbot.evaluation.make_dataset 50

eval:
	python -m ragbot.evaluation.run

test:
	pytest -q
