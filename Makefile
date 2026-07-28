.PHONY: setup data train app test

# One-time setup: create a venv and install dependencies.
setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Download the real MovieLens catalog + regenerate the synthetic
# subscriber/session/churn-label layer on top of it.
data:
	.venv/bin/python scripts/build_data.py

# Train the churn model and write models/churn_model.pkl + models/metrics.json.
train:
	.venv/bin/python -m src.model

# Run the self-serve Streamlit data product.
app:
	.venv/bin/streamlit run app.py

# Full test suite.
test:
	.venv/bin/python -m pytest -q
