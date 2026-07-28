# Streaming Engagement Data Product

A solo, end-to-end **data-product demo**: a churn model and a self-serve
analytics tool built on a real movie catalog plus an explicitly synthetic
subscriber-engagement layer, wrapped in a Streamlit app a non-technical
stakeholder can use directly - dashboard tiles, driver charts, and a
business-question box that only ever answers from real computed numbers.

It is not a production system with real users or real business-impact
numbers; it is a demonstration of the full loop a data-product role does
day to day: build the data, engineer features, train and evaluate a model,
turn the model and data into a constrained GenAI question-answering layer
with a grounding guardrail, expose that layer as both a UI and an MCP
server, and write the enablement material a business stakeholder would
actually need. See "Honest scope" below for exactly what is real vs.
simulated.

## What it demonstrates

- **Hands-on DS work directly on data.** A real ETL step (download, unzip,
  join), documented synthetic-data generation, feature engineering, model
  training/evaluation, and a business-question layer - all in one small,
  readable codebase, not a slide deck.
- **Real data, honestly labeled.** The movie catalog (9,742 titles, real
  genres) and 100,836 ratings are the real, public
  [MovieLens `ml-latest-small`](https://grouplens.org/datasets/movielens/)
  dataset (GroupLens Research), downloaded live over HTTPS with no API key.
  The subscriber base, viewing sessions, and churn label are synthetic and
  explicitly disclosed as such - see "Honest scope."
- **Translating a business question into DS work.** `src/insight_agent.py`
  takes a free-text business question and maps it to one of four supported
  pandas/DuckDB computations - it never lets an LLM invent a number.
  Unsupported questions get an explicit "I can't answer that" instead of a
  guess (the grounding guardrail).
- **A self-serve data product, not a notebook.** `app.py` is a Streamlit
  dashboard: KPI tiles, a churn-driver chart, a churn-by-segment chart, a
  genre-completion chart, and the business-question box - built for a
  business stakeholder, not a data scientist, to use unassisted.
- **GenAI applied with a safety/quality guardrail.** Claude
  (`langchain-anthropic`) is optional: if `ANTHROPIC_API_KEY` is set, it
  phrases answers in natural language; the numbers it's given are the only
  numbers it can use. Without a key, a deterministic template composer runs
  instead, so the whole demo works with zero API keys.
- **The same tools over MCP.** `src/mcp_server.py` exposes
  `churn_rate_by_segment`, `top_churn_drivers`, `completion_rate_by_genre`,
  `engagement_trend_over_time`, and `answer_business_question` via the
  official `mcp` SDK's `FastMCP`, so any MCP-compatible client can call them
  directly.
- **Enablement material for business users.** `PLAYBOOK.md` defines every
  metric and gives example questions in plain language; `PRODUCT_BRIEF.md`
  is a one-page internal spec (problem, users, KPIs, scope, governance) in
  the style of an actual data-product brief.

## Architecture

```
Real data                    Synthetic layer                    Model + analytics
┌──────────────────┐  HTTP  ┌─────────────────────────┐        ┌──────────────────────────┐
│ MovieLens          │──────▶│ scripts/build_data.py     │       │ src/features.py           │
│ ml-latest-small    │       │  synthetic subscribers +  │──────▶│  shared feature table     │
│ (real titles,      │       │  sessions + documented    │       │ src/model.py               │
│  genres, ratings)  │       │  rule + noise churn label │       │  LightGBM + SHAP drivers   │
└──────────────────┘        └─────────────────────────┘        └────────────┬─────────────┘
                                                                              │ models/metrics.json
                                                                              ▼
                                                                ┌──────────────────────────┐
                                                                │ src/insight_agent.py       │
                                                                │  rule-based router         │
                                                                │  → pandas/DuckDB compute    │
                                                                │  → grounded composer        │
                                                                │    (template, or Claude     │
                                                                │     if ANTHROPIC_API_KEY)   │
                                                                └───────┬──────────┬────────┘
                                                                        │          │
                                                            ┌───────────▼─┐   ┌────▼───────────┐
                                                            │ app.py       │   │ src/mcp_server   │
                                                            │ (Streamlit,  │   │ (FastMCP, same   │
                                                            │  self-serve  │   │  5 tools, any    │
                                                            │  dashboard)  │   │  MCP client)     │
                                                            └──────────────┘   └────────────────┘
```

## Run it

```bash
make setup   # venv + pip install -r requirements.txt
make data    # downloads real MovieLens data, generates synthetic layer -> data/*.csv
make train   # trains the churn model -> models/churn_model.pkl, models/metrics.json
make app     # streamlit run app.py -> http://localhost:8501
make test    # 20 tests: data schema, features, model, grounding, MCP

# Inspect/run the MCP server directly (needs `mcp[cli]`, see requirements.txt)
.venv/bin/mcp dev src/mcp_server.py       # opens the MCP Inspector UI
# or, for stdio:
.venv/bin/python -m src.mcp_server
```

No API key is required for any of the above. Set `ANTHROPIC_API_KEY` before
`make app` to have Claude phrase business-question answers in natural
language via `langchain-anthropic`; the numbers behind the answer are
identical either way (see `src/insight_agent.py`).

## Design decisions

- **One feature definition, two consumers.** `src/features.py`'s
  `compute_feature_table` is imported by both `scripts/build_data.py` (to
  derive the synthetic churn label) and `src/model.py` (to build the
  training matrix), so the label-generation logic and the model-training
  logic can never silently drift apart - there is exactly one definition of
  "engagement" in the codebase.
- **SHAP at training time only.** `src/model.py` computes SHAP
  (`TreeExplainer`) once and writes the ranked drivers into
  `models/metrics.json`. The Streamlit app and insight agent read that
  precomputed file and never import `shap` themselves, so the app stays
  fast to start even though SHAP's own import cost is non-trivial. See
  `src/model.py`'s docstring for why SHAP was used over
  `feature_importances_` here.
- **A rule-based router, not a general chat agent.** `insight_agent.py`
  supports exactly four analyses. This is a deliberate constraint: it makes
  every possible answer traceable to a real computation and makes the
  grounding property unit-testable (`tests/test_insight_agent.py` asserts
  the agent's numbers equal numbers computed directly via pandas on the
  same data - the single most important test in this repo).
- **The key-free composer is the default, not a fallback bolted on later.**
  Both `insight_agent.py`'s composer and the sibling design pattern it
  mirrors (`clinical-trials-genai-assistant/backend/app/planner.py`) treat
  the deterministic template as the primary path and Claude as an optional
  phrasing layer over the same numbers - never a second source of numbers.

## Honest scope

- **Real:** the MovieLens `ml-latest-small` catalog and ratings (9,742
  movies, real titles/genres, 100,836 real user ratings; GroupLens
  Research - see `data/movielens/README.txt` for the dataset's own citation
  requirement). These raw files are committed to this repo (not
  regenerated at runtime): the GroupLens license permits redistribution,
  and committing them removes a live network dependency from the deployed
  app - see the note on `ensure_movielens_catalog()` in
  `scripts/build_data.py` for why that mattered in practice. `make data`
  still re-downloads them fresh if you want to verify that independently.
- **Synthetic, and explicitly disclosed:** the subscriber base
  (`data/subscribers.csv`), viewing sessions (`data/sessions.csv`), and the
  churn label. All are generated by `scripts/build_data.py` with a fixed
  random seed. The churn label is produced by a documented rule-based-plus-
  noise process (see that script's module docstring): a latent per-
  subscriber "engagement propensity" drives session frequency, completion
  rate, and recency; those *observable* engagement features (not the latent
  value) are then combined with fixed, documented weights plus Gaussian
  noise into a churn probability, from which the label is drawn. This makes
  the churn signal realistic and learnable, but it is not real subscriber
  behavior - no production subscription data is used or claimed anywhere in
  this repo.
- **No fabricated business impact.** This repo reports model evaluation
  metrics computed on a held-out test split (ROC-AUC, precision, recall) and
  nothing else - no "% reduction in churn," no user-facing outcome numbers.
  It is a solo demo project, not a deployed product with measured impact.
- **The churn label is partially circular by construction.** Because the
  label is a noisy function of the same engagement features the model
  trains on, the model's ROC-AUC (~0.87 on the held-out split - see
  `models/metrics.json` after `make train`) reflects how well a documented
  synthetic-generation process can be recovered, not a claim about
  predicting real-world subscriber churn. The Gaussian noise term in the
  label-generation rule was specifically included so the task isn't
  trivially/perfectly separable (which would misrepresent what a churn
  model can do in practice).
- **`engagement_trend_over_time` is an engagement trend, not a churn
  trend.** This dataset stores churn as one static label per subscriber,
  not a time-indexed event, so no analysis here can measure a churn "spike"
  over a period - the insight agent says so explicitly rather than
  fabricating a trend (see `src/insight_agent.py`'s router comments and
  `tests/test_insight_agent.py`).
- **The MovieLens download can fail without network access.** If it does,
  `scripts/build_data.py` falls back to a small locally-generated
  placeholder catalog with the same schema, clearly labeled
  `[FALLBACK-PLACEHOLDER]` in every title - this path is logged loudly and
  is a last resort, not the intended data source.
- **Not monitored, not production.** No retraining schedule, drift
  monitoring, or PII handling is implemented - see `PRODUCT_BRIEF.md`'s
  "Governance & monitoring" section for what a production version would
  need and why this repo intentionally doesn't build it.

## Project layout

```
scripts/
  build_data.py       downloads real MovieLens data, generates the synthetic
                       subscriber/session layer, derives the churn label
src/
  features.py         shared engagement feature table (single source of truth)
  model.py             LightGBM churn model + SHAP top-driver attribution
  insight_agent.py     rule-based router + pandas/DuckDB analyses + composer
  mcp_server.py        FastMCP server exposing the same analyses
app.py                Streamlit self-serve dashboard + business-question box
tests/                20 tests: data schema, features, model, grounding, MCP
data/                 subscribers.csv, sessions.csv (generated); movielens/
                       (real MovieLens data, committed - re-verify with `make data`)
models/               churn_model.pkl, metrics.json (generated by `make train`)
PLAYBOOK.md           business-user guide to the metrics and example questions
PRODUCT_BRIEF.md      one-page internal product spec (scope, KPIs, governance)
```
