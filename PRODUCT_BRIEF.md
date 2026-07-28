# Product Brief: Subscriber Engagement Data Product

**Status:** demo / MVP, solo-built, not deployed. **Owner:** single data
scientist (this repo). **Audience:** internal - engineering/DS readers and
business stakeholders evaluating whether the approach is worth building on.

## Problem statement

Business stakeholders (retention, content, and product teams) need two
things a raw data warehouse doesn't give them: (1) a model that flags which
subscribers are at risk of churning and why, and (2) a way to ask
engagement questions in plain language and get a trustworthy, numeric
answer - without waiting on a DS team for every ad-hoc query or risking an
LLM inventing a plausible-sounding but wrong number. This project
demonstrates a small, honest version of both, end to end, on real movie
metadata plus a synthetic subscriber layer that stands in for a production
subscriber/engagement database.

## Target users

- **Retention / lifecycle marketing**: wants to know which segments churn
  more and what's driving it, without writing SQL.
- **Content strategy**: wants completion-rate-by-genre style answers to
  inform acquisition/licensing conversations.
- **A DS/analytics team evaluating the pattern**: wants to see whether a
  constrained, grounded question-answering layer (vs. an open-ended LLM
  chat interface) is viable for this kind of stakeholder self-service.

## Core KPIs surfaced

- Overall and segment-level churn rate (by plan tier, by region)
- Average content completion rate (overall, by genre)
- Top churn-driving engagement signals (from the trained model's SHAP
  attribution)
- Monthly engagement trend (session volume, completion rate)
- Model quality: ROC-AUC, precision, recall on a held-out split

## MVP scope (what is actually in this repo)

- A one-command data pipeline (`scripts/build_data.py`) that pulls a real
  public dataset (MovieLens) and layers a synthetic, documented
  subscriber/session/churn dataset on top of it.
- A trained gradient-boosted churn classifier (LightGBM) with a held-out
  evaluation report and SHAP-based driver attribution.
- A constrained, rule-routed business-question layer over pandas/DuckDB,
  with a hard grounding guardrail (unsupported questions are declined, not
  guessed at) and an optional Claude phrasing layer over the same numbers.
- The same analyses exposed as an MCP server for any MCP-compatible client.
- A Streamlit dashboard usable by a non-technical stakeholder, plus a
  business-user playbook (`PLAYBOOK.md`) defining every metric.
- A test suite (20 tests) that specifically verifies the question-answering
  layer's numbers match direct pandas computations - the property that
  makes "no hallucination" a checked fact, not a claim.

## Explicitly out of scope / roadmap

- **Real subscriber/engagement data.** This repo uses a synthetic layer by
  design (see README "Honest scope"); a real version would connect to an
  actual subscription/viewing-events warehouse.
- **Time-indexed churn events.** Churn here is one static label per
  subscriber. A production version would need a churn *event* (a
  cancellation date/timestamp) to support real trend and cohort analysis -
  this is called out explicitly wherever the current dataset can't support
  it (e.g. `engagement_trend_over_time` is an engagement trend, not a churn
  trend).
- **Open-ended chat / arbitrary SQL generation.** The question-answering
  layer intentionally supports four analyses, not a general text-to-SQL
  agent, to keep every answer traceable and testable. Expanding the
  supported-analysis set is the natural next increment, not a rewrite.
- **Multi-user auth, rate limiting, hosting.** This is a local
  `streamlit run` demo, not a deployed multi-tenant service.
- **Model serving infrastructure.** The model is a local pickle file loaded
  by the app process; no serving endpoint, versioning registry, or
  A/B-testing harness is implemented.

## Governance & monitoring considerations (for a production version)

None of the following are implemented in this repo; they are named here
because a real "Data Product Scientist" scope includes thinking about them,
not because this demo needs them to run:

- **Retraining cadence.** A production churn model would need a defined
  retrain schedule (e.g. monthly) tied to how quickly subscriber behavior
  and content catalog shift, plus a champion/challenger comparison before
  promoting a new model.
- **Drift monitoring.** Feature-distribution drift (e.g. sessions-per-week
  distribution shifting after a UI change) and label drift (real-world
  churn definitions changing, e.g. a new cancellation flow) would both need
  monitoring; this demo has no live traffic to monitor.
- **Data freshness / pipeline SLAs.** A production version needs defined
  freshness guarantees on the upstream events feeding engagement features,
  with alerting on staleness - this demo's data is generated once, on
  demand, from a fixed reference date.
- **PII handling.** Subscriber-level viewing and account data is sensitive;
  a production pipeline would need access controls, anonymization/
  pseudonymization of subscriber IDs before they reach an analytics or LLM
  layer, and a data-retention policy. This demo's subscriber IDs and all
  underlying data are synthetic, so no real PII exists here, but a real
  implementation must not skip this.
- **LLM safety/quality for the optional Claude path.** The grounding
  guardrail (numbers only from computed data, explicit decline on
  unsupported questions) is the main safety mechanism implemented here. A
  production deployment would add: logging/eval of phrased answers against
  the grounded numbers they were given (to catch subtle rephrasing errors),
  a policy on what business questions are in-bounds for a stakeholder-
  facing tool, and human review before any Claude-phrased answer reaches an
  external audience.
