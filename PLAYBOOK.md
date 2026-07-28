# Playbook: Subscriber Engagement Data Product

A guide for business stakeholders using the dashboard (`app.py`, run with
`make app`). No coding or data background needed - this page explains what
each number means, how to read the charts, and what you can ask the
assistant.

> Heads up: the subscriber, viewing-session, and churn data behind this
> dashboard are **synthetic** (generated for this demo, not real customer
> data). The movie catalog is real (MovieLens). See the main README's
> "Honest scope" section for the full disclosure. Treat every number here
> as illustrative of what the tool can do, not as a real business result.

## The metrics, defined

**Churn rate**
The share of subscribers labeled as churned, out of all subscribers in a
group. Shown as a percentage (e.g. 44% churn rate means 44 out of every 100
subscribers in that group are labeled churned). In this demo, churn is a
single snapshot label per subscriber - it does not tell you *when* someone
churned, only whether they are labeled as having churned.

**Completion rate**
The average share of a title a subscriber watches per viewing session,
averaged across sessions (e.g. 65% completion means, on average, people
watch about two-thirds of what they start). A useful proxy for whether
content is actually landing with viewers, as opposed to just being started.

**Engagement score (the model's inputs)**
The churn model doesn't use one single "engagement score" - it uses four
underlying signals, shown together on the "Top churn drivers" chart:
- **Sessions per week** - how often someone watches something, relative to
  how long they've been a subscriber.
- **Avg. completion rate** - see above.
- **Recency** - how many days since someone's last viewing session. Higher
  recency (more days since they last watched) means more disengaged.
- **Genre diversity** - how many distinct genres someone has watched. A
  proxy for how broadly someone uses the catalog versus watching one narrow
  slice of it.

**Recency**
Days since a subscriber's most recent viewing session, as of the dataset's
fixed reference date (2025-06-30 in this demo). A subscriber who has never
watched anything gets a recency equal to their full tenure - they've been
"not watching" the whole time they've had an account.

## How to read the churn-driver chart

The "Top churn drivers" chart ranks the four engagement signals above (plus
tenure) by how strongly they push the model's predictions toward "this
subscriber will churn." Longer bars = the model relies on that signal more
heavily when predicting churn. This is not a causal claim ("low completion
rate *causes* churn") - it's an attribution of the trained model's behavior,
computed with SHAP (a standard model-explanation technique). Use it to spot
which signals are worth watching, not as proof of a causal mechanism.

## How to read the churn-by-segment and genre charts

- **Churn rate by segment**: pick "Plan tier" or "Region" from the dropdown
  to see churn rate broken out by group, sorted highest to lowest. Compare
  bar heights across groups - a taller bar means a higher churn rate in
  that group, in this dataset.
- **Completion rate by genre**: which movie genres get watched most
  completely, on average, across all sessions in this dataset. Genres with
  fewer than 20 sessions are excluded so the average isn't based on too
  little data.

## Asking the assistant a question

Type a question in plain English into the "Ask a business question" box.
The assistant only answers using four kinds of analysis - if your question
doesn't fit one of them, it will tell you so directly instead of guessing.
Every number in its answer comes from a real calculation on the dashboard's
own data (visible under "Show supporting numbers").

Example questions to try:

1. "What is the churn rate by plan tier?"
2. "What is the churn rate by region?"
3. "What are the top churn drivers?"
4. "Which genre has the highest completion rate?"
5. "What is the engagement trend over the last few months?"
6. "Why did churn spike for Premium subscribers last quarter?" - a good
   example of the assistant's honesty guardrail: this dataset only stores a
   single churn label per subscriber (not a month-by-month history), so the
   assistant will tell you that a "spike" can't be measured here and give
   you the closest real number it has instead (current churn rate by plan
   tier), rather than inventing an explanation.

If you ask something outside these four areas (e.g. "what's the weather
today?"), the assistant will say plainly that it can't answer that with the
analyses it supports - it will never fabricate a number to fill the gap.
