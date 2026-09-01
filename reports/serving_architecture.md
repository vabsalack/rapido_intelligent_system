# Serving the Rapido Intelligent System — Architecture & Teaching Guide

This document explains the serving layer we just built: how the four trained
pipelines become a live HTTP API, how that API is exposed **from a marimo
notebook running as a script**, and how a second marimo notebook consumes it as
a prediction dashboard.

---

## 1. Big picture

```
 ┌─────────────────────────── notebooks/                              ┌─────────────────────────── apps/
 │ 0.03–0.06  modelling notebooks                                     │ 0.07 API-server notebook   │ 0.08 dashboard notebook
 │   train 4 pipelines ─────────────────────────────┐                 │   (FastAPI as script)      │   (marimo frontend)
 └─────────────────────────────────────────────────┼                 └────────────┬───────────────┘                │
                                                    ▼                               │    POST /predict/{id}               │  GET /models
                                          ┌─────────────────┐                       │                                   │
                                          │ models/*.joblib │  (saved + _meta.json) │                                   │
                                          │ 4 pipelines     │                       │                                   ▼
                                          └─────────────────┘   ───────────────────▶ rapido_intelligent_system/api/app.py
                                                                  loads them into   (FastAPI app + registry + schemas)
                                                                  a server            │
                                                                                      ▼  httpx
                                                                        reports/predictions (deferred — not part of this build)
```

The two new entry points are both **marimo notebooks** (`notebooks/0.07` and
`notebooks/0.08`), but they play opposite roles:

| File | Role | Engine | Runs in |
|---|---|---|---|
| `0.07-kittu-api-server.py` | **Server** | FastAPI inside a marimo notebook | script mode (and interactive) |
| `0.08-kittu-predict-dashboard.py` | **Frontend** | marimo UI + `httpx` | interactive (and script mode) |

---

## 2. The four served models

| Model id | Task | Ended up with | Headline metric |
|---|---|---|---|
| `booking_status` | multiclass (Cancelled / Completed / Incomplete) | Logistic Regression (softmax) | F1-weighted 0.615, acc 0.678 |
| `booking_value` | regression (fare in ₹) | HistGradientBoosting | R² 0.9968, RMSE ₹11.76 |
| `customer_cancel_flag` | binary (0/1) | SGDClassifier | F1 1.0* |
| `driver_delay_flag` | binary (0/1) | HistGradientBoosting | CV F1 0.781, test F1 0.850 |

\* The customer-cancel label is deterministic in the current synthetic data, so
the perfect F1 reflects the data, not model magic.

Each model is a single serialised object: **a scikit-learn `Pipeline`** saved to
`models/<id>_model.joblib` with a `models/<id>_model_meta.json` sidecar holding
tuning/metric metadata. The pipeline **already contains the whole preprocessing
chain** — that is the single most important design decision in the whole build.

---

## 3. Key idea: the pipeline carries its own preprocessing

A request to the API sends only the **raw** feature values. The pipeline does
the rest internally:

```
raw row  ─▶ (1) engineer_*_features   (create derived features)
         ─▶ (2) impute missing values
         ─▶ (3) scale numerics + one-hot encode categories (handle_unknown="ignore")
         ─▶ (4) learner.predict / predict_proba
```

This is why a frontend can send, e.g. just `{"ride_distance_km": 12.0,
"base_fare": 55.0, ...}` for booking status — it never needs to replicate the
feature-engineering code. As long as the HTTP body's keys match the feature
catalog, the model re-derives everything at inference time. Because the
`OneHotEncoder` is configured with `handle_unknown="ignore"`, even a brand-new
category value the model never saw during training won't crash the pipeline.

Consequence: **the feature catalog is a single source of truth.** It lives as
constants in `rapido_intelligent_system/modeling_mnb.py`
(`BOOKING_FEATURES`, `BOOKING_NUM_FEATURES`, `BOOKING_CAT_FEATURES`,
`BOOKING_CATEGORIES`, and the mirrors for value / customer / driver). Both the
server (to build request schemas) and the dashboard (to build input forms)
read from that one definition, so the UI can never drift from the model.

---

## 4. The server — `rapido_intelligent_system/api/app.py`

A plain FastAPI module. Highlights:

1. **Registry** — a `ModelSpec` dataclass holds `id, task, target, pipeline,
   features, num_features, cat_features, categories, class_labels, meta`. The
   `REGISTRY` dict is built once at import time by `load_model(...)` on each of
   the four joblib files plus its `_meta.json`.

2. **Dynamic request schemas** — instead of writing four Pydantic models by
   hand, `_request_model(spec)` calls `pydantic.create_model` to generate one
   per model from its feature catalog:
   ```python
   fields[feature] = float   # for each numeric feature
   fields[feature] = str     # for each categorical feature
   ```
   `REQUEST_MODELS = {spec.id: _request_model(spec) ...}`. This guarantees the
   OpenAPI/Swagger payload matches exactly what each pipeline expects — and if
   a feature is added to the catalog, the API schema updates automatically.

3. **One shared inference helper** — `predict_one(spec, payload_data)`:
   - wraps the JSON body in a one-row **polars** `DataFrame`
     (`pl.DataFrame([payload_data]).select(spec.features)`), reusing the same
     dataframe library the modelling notebooks use;
   - calls `spec.pipeline.predict(frame)` (and `predict_proba` where available);
   - returns a JSON-safe dict (`model`, `task`, `prediction`, `label`,
     `probabilities` as `[{class: prob}, ...]`).

4. **Routes**
   - `GET /`        — service info + list of predict endpoints
   - `GET /health`  — liveness + which models loaded
   - `GET /models`  — full typed schema per model (learner, headline metric,
     feature list with type + allowed categories, class labels). **This is the
     contract the dashboard consumes to build its forms.**
   - `POST /predict/{booking_status|booking_value|customer_cancel_flag|driver_delay_flag}`
     — one endpoint per model, explicit path operation, Pydantic-validated body,
     response filtered with `response_model_exclude_none=True`.
   - `GET /predict/{model_id}/example` — a sample payload for manual testing.

5. **CORS** — allowed origins `["*"]` is a dev setting: the dashboard runs on a
   different port, so cross-origin requests must be permitted.

---

## 5. The clever bit — a marimo notebook that *is* the server

`notebooks/0.07-kittu-api-server.py` is not a "regular" FastAPI file — it's a
marimo notebook that can be executed **as a plain Python script**. This gives
one file that works in two modes:

```python
with app.setup:
    import uvicorn
    import marimo as mo
    from rapido_intelligent_system.api.runner import ServerHandle, serve_blocking
    from rapido_intelligent_system.api.app import app as api_app

is_script_mode = mo.app_meta().mode == "script"   # True when run with `python file.py`

@app.cell
def _(host, is_script_mode, mo, port, serve_blocking):
    if is_script_mode:
        serve_blocking(api_app, host, port)       # blocks → real server until Ctrl+C
```

- **Script mode** — `python notebooks/0.07-kittu-api-server.py` loads the
  models and starts uvicorn on `API_HOST:API_PORT` (default `0.0.0.0:8000`),
  blocking until you stop it. This is the production/dev entry point.
- **Interactive mode** — `marimo edit notebooks/0.07-kittu-api-server.py` shows
  a toggle (`mo.ui.switch`) that starts/stops the server on a daemon thread via
  the `ServerHandle` helper in `rapido_intelligent_system/api/runner.py`
  (uvicorn in a background thread, with `start()`/`stop()`/`is_running()`).

`mo.app_meta().mode == "script"` is the marimo API that tells the notebook which
mode it's in, so the same cell decides between "block forever, start the
server" and "show a start/stop control".

> Marimo gotcha we hit while building this: in my early version I used
> `mo.state(...)[0].value` in the script-exactly path. In headless/script mode a
> `mo.state` wrapper is inert and has no `.value`; the fix was to drive the
> server from a `mo.ui.switch(on_change=...)` instead. The lesson: don't depend
> on `.value` of state in a notebook that must also run headlessly.

---

## 6. The frontend — `notebooks/0.08-kittu-predict-dashboard.py`

A marimo dashboard whose whole job is to call the API over HTTP and render the
results nicely. Flow per model tab:

1. `GET /models` → typed schema (features, types, categories, class labels).
2. Build one **input form** per model from that schema (`mo.ui.form` of a
   `mo.ui.dictionary`). Numeric features → `mo.ui.number` (pre-filled with a
   sensible default); categorical → `mo.ui.dropdown` from the schema's allowed
   options. Because the forms come from the schema, they stay in lockstep with
   the models.
3. When the user presses **Predict**, the form's values are extracted and sent
   with `POST /predict/{model_id}` via **httpx** (`POST / → ...`).
4. The response is rendered as a markdown table (prediction + probabilities)
   plus a **plain-language interpretation** generated from the value's range —
   e.g. for booking value:
   - `₹< 30` → "very affordable short hop"
   - `₹30–60` → "affordable mid-range fare"
   - `₹60–100` → "moderate fare; standard city cab"
   - `₹100–150` → "above-average; long ride or surge"
   - `₹> 150`  → "premium; check surge / distance"
   and for the classifiers, driven by the positive-class probability (green →
   amber → red banding).

The dashboard is split into a handful of well-separated marimo cells
(constants + helpers, header, URL + health, schema-fetch + forms, predict +
interpret, tabs + guide) with **private underscore-prefixed loop locals**
(`_mid`) so the notebooks stays clean under marimo's cross-cell uniqueness
rule. It also degrades gracefully: if no server is running, it logs a warning
instead of crashing.

---

## 7. Why this architecture (what we pushed down the stack)

1. **Single source of truth for features.** One catalog in
   `modeling_mnb.py` → used by training, the API schemas, and the UI forms. No
   duplicated lists to keep in sync.
2. **Inference logic lives inside the pipeline, not the server.** The FastAPI
   layer is a thin adapter: wrap a row, call `predict`, serialise. Nothing to
   re-implement at deploy time.
3. **Schemas are generated, not hand-written.** `create_model` + the catalog
   means the contract always fits the model.
4. **Notebook-as-server.** Reusing the modelling/serving notebook style keeps
   the whole project reproducible and gives an on-ramp from EDA to deployment
   with the same tooling.
5. **Frontend decoupled via HTTP.** The dashboard only needs the OpenAPI schema
   (`/models`) — it never imports the pipelines. That keeps the UI lightweight
   and means the server can later move to GPU/triton/containers without touching
   the dashboard.

---

## 8. Running it

```bash
# Terminal A — start the API server (script mode → real uvicorn server)
python notebooks/0.07-kittu-api-server.py          # API_HOST/API_PORT env overrides
# or, interactive server with start/stop toggle:
marimo edit notebooks/0.07-kittu-api-server.py

# Terminal B — the prediction dashboard
marimo edit notebooks/0.08-kittu-predict-dashboard.py
```

Then everything behind the API can be exercised directly too:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/models
curl -XPOST http://127.0.0.1:8000/predict/booking_value \
     -H 'content-type: application/json' \
     -d '{"ride_distance_km": 10.0, "estimated_ride_time_min": 20.0, ...}'
# interactive API browser at http://127.0.0.1:8000/docs
```

Notes:
- The server entry point is registered for the FastAPI CLI in `pyproject.toml`:
  `[tool.fastapi] entrypoint = "rapido_intelligent_system.api.app:app"`, so
  `fastapi run rapido_intelligent_system/api/app.py` works too. That CLI needs
  `fastapi[standard]` installed (`uv add fastapi[standard]`), not bare `fastapi`.
- Run from the repo root so the `models/` (derived from `app.py`'s location) and
  `data/` paths resolve.
- Validated: all four endpoints return 200 with sane outputs, `/models` schema
  loads, `marimo export html` on 0.08 → no tracebacks, `marimo check` clean,
  and `ruff check rapido_intelligent_system/api/` passes.