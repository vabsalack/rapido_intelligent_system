import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import os
    import marimo as mo
    import polars as pl
    from loguru import logger
    from rapido_intelligent_system.api.app import app as api_app
    from rapido_intelligent_system.api.app import REGISTRY, REQUEST_MODELS
    from rapido_intelligent_system.api.runner import ServerHandle, serve_blocking


@app.cell
def _():
    mo.md(r"""
    # 0.07 — Model API server (marimo notebook as a FastAPI server)

    This notebook **is the server**: it loads the four saved pipelines from
    `models/` and serves them over HTTP with FastAPI + uvicorn.

    ### Two ways to run it

    | Mode | Command | What happens |
    |---|---|---|
    | **Script** (production/dev) | `python notebooks/0.07-kittu-api-server.py` | loads the models, starts uvicorn on `API_HOST:API_PORT` (default `0.0.0.0:8000`), blocks until Ctrl+C |
    | **Interactive** (inspect + try) | `marimo edit notebooks/0.07-kittu-api-server.py` | registry + endpoints below, start/stop buttons host a dev server |
    | Alternate | `fastapi run` (entrypoint in `pyproject.toml`) | serves `rapido_intelligent_system.api.app:app` directly |

    Once running, open `http://localhost:8000/docs` — the interactive Swagger UI
    lets you POST sample payloads and see predictions immediately.
    """)
    return


@app.cell
def _(REGISTRY):
    registry_table = pl.DataFrame(
        [
            {
                "model_id": spec.id,
                "task": spec.task,
                "target": spec.target,
                "raw_features": len(spec.features),
                "learner": spec.meta.get("model"),
                "class_labels": ", ".join(spec.class_labels) if spec.class_labels else "—",
            }
            for spec in REGISTRY.values()
        ]
    )
    registry_table
    return (registry_table,)


@app.cell
def _():
    mo.md(r"""
    ### What the API exposes

    `POST / predict/<model>` accepts a JSON body with the **raw feature values**
    (numbers for numeric columns, strings for categories). Every pipeline carries
    its own preprocessing (feature engineering → impute → scale / one-hot), so the
    request payload needs only what a human would type into a form — no
    transformations on the client side.

    | Endpoint | Task | Returns |
    |---|---|---|
    | `POST /predict/booking_status` | multiclass (Cancelled/Completed/Incomplete) | label + class probabilities |
    | `POST /predict/booking_value` | regression (₹) | predicted fare |
    | `POST /predict/customer_cancel_flag` | binary 0/1 | label + probability |
    | `POST /predict/driver_delay_flag` | binary 0/1 | label + probability |
    | `GET /models` | introspection | full typed schema per model (fields, allowed categories) |
    | `GET /health` | liveness | model IDs loaded |

    The **frontend dashboard** (notebook 0.08) will call these `POST / predict/…`
    endpoints with `httpx` and render the responses in marimo UI.

    ### Architecture

    ```
    marimo dashboard (notebook 0.08, future)     marimo API notebook (0.07)
    [forms / sliders / dropdowns]  ──HTTP──▶  [FastAPI app]  ──▶  [4 saved .joblib pipelines]
                                                  │                      │
                                            FastAPI app, schemas,       engineering + scaling
                                            registry: api/app.py        + one-hot happen here
    ```
    """)
    return


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    is_script_mode
    return (is_script_mode,)


@app.cell
def _(mo):
    default_host = os.environ.get("API_HOST", "0.0.0.0")
    default_port = int(os.environ.get("API_PORT", "8000"))
    bind_host = mo.ui.text(value=default_host, label="Bind host")
    bind_port = mo.ui.number(1, 65535, value=default_port, step=1, label="Bind port")
    mo.hstack([bind_host, bind_port])
    return bind_host, bind_port


@app.cell
def _(bind_host, bind_port):
    host = os.environ.get("API_HOST", bind_host.value)
    port = int(os.environ.get("API_PORT", bind_port.value))
    return host, port


@app.cell
def _(ServerHandle, api_app, host, mo, port):
    _dev_server = ServerHandle(api_app)

    def _on_run_toggle(should_run: bool) -> None:
        if should_run:
            _dev_server.start(host, port)
        else:
            _dev_server.stop()

    run_toggle = mo.ui.switch(
        value=False,
        label=f"Dev API server at {host}:{port}",
        on_change=_on_run_toggle,
    )
    run_toggle
    return (run_toggle,)


@app.cell
def _(host, is_script_mode, mo, port, serve_blocking, api_app):
    if is_script_mode:
        _ = mo.md(f"Script mode — starting FastAPI server at `http://{host}:{port}` (Ctrl+C to stop).")
        serve_blocking(api_app, host, port)
    else:
        _ = mo.md("Use the buttons above to start/stop a dev server from this notebook.")
    return


@app.cell
def _(REGISTRY, REQUEST_MODELS, api_app, is_script_mode):
    if not is_script_mode:
        _schema_lines = []
        for _id, _spec in REGISTRY.items():
            _fields = []
            for _f in _spec.features:
                _kind = "number" if _f in _spec.num_features else "category"
                _opt = _spec.categories.get(_f)
                _fields.append(
                    f"`{_f}` ({_kind}" + (f", choices {_opt}" if _opt else "") + ")"
                )
            _schema_lines.append(f"- **{_id}** — " + ", ".join(_fields))
        _ = mo.md(
            "\n### Live API request schema\n" + "\n".join(_schema_lines)
        )
    else:
        _ = None
    return


if __name__ == "__main__":
    app.run()