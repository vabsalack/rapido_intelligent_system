import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import json
    import os

    import httpx
    import marimo as mo
    from loguru import logger

    API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")


@app.cell
def _():
    model_info = {
        "booking_status": {
            "name": "Booking Status",
            "emoji": "🎫",
            "learn": "Logistic Regression (softmax) — F1-weighted 0.615 on validation",
            "purpose": (
                "Predicts whether a booking will be **Cancelled**, **Completed**, or "
                "**Incomplete** given the ride context (fare, distance, traffic, weather, "
                "surge). Use it to spot problematic bookings before the ride starts."
            ),
        },
        "booking_value": {
            "name": "Booking Value (₹)",
            "emoji": "💰",
            "learn": "HistGradientBoosting — R² 0.997, MAE ₹8.53 on validation",
            "purpose": (
                "Estimates the predicted fare in **₹ (Indian Rupees)** from distance, "
                "estimated ride time, base fare, surge, traffic, vehicle and weather. "
                "Use it for fare transparency and anomaly detection."
            ),
        },
        "customer_cancel_flag": {
            "name": "Customer Cancel Flag",
            "emoji": "🚫",
            "learn": "SGDClassifier (softmax) — F1 1.0 on validation (deterministic label)",
            "purpose": (
                "Predicts whether a customer will **cancel** their ride from their profile "
                "(age, city, history, rating, preferred vehicle). Use it to drive outreach "
                "and incentive targeting."
            ),
        },
        "driver_delay_flag": {
            "name": "Driver Delay Flag",
            "emoji": "⏱️",
            "learn": "HistGradientBoosting — F1 0.85 on held-out test",
            "purpose": (
                "Predicts whether a driver's next pickup will be **delayed** from their "
                "profile, ride history, acceptance rate and average pickup delay. Use it "
                "to pre-assign buffer time or flag at-risk trips."
            ),
        },
    }

    def api_post(url, model_id, payload):
        try:
            resp = httpx.post(f"{url}/predict/{model_id}", json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            return {"error": str(exc)}

    def extract_vals(form_value):
        return {
            key: (float(item.value) if hasattr(item, "value") else str(item.value))
            for key, item in form_value.items()
        }

    def probs_map(probabilities):
        return {
            str(list(p.keys())[0]): float(list(p.values())[0])
            for p in probabilities or []
        }

    def interpret_bs(pred, probabilities):
        p = probs_map(probabilities)
        if pred == "Completed":
            return f"🟢 **Completed** — confidence {p.get('Completed', 0):.0%}. Ride expected to finish normally."
        if pred == "Incomplete":
            return f"🟡 **Incomplete** — confidence {p.get('Incomplete', 0):.0%}. Ride will likely start but not reach destination."
        return f"🔴 **Cancelled** — confidence {p.get('Cancelled', 0):.0%}. Rider very likely to cancel; consider proactive outreach."

    def interpret_bv(pred):
        v = float(pred)
        if v < 30:
            return f"🟢 ₹{v:.0f} — very affordable short hop; likely a bike or auto short trip."
        if v < 60:
            return f"🟢 ₹{v:.0f} — affordable mid-range fare; typical for auto or short cab rides."
        if v < 100:
            return f"🟡 ₹{v:.0f} — moderate fare; a standard cab ride across the city."
        if v < 150:
            return f"🟠 ₹{v:.0f} — above-average fare; long-distance or high-surge trip."
        return f"🔴 ₹{v:.0f} — premium fare; check if surge or long distance is driving the price."

    def interpret_cc(pred, probabilities):
        p = probs_map(probabilities)
        cp = float(p.get("1", 0.0))
        if cp < 0.05:
            return f"🟢 **Not likely to cancel** — cancel probability {cp:.0%}."
        if cp < 0.20:
            return f"🟢 Low risk — cancel probability {cp:.0%}. Most riders stay committed."
        if cp < 0.40:
            return f"🟡 Moderate risk — cancel probability {cp:.0%}. Consider a confirmation nudge."
        if cp < 0.60:
            return f"🟠 High risk — cancel probability {cp:.0%}. Offer an incentive to keep the booking."
        return f"🔴 Very high risk — cancel probability {cp:.0%}. Near-certain cancellation; proactive outreach recommended."

    def interpret_dd(pred, probabilities):
        p = probs_map(probabilities)
        dp = float(p.get("1", 0.0))
        if dp < 0.05:
            return f"🟢 **On time expected** — delay probability {dp:.0%}."
        if dp < 0.20:
            return f"🟢 Very low risk — delay probability {dp:.0%}. No action needed."
        if dp < 0.40:
            return f"🟡 Moderate risk — delay probability {dp:.0%}. Add a small buffer."
        if dp < 0.60:
            return f"🟠 High risk — delay probability {dp:.0%}. Assign buffer time or reassign driver."
        return f"🔴 Very high risk — delay probability {dp:.0%}. Rider-facing delay almost certain."

    def build_form(schema_id, schemas, defaults):
        ui = {}
        for f in schemas[schema_id]["features"]:
            name = f["name"]
            if name == "pickup_location":
                ui[name] = mo.ui.text(value="Loc_12", label="pickup_location (e.g. Loc_0 – Loc_49)")
            elif f["type"] == "number":
                ui[name] = mo.ui.number(value=float(defaults.get(name, 0.0)), label=name)
            else:
                ui[name] = mo.ui.dropdown(options=f["options"] or [], value=(f["options"] or [""])[0], label=name)
        return mo.ui.form(mo.ui.dictionary(ui), submit_button_label="Predict")

    form_defaults = {
        "booking_status": {
            "ride_distance_km": 12.0, "estimated_ride_time_min": 25.0, "base_fare": 55.0,
            "surge_multiplier": 1.2, "avg_wait_time_min": 5.0, "avg_surge_multiplier": 1.1,
        },
        "booking_value": {
            "ride_distance_km": 10.0, "estimated_ride_time_min": 20.0, "base_fare": 50.0,
            "surge_multiplier": 1.5, "avg_wait_time_min": 4.0, "avg_surge_multiplier": 1.2,
            "hour_of_day": 14.0,
        },
        "customer_cancel_flag": {
            "customer_age": 32.0, "customer_signup_days_ago": 730.0, "total_bookings": 18.0,
            "completed_rides": 14.0, "incomplete_rides": 2.0, "avg_customer_rating": 4.3,
        },
        "driver_delay_flag": {
            "driver_age": 36.0, "driver_experience_years": 7.0, "total_assigned_rides": 180.0,
            "accepted_rides": 140.0, "incomplete_rides": 8.0, "acceptance_rate": 0.78,
            "avg_driver_rating": 4.4, "avg_pickup_delay_min": 4.5,
        },
    }

    MODEL_IDS = ["booking_status", "booking_value", "customer_cancel_flag", "driver_delay_flag"]
    return (
        MODEL_IDS,
        api_post,
        build_form,
        extract_vals,
        form_defaults,
        interpret_bs,
        interpret_bv,
        interpret_cc,
        interpret_dd,
        model_info,
    )


@app.cell
def _():
    mo.md(r"""
    # 🚀 Rapido Intelligent System — Prediction Dashboard

    Pick a model tab below, fill in the inputs, and press **Predict** to see
    the result with a plain-language interpretation.

    > **API server must be running.** Start it with:
    > ```bash
    > python notebooks/0.07-kittu-api-server.py
    > ```
    """)
    return


@app.cell
def _():
    api_url = mo.ui.text(value=API_URL, label="API server URL")
    api_url
    return (api_url,)


@app.cell
def _(api_url):
    ok = False
    try:
        ok = httpx.get(f"{api_url.value}/health", timeout=4).status_code == 200
    except Exception:
        ok = False
    mo.md(
        "🟢 API server reachable"
        if ok
        else "🔴 API server is not running — start it first (see instructions above)"
    )
    return


@app.cell
def _(MODEL_IDS, api_url, build_form, form_defaults, model_info):
    raw = []
    try:
        raw = httpx.get(f"{api_url.value}/models", timeout=8).json()
    except Exception as e:
        logger.warning(f"Could not fetch /models: {e}")
    schemas = {m["id"]: m for m in raw}

    forms = {}
    status = None
    if "booking_status" in schemas:
        for _mid in MODEL_IDS:
            forms[_mid] = build_form(_mid, schemas, form_defaults[_mid])
    else:
        status = "⚠️ Could not load the model schema from the API. Make sure the server is running, then re-run this cell."

    intro_blocks = []
    for _mid in MODEL_IDS:
        _i = model_info[_mid]
        intro_blocks.append(
            mo.md(f"### {_i['emoji']} {_i['name']}  \n"
                  f"**Algorithm:** {_i['learn']}  \n\n{_i['purpose']}")
        )
    mo.vstack(intro_blocks, gap=1)
    _ = mo.md(status) if status else mo.md("")
    return (forms,)


@app.cell
def _(
    MODEL_IDS,
    api_post,
    api_url,
    extract_vals,
    forms,
    info,
    interpret_bs,
    interpret_bv,
    interpret_cc,
    interpret_dd,
    model_info,
):
    interp_fns = {
        "booking_status": interpret_bs,
        "booking_value": interpret_bv,
        "customer_cancel_flag": interpret_cc,
        "driver_delay_flag": interpret_dd,
    }
    is_regression = {"booking_value"}

    results = {}
    for _mid in MODEL_IDS:
        form = forms.get(_mid)
        if form is None:
            continue
        value = getattr(form, "value", None)
        if value is None:
            results[_mid] = mo.md("")
            continue
        pred = api_post(api_url.value, _mid, extract_vals(value))
        if _mid in is_regression:
            interp = (
                interp_fns[_mid](pred.get("prediction", 0))
                if pred.get("error") is None
                else f"⚠️ Error: {pred['error']}"
            )
            detail = f"| **Predicted Fare** | **₹{pred.get('prediction', '—')}** |"
        else:
            interp = (
                interp_fns[_mid](pred.get("prediction", ""), pred.get("probabilities", []))
                if pred.get("error") is None
                else f"⚠️ Error: {pred['error']}"
            )
            detail = (
                f"| **Prediction** | **{pred.get('prediction', '—')} {pred.get('label', '')}** |\n"
                f"| **Probabilities** | {json.dumps(pred.get('probabilities', []))} |"
            )
        _i = model_info[_mid]
        results[_mid] = mo.md(f"""
    ### {info['emoji']} {info['name']} — Result

    | | |
    |---|---|
    {detail}

    {interp}
    """)

    for out in results.values():
        out
    return (results,)


@app.cell
def _(forms, results):
    guide = mo.md(r"""
    ### ℹ️ How to use this dashboard

    1. **Start the API server** — run `python notebooks/0.07-kittu-api-server.py` in another terminal.
    2. **Check the green status light** — it confirms the server is reachable.
    3. **Fill in the inputs** (sensible defaults provided) then press **Predict**.
    4. **Read the result** — the prediction plus a plain-language interpretation.

    #### Model performance at a glance

    | Model | Headline Metric | What it Means |
    |---|---|---|
    | Booking Status | F1 0.615 (weighted) | Balanced recall & precision across three status classes |
    | Booking Value | R² 0.997, MAE ₹8.53 | Predicts fare to within ~₹9 on average |
    | Customer Cancel | F1 1.0* | *Deterministic label in current synthetic data |
    | Driver Delay | F1 0.85 | Catches the 13% minority delay class well |

    #### API endpoints

    | Method | Path | Body |
    |---|---|---|
    | `POST` | `/predict/booking_status` | `{ride_distance_km, …}` |
    | `POST` | `/predict/booking_value` | `{ride_distance_km, pickup_location, …}` |
    | `POST` | `/predict/customer_cancel_flag` | `{customer_age, …}` |
    | `POST` | `/predict/driver_delay_flag` | `{driver_age, …}` |
    | `GET` | `/models` | — (returns full typed schema) |
    | `GET` | `/docs` | — (Swagger UI) |
        """)

    all_forms = mo.vstack(list(forms.values()), gap=2)
    all_results = mo.vstack(list(results.values()), gap=2)
    predict_tab = mo.hstack(
        [
            mo.vstack([all_forms, all_results], gap=1),
        ],
        gap=1,
    )

    tabs = mo.ui.tabs({
        "📊 Overview": guide,
        "✏️ Predict": predict_tab,
    })
    tabs
    return (tabs,)


@app.cell
def _(tabs):
    mo.md("---")
    tabs
    return


if __name__ == "__main__":
    app.run()
