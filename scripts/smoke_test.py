"""End-to-end smoke test for the Rapido Intelligent System serving layer.

Starts the FastAPI app in-process (ServerHandle), hits every endpoint, and
checks the four prediction models return sane JSON. Also verifies input
validation (422) and unknown-model handling (404).

Usage:
    .venv/bin/python scripts/smoke_test.py
Exit 0 = all checks passed.
"""
from __future__ import annotations

import os
import sys
import time

import httpx
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rapido_intelligent_system.api.app import app  # noqa: E402
from rapido_intelligent_system.api.runner import ServerHandle  # noqa: E402
import rapido_intelligent_system.modeling_mnb as MM  # noqa: E402

HOST, PORT = "127.0.0.1", 8055
BASE = f"http://{HOST}:{PORT}"

FEATS = {
    "booking_status": MM.BOOKING_FEATURES,
    "booking_value": MM.BOOKING_VALUE_FEATURES,
    "customer_cancel_flag": MM.CUSTOMER_FEATURES,
    "driver_delay_flag": MM.DRIVER_FEATURES,
}

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    handle = ServerHandle(app)
    handle.start(host=HOST, port=PORT)
    c = httpx.Client(base_url=BASE, timeout=30)
    try:
        for _ in range(60):
            try:
                if c.get("/health").status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)

        # --- meta endpoints ---
        r = c.get("/")
        check("GET /", r.status_code == 200 and "models" in r.json())
        r = c.get("/health")
        check("GET /health", r.status_code == 200 and r.json().get("status") == "ok")
        h_models = r.json().get("models", [])
        check("health lists 4 models", len(h_models) == 4, str(h_models))

        r = c.get("/models")
        check("GET /models -> 200", r.status_code == 200)
        model_info = r.json()
        check("models schema has 4 entries", len(model_info) == 4)
        ids = {m["id"] for m in model_info}
        check("registry ids correct",
              ids == set(FEATS), ",".join(sorted(ids)))

        # schema contract: every model's features must match the catalog
        for m in model_info:
            schema_feats = {f["name"] for f in m["features"]}
            check(f"schema[{m['id']}] features match catalog",
                  schema_feats == set(FEATS[m["id"]]),
                  f"{len(schema_feats)} features")

        # --- predictions on real test rows ---
        for mid, feats in FEATS.items():
            df = pl.read_csv(f"data/processed/{mid}_test.csv").head(1).drop(mid)
            payload = {
                k: (float(v) if isinstance(v, (int, float)) else str(v))
                for k, v in df.to_dicts()[0].items()
                if k in feats
            }
            r = c.post(f"/predict/{mid}", json=payload)
            ok = r.status_code == 200
            detail = r.text[:100] if not ok else ""
            check(f"POST /predict/{mid} -> 200", ok, detail)
            if not ok:
                continue
            body = r.json()
            check(f"{mid} has 'prediction'", "prediction" in body)
            check(f"{mid} task field", "task" in body, body.get("task", ""))

            if mid == "booking_value":
                val = body.get("prediction")
                check(f"{mid} numeric fare",
                      isinstance(val, (int, float)) and val > 0, f"val={val}")
            else:
                check(f"{mid} has label", "label" in body, body.get("label", ""))
                check(f"{mid} has probabilities",
                      isinstance(body.get("probabilities"), list)
                      and len(body["probabilities"]) > 0,
                      str(body.get("probabilities")))

        # --- error / validation paths ---
        r = c.get("/predict/nope/example")
        check("unknown model example -> 404", r.status_code == 404)
        r = c.post("/predict/booking_value", json={"ride_distance_km": "not-a-number"})
        check("bad type -> 422", r.status_code == 422)
        r = c.get("/predict/booking_status/example")
        check("example payload -> 200", r.status_code == 200 and len(r.json()) > 0)

    finally:
        handle.stop()

    print("----")
    if FAILURES:
        print(f"SMOKE TEST FAILED: {len(FAILURES)} check(s) failed")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("SMOKE TEST PASSED (all checks green)")
    return 0


if __name__ == "__main__":
    sys.exit(main())