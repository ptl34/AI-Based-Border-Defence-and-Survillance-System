"""
anomaly_trigger/__init__.py
----------------------------
Azure Function — HTTP-triggered anomaly processor.

Receives a JSON payload with frame metadata + raw detection data,
computes an anomaly score, assigns priority, and stores the alert
in Azure Cosmos DB.

Deploy:
    func azure functionapp publish <YOUR_FUNCTION_APP_NAME>

Test locally:
    func start
    curl -X POST http://localhost:7071/api/anomaly_trigger \
         -H "Content-Type: application/json" \
         -d '{"frame_path":"frame_00001.jpg","confidence":0.88,
              "detections":[{"class":"person","conf":0.88}],
              "location":"sector_01"}'
"""

import azure.functions as func
import json
import uuid
import logging
import os
from datetime import datetime


def _compute_anomaly_score(detections: list, confidence: float) -> float:
    """
    Lightweight anomaly scoring that runs inside the Function
    (no scikit-learn dependency needed for the serverless tier).

    Rule-based score — replace with model.predict() if you package
    the .pkl files with the Function App.
    """
    score = 0.0

    # Weapon detected → immediate high score
    if any(d.get("class") == "weapon" for d in detections):
        score += 0.60

    # Crowd (>5 persons)
    n_persons = sum(1 for d in detections if d.get("class") == "person")
    if n_persons > 5:
        score += 0.30
    elif n_persons > 2:
        score += 0.15

    # Vehicle surge (>3 vehicles)
    n_vehicles = sum(1 for d in detections if d.get("class") == "vehicle")
    if n_vehicles > 3:
        score += 0.20

    # Weight by detection confidence
    score *= (0.5 + 0.5 * min(confidence, 1.0))

    return min(float(score), 1.0)


def _priority(score: float) -> str:
    if score > 0.70:
        return "HIGH"
    if score > 0.40:
        return "MEDIUM"
    return "LOW"


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("anomaly_trigger: request received")

    # ── Parse request body ────────────────────────────────────────────────────
    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Request body must be valid JSON"}),
            status_code=400, mimetype="application/json")

    frame_path  = body.get("frame_path", "")
    confidence  = float(body.get("confidence", 0.0))
    detections  = body.get("detections", [])
    location    = body.get("location", "unknown")

    # ── Compute score ─────────────────────────────────────────────────────────
    score    = _compute_anomaly_score(detections, confidence)
    priority = _priority(score)

    # ── Build alert document ──────────────────────────────────────────────────
    alert = {
        "id":            str(uuid.uuid4()),
        "alert_id":      str(uuid.uuid4())[:8].upper(),
        "timestamp":     datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frame_path":    frame_path,
        "confidence":    round(confidence, 4),
        "anomaly_score": round(score, 4),
        "priority":      priority,
        "location":      location,
        "detections":    detections,
    }

    logging.info(f"anomaly_trigger: alert={alert['alert_id']} "
                 f"score={score:.3f} priority={priority}")

    # ── Store in Cosmos DB ────────────────────────────────────────────────────
    cosmos_endpoint = os.environ.get("COSMOS_ENDPOINT", "")
    cosmos_key      = os.environ.get("COSMOS_KEY", "")

    if cosmos_endpoint and cosmos_key:
        try:
            from azure.cosmos import CosmosClient
            client     = CosmosClient(cosmos_endpoint, cosmos_key)
            db         = client.get_database_client("surveillance_db")
            container  = db.get_container_client("alerts")
            container.upsert_item(alert)
            logging.info("anomaly_trigger: alert saved to Cosmos DB")
        except Exception as exc:
            logging.error(f"anomaly_trigger: Cosmos DB error — {exc}")
            # Do not fail the function — just log and continue

    # ── Return response ───────────────────────────────────────────────────────
    response_body = {
        "status":        "ok",
        "alert_id":      alert["alert_id"],
        "anomaly_score": alert["anomaly_score"],
        "priority":      priority,
        "timestamp":     alert["timestamp"],
    }
    return func.HttpResponse(
        json.dumps(response_body),
        status_code=200,
        mimetype="application/json",
    )
