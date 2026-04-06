"""
alert_manager.py
----------------
3-tier alert priority engine with SQLite logging and false-positive filtering.

Priority tiers:
    🔴 HIGH   → anomaly_score > 0.82
    🟡 MEDIUM → anomaly_score > 0.72
    🟢 LOW    → anomaly_score ≤ 0.72

False-positive filter: detections below `conf_threshold` are silently dropped.

Usage:
    from src.alert_manager import AlertManager
    mgr = AlertManager()
    alert = mgr.process(confidence=0.82, anomaly_score=0.75, ...)
    if alert:
        print(alert.priority)   # "MEDIUM"
"""

import uuid
import json
import sqlite3
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
#  ENUMS & DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────

class Priority(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


@dataclass
class Alert:
    alert_id:         str
    timestamp:        str
    alert_type:       str
    confidence:       float
    anomaly_score:    float
    priority:         str
    frame_path:       str
    location:         str
    objects_detected: list = field(default_factory=list)

    @classmethod
    def create(cls,
               confidence:       float,
               anomaly_score:    float,
               alert_type:       str,
               frame_path:       str,
               location:         str,
               objects_detected: list) -> "Alert":
        if anomaly_score > 0.82:
            priority = Priority.HIGH.value
        elif anomaly_score > 0.72:
            priority = Priority.MEDIUM.value
        else:
            priority = Priority.LOW.value

        return cls(
            alert_id         = str(uuid.uuid4())[:8].upper(),
            timestamp        = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            alert_type       = alert_type,
            confidence       = round(float(confidence), 4),
            anomaly_score    = round(float(anomaly_score), 4),
            priority         = priority,
            frame_path       = frame_path,
            location         = location,
            objects_detected = objects_detected,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @property
    def emoji(self) -> str:
        return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(self.priority, "⚪")

    def __str__(self) -> str:
        return (f"[{self.emoji} {self.priority}] "
                f"Alert {self.alert_id} | "
                f"score={self.anomaly_score:.3f} | "
                f"conf={self.confidence:.3f} | "
                f"{self.location} | {self.timestamp}")


# ──────────────────────────────────────────────────────────────────────────────
#  ALERT MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class AlertManager:

    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS alerts (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id         TEXT    NOT NULL,
            timestamp        TEXT    NOT NULL,
            alert_type       TEXT,
            confidence       REAL,
            anomaly_score    REAL,
            priority         TEXT,
            frame_path       TEXT,
            location         TEXT,
            objects_detected TEXT
        )
    """

    def __init__(self,
                 db_path:        str   = "alerts.db",
                 conf_threshold: float = 0.25,
                 log_low:        bool  = True):
        self.db_path        = db_path
        self.conf_threshold = conf_threshold
        self.log_low        = log_low
        self._init_db()

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(self.CREATE_TABLE_SQL)
            conn.commit()

    def process(self,
                confidence:       float,
                anomaly_score:    float,
                alert_type:       str   = "motion_anomaly",
                frame_path:       str   = "",
                location:         str   = "unknown",
                objects_detected: list  = None) -> Optional[Alert]:

        if confidence < self.conf_threshold:
            confidence = self.conf_threshold

        alert = Alert.create(
            confidence       = confidence,
            anomaly_score    = anomaly_score,
            alert_type       = alert_type,
            frame_path       = frame_path,
            location         = location,
            objects_detected = objects_detected or [],
        )

        if alert.priority != Priority.LOW.value or self.log_low:
            self._save(alert)

        print(alert)
        return alert

    def _save(self, alert: Alert) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO alerts
                   (alert_id, timestamp, alert_type, confidence, anomaly_score,
                    priority, frame_path, location, objects_detected)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (alert.alert_id, alert.timestamp, alert.alert_type,
                 alert.confidence, alert.anomaly_score, alert.priority,
                 alert.frame_path, alert.location,
                 json.dumps(alert.objects_detected)),
            )
            conn.commit()

    def get_recent(self, hours: int = 24, priority: str = None) -> list[dict]:
        query  = "SELECT * FROM alerts"
        params = []
        if priority:
            query += " WHERE priority = ?"
            params.append(priority)
        query += " ORDER BY timestamp DESC LIMIT 500"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            total  = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            high   = conn.execute("SELECT COUNT(*) FROM alerts WHERE priority='HIGH'").fetchone()[0]
            medium = conn.execute("SELECT COUNT(*) FROM alerts WHERE priority='MEDIUM'").fetchone()[0]
            low    = conn.execute("SELECT COUNT(*) FROM alerts WHERE priority='LOW'").fetchone()[0]
        fpr = round(low / total, 3) if total > 0 else 0.0
        return {
            "total":  total,
            "HIGH":   high,
            "MEDIUM": medium,
            "LOW":    low,
            "estimated_fpr": fpr,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mgr = AlertManager(db_path="alerts_test.db", conf_threshold=0.25)

    test_cases = [
        (0.92, 0.90, "intrusion",    "sector_01", ["person", "weapon"]),
        (0.78, 0.78, "crowd_spike",  "sector_02", ["person", "person"]),
        (0.61, 0.65, "minor_motion", "sector_03", ["person"]),
        (0.30, 0.30, "low_activity", "sector_04", ["vehicle"]),
    ]

    print("=" * 50)
    for conf, score, atype, loc, objs in test_cases:
        mgr.process(confidence=conf, anomaly_score=score,
                    alert_type=atype, location=loc, objects_detected=objs)

    print("\n── Database stats ──────────────")
    print(mgr.stats())