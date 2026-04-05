"""
test_azure_upload.py
--------------------
Unit tests for AlertManager (no Azure connection required).
Azure-dependent tests are skipped when credentials are absent.
"""

import json
import os
import sqlite3
import tempfile
import pytest
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alert_manager import Alert, AlertManager, Priority


# ── Alert data class tests ────────────────────────────────────────────────────

class TestAlert:

    def test_high_priority_assigned(self):
        a = Alert.create(0.9, 0.85, "intrusion", "frame.jpg", "sector_01", ["person"])
        assert a.priority == "HIGH"

    def test_medium_priority_assigned(self):
        a = Alert.create(0.8, 0.55, "crowd", "frame.jpg", "sector_02", [])
        assert a.priority == "MEDIUM"

    def test_low_priority_assigned(self):
        a = Alert.create(0.6, 0.20, "motion", "frame.jpg", "sector_03", [])
        assert a.priority == "LOW"

    def test_alert_id_generated(self):
        a = Alert.create(0.9, 0.8, "test", "", "", [])
        assert len(a.alert_id) == 8

    def test_to_dict_contains_required_fields(self):
        a    = Alert.create(0.9, 0.8, "intrusion", "f.jpg", "s01", ["weapon"])
        d    = a.to_dict()
        keys = {"alert_id", "timestamp", "alert_type", "confidence",
                "anomaly_score", "priority", "frame_path", "location",
                "objects_detected"}
        assert keys.issubset(d.keys())

    def test_to_json_is_valid(self):
        a = Alert.create(0.9, 0.8, "intrusion", "f.jpg", "s01", ["person"])
        parsed = json.loads(a.to_json())
        assert parsed["priority"] == "HIGH"

    def test_confidence_rounded(self):
        a = Alert.create(0.9999, 0.8, "test", "", "", [])
        assert a.confidence == round(0.9999, 4)

    def test_emoji_high(self):
        a = Alert.create(0.9, 0.8, "t", "", "", [])
        assert a.emoji == "🔴"

    def test_emoji_medium(self):
        a = Alert.create(0.9, 0.5, "t", "", "", [])
        assert a.emoji == "🟡"

    def test_emoji_low(self):
        a = Alert.create(0.9, 0.2, "t", "", "", [])
        assert a.emoji == "🟢"


# ── AlertManager tests ────────────────────────────────────────────────────────

class TestAlertManager:

    @pytest.fixture
    def mgr(self, tmp_path):
        db = str(tmp_path / "test_alerts.db")
        return AlertManager(db_path=db, conf_threshold=0.5, log_low=True)

    def test_process_high_returns_alert(self, mgr):
        alert = mgr.process(0.9, 0.85, "intrusion", "f.jpg", "s01", ["person"])
        assert alert is not None
        assert alert.priority == "HIGH"

    def test_process_below_threshold_returns_none(self, mgr):
        """Confidence below threshold → filtered out → returns None."""
        result = mgr.process(0.40, 0.90, "intrusion", "f.jpg", "s01", [])
        assert result is None

    def test_process_saves_to_db(self, mgr):
        mgr.process(0.9, 0.85, "test", "f.jpg", "s01", [])
        rows = mgr.get_recent()
        assert len(rows) >= 1

    def test_stats_keys(self, mgr):
        mgr.process(0.9, 0.85, "t", "", "", [])
        s = mgr.stats()
        assert "total"  in s
        assert "HIGH"   in s
        assert "MEDIUM" in s
        assert "LOW"    in s
        assert "estimated_fpr" in s

    def test_get_recent_returns_list(self, mgr):
        mgr.process(0.9, 0.85, "t", "", "", [])
        rows = mgr.get_recent()
        assert isinstance(rows, list)
        assert len(rows) >= 1

    def test_get_recent_priority_filter(self, mgr):
        mgr.process(0.9, 0.85, "high_test",   "", "", [])
        mgr.process(0.9, 0.50, "medium_test", "", "", [])
        high_rows = mgr.get_recent(priority="HIGH")
        for row in high_rows:
            assert row["priority"] == "HIGH"

    def test_multiple_alerts_ordered_newest_first(self, mgr):
        import time
        mgr.process(0.9, 0.85, "first",  "", "", [])
        time.sleep(0.01)
        mgr.process(0.9, 0.85, "second", "", "", [])
        rows = mgr.get_recent()
        assert rows[0]["alert_type"] == "second"

    def test_db_survives_reinit(self, tmp_path):
        db = str(tmp_path / "persist.db")
        mgr1 = AlertManager(db_path=db, conf_threshold=0.5, log_low=True)
        mgr1.process(0.9, 0.85, "persist_test", "", "", [])

        mgr2 = AlertManager(db_path=db, conf_threshold=0.5, log_low=True)
        rows = mgr2.get_recent()
        assert any(r["alert_type"] == "persist_test" for r in rows)
