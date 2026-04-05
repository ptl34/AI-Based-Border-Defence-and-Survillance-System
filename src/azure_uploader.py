"""
azure_uploader.py
-----------------
Azure Blob Storage  →  upload frames and video clips
Azure Cosmos DB     →  store and query alert documents

Usage:
    from src.azure_uploader import AzureUploader
    uploader = AzureUploader(blob_conn_str, cosmos_endpoint, cosmos_key)
    url  = uploader.upload_frame("data/processed/frame_00001.jpg")
    uploader.save_alert(alert.to_dict())
"""

import os
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.cosmos import CosmosClient, PartitionKey, exceptions as cosmos_exc


# Container names — must match what you created in the Azure Portal
CONTAINER_VIDEOS  = "surveillance-videos"
CONTAINER_FRAMES  = "processed-frames"
CONTAINER_LOGS    = "alert-logs"


class AzureUploader:
    """
    Wraps Azure Blob Storage + Cosmos DB operations for the surveillance pipeline.
    All methods fail gracefully and print an error rather than crashing the pipeline.
    """

    def __init__(self,
                 blob_connection_string: str,
                 cosmos_endpoint:        str,
                 cosmos_key:             str,
                 cosmos_database:        str = "surveillance_db",
                 cosmos_container:       str = "alerts"):
        """
        Args:
            blob_connection_string: From Azure Portal → Storage Account → Access keys
            cosmos_endpoint:        From Azure Portal → Cosmos DB → Keys
            cosmos_key:             Primary key for Cosmos DB
            cosmos_database:        Cosmos DB database name (auto-created)
            cosmos_container:       Cosmos DB container name  (auto-created)
        """
        # ── Blob Storage ─────────────────────────────────────────────────────
        self.blob_service = BlobServiceClient.from_connection_string(
            blob_connection_string)
        self._ensure_containers()

        # ── Cosmos DB ────────────────────────────────────────────────────────
        self.cosmos_client    = CosmosClient(cosmos_endpoint, cosmos_key)
        self.cosmos_db        = self.cosmos_client.create_database_if_not_exists(
                                    id=cosmos_database)
        self.alerts_container = self.cosmos_db.create_container_if_not_exists(
                                    id=cosmos_container,
                                    partition_key=PartitionKey(path="/priority"),
                                    offer_throughput=400)   # minimum RU — stays in free tier

        print("[AzureUploader] Connected to Blob Storage and Cosmos DB ✓")

    # ── Blob helpers ─────────────────────────────────────────────────────────

    def _ensure_containers(self) -> None:
        """Create the three blob containers if they don't already exist."""
        for name in (CONTAINER_VIDEOS, CONTAINER_FRAMES, CONTAINER_LOGS):
            try:
                self.blob_service.create_container(name)
                print(f"  [Blob] Created container: {name}")
            except Exception:
                pass   # container already exists

    def upload_frame(self, local_path: str,
                     blob_name: str = None) -> str:
        """
        Upload a single frame (.jpg) to the processed-frames container.

        Returns:
            Public-ish blob URL (access depends on container ACL).
        """
        blob_name  = blob_name or Path(local_path).name
        blob_client = self.blob_service.get_blob_client(
            container=CONTAINER_FRAMES, blob=blob_name)

        try:
            with open(local_path, "rb") as f:
                blob_client.upload_blob(f, overwrite=True)
            print(f"  [Blob] Uploaded frame → {blob_name}")
            return blob_client.url
        except Exception as exc:
            print(f"  [Blob] Upload failed: {exc}")
            return ""

    def upload_video(self, local_path: str,
                     blob_name: str = None) -> str:
        """Upload a video clip to the surveillance-videos container."""
        blob_name   = blob_name or Path(local_path).name
        blob_client = self.blob_service.get_blob_client(
            container=CONTAINER_VIDEOS, blob=blob_name)

        try:
            with open(local_path, "rb") as f:
                blob_client.upload_blob(f, overwrite=True)
            print(f"  [Blob] Uploaded video → {blob_name}")
            return blob_client.url
        except Exception as exc:
            print(f"  [Blob] Upload failed: {exc}")
            return ""

    def upload_alert_csv(self, local_path: str) -> str:
        """Upload an alert log CSV to the alert-logs container."""
        blob_name   = f"alerts_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        blob_client = self.blob_service.get_blob_client(
            container=CONTAINER_LOGS, blob=blob_name)

        try:
            with open(local_path, "rb") as f:
                blob_client.upload_blob(f, overwrite=True)
            print(f"  [Blob] Uploaded log  → {blob_name}")
            return blob_client.url
        except Exception as exc:
            print(f"  [Blob] Upload failed: {exc}")
            return ""

    # ── Cosmos DB helpers ────────────────────────────────────────────────────

    def save_alert(self, alert_data: dict) -> str:
        """
        Upsert an alert document to Cosmos DB.
        Adds a unique 'id' field required by Cosmos if not present.

        Returns:
            The document 'id'.
        """
        if "id" not in alert_data:
            alert_data["id"] = str(uuid.uuid4())

        try:
            self.alerts_container.upsert_item(alert_data)
            print(f"  [Cosmos] Saved alert {alert_data.get('alert_id', '?')} "
                  f"[{alert_data.get('priority', '?')}]")
            return alert_data["id"]
        except cosmos_exc.CosmosHttpResponseError as exc:
            print(f"  [Cosmos] Save failed: {exc}")
            return ""

    def get_recent_alerts(self, limit: int = 100,
                          priority: str = None) -> list[dict]:
        """
        Query the most recent alerts from Cosmos DB.

        Args:
            limit:    Maximum number of documents to return.
            priority: Optional filter — "HIGH" / "MEDIUM" / "LOW".

        Returns:
            List of alert dicts, newest first.
        """
        if priority:
            query = (f"SELECT * FROM c WHERE c.priority = '{priority}' "
                     f"ORDER BY c.timestamp DESC OFFSET 0 LIMIT {limit}")
        else:
            query = (f"SELECT * FROM c "
                     f"ORDER BY c.timestamp DESC OFFSET 0 LIMIT {limit}")

        try:
            items = list(self.alerts_container.query_items(
                query=query, enable_cross_partition_query=True))
            return items
        except Exception as exc:
            print(f"  [Cosmos] Query failed: {exc}")
            return []

    def get_alert_counts(self) -> dict:
        """Return count of alerts per priority level."""
        query = ("SELECT c.priority, COUNT(1) as cnt "
                 "FROM c GROUP BY c.priority")
        try:
            rows = list(self.alerts_container.query_items(
                query=query, enable_cross_partition_query=True))
            return {r["priority"]: r["cnt"] for r in rows}
        except Exception as exc:
            print(f"  [Cosmos] Count query failed: {exc}")
            return {}


# ──────────────────────────────────────────────────────────────────────────────
#  QUICK TEST  (requires valid credentials in config.yaml)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    az = cfg["azure"]
    uploader = AzureUploader(
        blob_connection_string = az["blob_connection_string"],
        cosmos_endpoint        = az["cosmos_endpoint"],
        cosmos_key             = az["cosmos_key"],
    )

    # Test: save 3 dummy alerts
    for i in range(3):
        dummy = {
            "alert_id":      f"TEST{i:03d}",
            "timestamp":     datetime.utcnow().isoformat(),
            "alert_type":    "test_alert",
            "confidence":    0.85,
            "anomaly_score": 0.75 + i * 0.05,
            "priority":      "HIGH",
            "location":      f"sector_{i:02d}",
        }
        uploader.save_alert(dummy)

    print("\nAlert counts:", uploader.get_alert_counts())
