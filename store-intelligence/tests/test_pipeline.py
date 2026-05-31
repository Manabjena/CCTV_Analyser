import uuid
import pytest
from fastapi.testclient import TestClient
from app.main import app, ingestion_engine

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_ingestion_state():
    """
    Resets the internal engine storage state before executing each test case.
    """
    ingestion_engine.clear_active_datastore_records()

def test_system_health_endpoint_returns_success():
    """
    Asserts that the system health endpoint answers with a 200 OK code and clean health state markers.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["thread_safe_lock_active"] is True

def test_valid_event_ingestion_stream_succeeds():
    """
    Verifies that a well-formed edge event package satisfies the Pydantic parser contracts.
    """
    payload = [{
        "event_id": str(uuid.uuid4()),
        "store_id": "ST1008",
        "camera_id": "CAM_3",
        "visitor_id": "VIS_GLOBAL_101",
        "event_type": "ENTRY",
        "timestamp": "2026-04-10T16:50:00Z",
        "zone_id": "ENTRANCE_ZONE",
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.92,
        "metadata": {
            "queue_depth": None,
            "sku_zone": None,
            "session_seq": 1,
            "frame_index": 1500,
            "video_time_seconds": 100.0
        }
    }]
    response = client.post("/events/ingest", json=payload)
    assert response.status_code == 200
    assert response.json()["ingested_records"] == 1

def test_idempotency_ring_buffer_filters_duplicate_event_ids():
    """
    Asserts that duplicate event IDs are caught in O(1) time and filtered out silently.
    """
    shared_uuid = str(uuid.uuid4())
    payload = {
        "event_id": shared_uuid,
        "store_id": "ST1008",
        "camera_id": "CAM_1",
        "visitor_id": "VIS_GLOBAL_101",
        "event_type": "ZONE_ENTER",
        "timestamp": "2026-04-10T16:51:00Z",
        "zone_id": "MAKEUP",
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.88,
        "metadata": {
            "queue_depth": None,
            "sku_zone": "MAKEUP",
            "session_seq": 2,
            "frame_index": 2000,
            "video_time_seconds": 133.33
        }
    }
    
    response_first = client.post("/events/ingest", json=[payload])
    assert response_first.status_code == 200
    assert response_first.json()["ingested_records"] == 1
    
    response_second = client.post("/events/ingest", json=[payload])
    assert response_second.status_code == 200
    assert response_second.json()["ingested_records"] == 0
    assert response_second.json()["dropped_duplicates"] == 1

def test_invalid_store_id_registration_is_rejected():
    """
    Verifies that incoming payloads with unauthorized store IDs fail validation parameters.
    """
    payload = [{
        "event_id": str(uuid.uuid4()),
        "store_id": "ST9999",
        "camera_id": "CAM_3",
        "visitor_id": "VIS_GLOBAL_101",
        "event_type": "ENTRY",
        "timestamp": "2026-04-10T16:50:00Z",
        "zone_id": "ENTRANCE_ZONE",
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.90,
        "metadata": {
            "queue_depth": None,
            "sku_zone": None,
            "session_seq": 1,
            "frame_index": 100,
            "video_time_seconds": 6.66
        }
    }]
    response = client.post("/events/ingest", json=payload)
    assert response.status_code == 200
    assert response.json()["ingested_records"] == 0

def test_metrics_calculation_matching_sliding_window_ledgers():
    """
    Verifies the rolling store conversion metric calculations.
    Matches a test visitor billing queue record to the transaction time in the CSV file.
    """
    payload = [
        {
            "event_id": str(uuid.uuid4()),
            "store_id": "ST1008",
            "camera_id": "CAM_3",
            "visitor_id": "VIS_GLOBAL_999",
            "event_type": "ENTRY",
            "timestamp": "2026-04-10T16:52:00Z",
            "zone_id": "ENTRANCE_ZONE",
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.95,
            "metadata": {
                "queue_depth": None,
                "sku_zone": None,
                "session_seq": 1,
                "frame_index": 100,
                "video_time_seconds": 6.66
            }
        },
        {
            "event_id": str(uuid.uuid4()),
            "store_id": "ST1008",
            "camera_id": "CAM_5",
            "visitor_id": "VIS_GLOBAL_999",
            "event_type": "BILLING_QUEUE_JOIN",
            "timestamp": "2026-04-10T16:55:00Z",
            "zone_id": "BILLING_ZONE",
            "dwell_ms": 45000,
            "is_staff": False,
            "confidence": 0.94,
            "metadata": {
                "queue_depth": 1,
                "sku_zone": "BILLING_ZONE",
                "session_seq": 2,
                "frame_index": 500,
                "video_time_seconds": 33.33
            }
        }
    ]
    
    ingest_resp = client.post("/events/ingest", json=payload)
    assert ingest_resp.status_code == 200
    
    metrics_resp = client.get("/stores/ST1008/metrics")
    assert metrics_resp.status_code == 200
    metrics_data = metrics_resp.json()
    
    assert metrics_data["unique_visitors"] == 1
    assert metrics_data["conversion_rate"] == 1.0
    assert "BILLING_ZONE" in metrics_data["average_dwell_by_zone"]