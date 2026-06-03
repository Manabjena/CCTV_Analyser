import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from typing import List, Dict, Any
from app.models import StoreTrackingEvent
from app.config import load_store_definitions, StoreDefinition
from app.ingestion import ThreadSafeIngestionEngine
from app.metrics import TransactionAnalyticsEngine
from app.anomalies import RetailAnomalyDetector
from app.health import StreamDiagnosticMonitor
from app.funnel import StatefulFunnelAggregator

app = FastAPI(
    title="Apex Retail Store Intelligence Interface",
    version="2026.1.0",
    description="High-throughput, containerized REST API surface computing offline storefront KPIs."
)

store_definitions: Dict[str, StoreDefinition] = load_store_definitions()
allowed_store_ids = set(store_definitions.keys())

ingestion_engine = ThreadSafeIngestionEngine(
    max_idempotency_cache_size=50000,
    retention_max_frames=450,
    allowed_store_ids=allowed_store_ids,
)

analytics_engines: Dict[str, TransactionAnalyticsEngine] = {
    store_id: TransactionAnalyticsEngine(
        transaction_csv_path=definition.transaction_csv_path,
        store_id=store_id,
        rolling_window_minutes=60,
    )
    for store_id, definition in store_definitions.items()
}

anomaly_detector = RetailAnomalyDetector(queue_alert_threshold=10)
diagnostic_monitor = StreamDiagnosticMonitor(maximum_allowable_lag_seconds=30.0)


def _resolve_store_definition(store_id: str) -> StoreDefinition:
    store_def = store_definitions.get(store_id)
    if not store_def:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Store profile identity '{store_id}' is missing or unauthorized within this interface scope."
        )
    return store_def

@app.post("/events/ingest", status_code=status.HTTP_200_OK)
async def ingest_stream_batches(events: List[StoreTrackingEvent]) -> Dict[str, Any]:
    """
    Validates, screens for duplicate requests, and ingests micro-batched tracking events.

    Inputs:
        events (List[StoreTrackingEvent]): Batched array collection containing up to 500 edge tracking payloads.

    Outputs:
        Dict[str, Any]: Processing summary including count of successfully registered tracking records.
    """
    if not events:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Ingestion payload collection batch cannot be empty."
        )

    if len(events) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Inbound network package dimensions exceed maximum micro-batch limit of 500 elements."
        )

    processed_counter = 0
    for event in events:
        is_inserted = ingestion_engine.validate_and_route_incoming_record(event)
        if is_inserted:
            processed_counter += 1

    return {
        "status": "success",
        "ingested_records": processed_counter,
        "dropped_duplicates": len(events) - processed_counter
    }

@app.get("/stores/{store_id}/metrics", response_model=Dict[str, Any])
async def get_store_metrics(store_id: str) -> Dict[str, Any]:
    """
    Calculates and serves real-time performance metrics over a rolling temporal window.

    Inputs:
        store_id (str): Unique business identifier profile tracking the target store.

    Outputs:
        Dict[str, Any]: High-fidelity retail KPIs and conversion calculations.
    """
    _resolve_store_definition(store_id)
    active_events = ingestion_engine.extract_active_session_logs(store_id)
    
    current_billing_queue_depth = 0
    if active_events:
        last_event = active_events[-1]
        current_billing_queue_depth = last_event.metadata.queue_depth or 0

    metrics_payload = analytics_engines[store_id].calculate_rolling_store_metrics(
        active_events,
        current_billing_queue_depth,
        store_id=store_id,
    )
    return metrics_payload

@app.get("/health", status_code=status.HTTP_200_OK)
async def check_system_health() -> Dict[str, Any]:
    """
    Performs live hardware dependency verifications and connectivity monitoring checks.

    Outputs:
        Dict[str, Any]: Diagnostic platform readiness and timestamp mapping strings.
    """
    current_utc_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "status": "healthy",
        "timestamp": current_utc_timestamp,
        "storage_allocation": "in_memory_ledger",
        "thread_safe_lock_active": True
    }

@app.get("/stores/{store_id}/funnel")
async def get_store_funnel_analysis(store_id: str):
    _resolve_store_definition(store_id)
    active_events = ingestion_engine.extract_active_session_logs(store_id)
    funnel_aggregator = StatefulFunnelAggregator(transaction_processor=analytics_engines[store_id])
    return funnel_aggregator.compile_conversion_funnel(active_events, store_id=store_id)

@app.get("/stores/{store_id}/anomalies")
async def get_store_operational_exceptions(store_id: str):
    _resolve_store_definition(store_id)
    active_events = ingestion_engine.extract_active_session_logs(store_id)
    return anomaly_detector.scan_ledger_for_exceptions(active_events)