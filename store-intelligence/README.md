# Store Intelligence Platform

This repository implements a retail store intelligence platform for the Purple Tech challenge.
It supports ingestion of store tracking events, rolling metrics computation, funnel analysis, and anomaly detection.

## Dataset and project reconstruction

The repository currently includes:
- `POS - sample transactionsb1e826f.csv`: Point-of-sale transaction ledger.
- `sample_eventsbe42122.jsonl`: Example event stream payloads.

The platform is reconstructed to support both the existing `ST1008` dataset and future store directories named with the dataset convention:
- `Store 1/CAM 1 - zone.mp4`
- `Store 1/CAM 2 - zone.mp4`
- `Store 1/CAM 3 - entry.mp4`
- `Store 1/CAM 5 - billing.mp4`
- `Store 1/Store 1 - layout.png`
- `Store 2/Store 2/billing_area.mp4`
- `Store 2/Store 2/entry 1.mp4`
- `Store 2/Store 2/entry 2.mp4`
- `Store 2/Store 2/store 2 - layout.png`
- `Store 2/Store 2/zone.mp4`

The code now discovers store folders under the repository root, maps the available camera files by `CAM {n}` names, and uses layout PNG paths when they exist.

## Running the service

Install dependencies and start the API server from the `store-intelligence` folder:

```bash
python3 -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

API endpoints:
- `POST /events/ingest`
- `GET /stores/{store_id}/metrics`
- `GET /stores/{store_id}/funnel`
- `GET /stores/{store_id}/anomalies`
- `GET /health`

## Notes

- `app/config.py` defines store discovery and available datasets.
- `app/dataset.py` discovers store directories and camera naming conventions.
- `app/main.py` now routes requests by store ID and supports dynamic store definitions.
- `app/metrics.py` now returns store-aware KPI payloads and handles optional transaction CSV files gracefully.

## Testing

Run tests from the `store-intelligence` folder:

```bash
pytest
```
