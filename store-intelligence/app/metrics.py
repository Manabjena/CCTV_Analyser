import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from app.models import StoreTrackingEvent, RetailEventType, StoreZoneID

class TransactionAnalyticsEngine:
    def __init__(self, transaction_csv_path: Optional[str] = None, store_id: Optional[str] = None, rolling_window_minutes: int = 60):
        """
        Initializes the production-grade analytics engine, transforming the flat POS 
        ledger into an O(1) temporally bucketed hash index.

        Inputs:
            transaction_csv_path (Optional[str]): File system path targeting the POS purchase transactions ledger.
            store_id (Optional[str]): Optional store identifier to filter ledger records.
            rolling_window_minutes (int): Time horizon window defining active calculations.
        """
        self.rolling_window_minutes = rolling_window_minutes
        self.bucketed_transactions: Dict[int, List[datetime]] = {}
        self.transaction_csv_path = Path(transaction_csv_path) if transaction_csv_path else None
        self.store_id = store_id

        if self.transaction_csv_path is None or not self.transaction_csv_path.exists():
            return

        raw_df = pd.read_csv(self.transaction_csv_path)
        if self.store_id:
            raw_df = raw_df[raw_df['store_id'] == self.store_id].copy()

        raw_df['parsed_datetime'] = pd.to_datetime(
            raw_df['order_date'] + ' ' + raw_df['order_time'],
            format='%d-%m-%Y %H:%M:%S'
        ).dt.tz_localize('UTC')

        self._build_temporal_transaction_index(raw_df)

    def _build_temporal_transaction_index(self, df: pd.DataFrame) -> None:
        """
        Transforms a linear DataFrame into an optimized, minute-by-minute transactional hash lookup index.

        Inputs:
            df (pd.DataFrame): Filtered transaction record matrix.
        """
        for _, row in df.iterrows():
            dt: datetime = row['parsed_datetime']
            minute_timestamp = int(dt.timestamp() // 60)
            self.bucketed_transactions.setdefault(minute_timestamp, []).append(dt)

    def _convert_iso_string_to_datetime(self, iso_str: str) -> datetime:
        """
        Normalizes inbound network string timestamps into standardized timezone-aware datetime entities.

        Inputs:
            iso_str (str): Raw timestamp string arriving from network sockets.

        Outputs:
            datetime: Timezone-normalized python datetime representation.
        """
        cleaned_str = iso_str.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned_str)

    def _verify_transaction_proximity_fast(self, event_time: datetime, padding_minutes: int = 5) -> bool:
        """
        Executes an O(1) hash map evaluation to match tracking timestamps to transactions 
        within a sliding padding window.

        Inputs:
            event_time (datetime): Core timestamp reference when the customer inhabited the checkout zone.
            padding_minutes (int): Temporal radius bounds checking invoice transactions.

        Outputs:
            bool: True if a matching transaction is found within the specified window boundary.
        """
        start_minute = int((event_time - timedelta(minutes=padding_minutes)).timestamp() // 60)
        end_minute = int((event_time + timedelta(minutes=padding_minutes)).timestamp() // 60)

        for min_bucket in range(start_minute, end_minute + 1):
            if min_bucket in self.bucketed_transactions:
                for txn_time in self.bucketed_transactions[min_bucket]:
                    if abs((txn_time - event_time).total_seconds()) <= (padding_minutes * 60):
                        return True
        return False

    def calculate_rolling_store_metrics(
        self,
        active_ledger_events: List[StoreTrackingEvent],
        current_queue_depth: int,
        store_id: str = "UNKNOWN",
    ) -> Dict[str, Any]:
        """
        Aggregates track data within a rolling time window, utilizing hash indexes 
        to compute real-time metrics and conversion rates.

        Inputs:
            active_ledger_events (List[StoreTrackingEvent]): Event streams pulled from the thread-safe database ledger.
            current_queue_depth (int): Active count of consumers inside checkout queues.

        Outputs:
            Dict[str, Any]: High-fidelity retail KPIs calculated over the active rolling time window.
        """
        if not active_ledger_events:
            return self._build_empty_metrics_payload(current_queue_depth, store_id=store_id)

        newest_event_time = self._convert_iso_string_to_datetime(active_ledger_events[-1].timestamp)
        horizon_cutoff_time = newest_event_time - timedelta(minutes=self.rolling_window_minutes)

        unique_visitors: Set[str] = set()
        converted_visitors: Set[str] = set()
        zone_dwell_times_map: Dict[str, List[int]] = {}

        for event in active_ledger_events:
            if event.is_staff:
                continue

            event_datetime = self._convert_iso_string_to_datetime(event.timestamp)
            if event_datetime < horizon_cutoff_time:
                continue

            unique_visitors.add(event.visitor_id)

            if event.zone_id and event.dwell_ms > 0:
                zone_dwell_times_map.setdefault(event.zone_id.value, []).append(event.dwell_ms)

            if event.camera_id == "CAM_5" and event.event_type in [RetailEventType.BILLING_QUEUE_JOIN, RetailEventType.ZONE_DWELL]:
                if self._verify_transaction_proximity_fast(event_datetime, padding_minutes=5):
                    converted_visitors.add(event.visitor_id)

        total_unique_visitors = len(unique_visitors)
        total_conversions = len(converted_visitors)
        conversion_rate = (total_conversions / total_unique_visitors) if total_unique_visitors > 0 else 0.0

        average_dwell_by_zone = {}
        for zone_name, durations in zone_dwell_times_map.items():
            average_dwell_by_zone[zone_name] = float(np.mean(durations))

        confidence_rating = "HIGH" if total_unique_visitors >= 5 else "LOW"

        return {
            "store_id": store_id,
            "calculation_horizon_minutes": self.rolling_window_minutes,
            "unique_visitors": total_unique_visitors,
            "conversion_rate": round(conversion_rate, 4),
            "average_dwell_by_zone": average_dwell_by_zone,
            "current_queue_depth": current_queue_depth,
            "data_confidence_rating": confidence_rating
        }

    def _build_empty_metrics_payload(self, current_queue_depth: int, store_id: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Constructs a clean baseline data payload skeleton when no active data matches the query parameters.

        Inputs:
            current_queue_depth (int): Active consumer counts gathered at checkout counters.
            store_id (str): The store identifier for this metrics payload.

        Outputs:
            Dict[str, Any]: Empty metrics structure fallback model.
        """
        return {
            "store_id": store_id,
            "calculation_horizon_minutes": self.rolling_window_minutes,
            "unique_visitors": 0,
            "conversion_rate": 0.0,
            "average_dwell_by_zone": {},
            "current_queue_depth": current_queue_depth,
            "data_confidence_rating": "LOW"
        }
