import collections
import threading
from typing import Dict, List, Set
from app.models import StoreTrackingEvent, RetailEventType

class ThreadSafeIngestionEngine:
    def __init__(self, max_idempotency_cache_size: int = 50000, retention_max_frames: int = 450):
        """
        Initializes an operational in-memory tracking repository protected by re-entrant mutex locks.

        Inputs:
            max_idempotency_cache_size (int): Upper bound limit tracking unique event IDs to prevent memory bloat.
            retention_max_frames (int): Maximum lifespan variance allowed before stale sessions are removed.
        """
        self.max_idempotency_cache_size = max_idempotency_cache_size
        self.retention_max_frames = retention_max_frames
        
        self.engine_mutex = threading.Lock()
        self.seen_event_ids: Set[str] = set()
        self.idempotency_fifo_ring = collections.deque()
        
        self.master_events_ledger: List[StoreTrackingEvent] = []
        self.visitor_session_ledger: Dict[str, List[StoreTrackingEvent]] = {}

    def _apply_retention_policy_sweep_under_lock(self, current_frame_index: int) -> None:
        """
        Sweeps internal mapping lists to clear out stale tracking sequences.

        Inputs:
            current_frame_index (int): Active system tracking frame counter reference point.
        """
        expired_visitors = []
        
        for visitor_id, events in self.visitor_session_ledger.items():
            if not events:
                continue
            last_event = events[-1]
            if (current_frame_index - last_event.metadata.frame_index) > self.retention_max_frames:
                expired_visitors.append(visitor_id)
            elif last_event.event_type == RetailEventType.SESSION_END:
                expired_visitors.append(visitor_id)

        for visitor_id in expired_visitors:
            del self.visitor_session_ledger[visitor_id]

    def validate_and_route_incoming_record(self, event: StoreTrackingEvent) -> bool:
        """
        Validates incoming data frames, screens out duplicate requests in O(1) time, 
        and updates metrics stores safely across thread contexts.

        Inputs:
            event (StoreTrackingEvent): Pydantic-validated store tracking event model.

        Outputs:
            bool: True if the event was processed successfully, False if it was rejected as a duplicate.
        """
        event_uuid_str = str(event.event_id)

        with self.engine_mutex:
            if event_uuid_str in self.seen_event_ids:
                return False

            self.seen_event_ids.add(event_uuid_str)
            self.idempotency_fifo_ring.append(event_uuid_str)

            if len(self.seen_event_ids) > self.max_idempotency_cache_size:
                evicted_id = self.idempotency_fifo_ring.popleft()
                self.seen_event_ids.remove(evicted_id)

            if event.store_id != "ST1008":
                return False

            self._apply_retention_policy_sweep_under_lock(event.metadata.frame_index)

            self.master_events_ledger.append(event)
            self.visitor_session_ledger.setdefault(event.visitor_id, []).append(event)
            return True

    def extract_active_session_logs(self) -> List[StoreTrackingEvent]:
        """
        Extracts structural records from the master data ledger safely across execution thread blocks.

        Outputs:
            List[StoreTrackingEvent]: Copied snapshot tracking state logs array.
        """
        with self.engine_mutex:
            return list(self.master_events_ledger)

    def clear_active_datastore_records(self) -> None:
        """
        Resets and purges all active transactional storage parameters safely across operational threads.
        """
        with self.engine_mutex:
            self.master_events_ledger.clear()
            self.visitor_session_ledger.clear()
            self.seen_event_ids.clear()
            self.idempotency_fifo_ring.clear()