import datetime
from typing import List, Dict, Any, Set
from app.models import StoreTrackingEvent, RetailEventType, StoreZoneID

class StatefulFunnelAggregator:
    def __init__(self, transaction_processor):
        """
        Initializes the funnel aggregator engine, linking it to the transaction engine 
        to track final purchase conversions.

        Inputs:
            transaction_processor (TransactionAnalyticsEngine): Pre-indexed transaction processing layer instance.
        """
        self.tx_engine = transaction_processor

    def compile_conversion_funnel(self, active_events: List[StoreTrackingEvent], store_id: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Aggregates customer pathways into a multi-stage funnel to calculate conversion rates.

        Inputs:
            active_events (List[StoreTrackingEvent]): Snapshot array of active customer entries.

        Outputs:
            Dict[str, Any]: Step-by-step conversion counts and percentage calculations for the storefront.
        """
        stage_1_entry_ids: Set[str] = set()
        stage_2_browse_ids: Set[str] = set()
        stage_3_queue_ids: Set[str] = set()
        stage_4_purchase_ids: Set[str] = set()

        browsing_zones = {StoreZoneID.MAKEUP, StoreZoneID.FRAGRANCE, StoreZoneID.SKIN, StoreZoneID.HAIR, StoreZoneID.MAIN_FLOOR}
        customer_events = [e for e in active_events if not e.is_staff]

        for event in customer_events:
            v_id = event.visitor_id
            stage_1_entry_ids.add(v_id)

            if event.zone_id in browsing_zones:
                stage_2_browse_ids.add(v_id)

            if event.camera_id == "CAM_5" and event.event_type in [RetailEventType.BILLING_QUEUE_JOIN, RetailEventType.ZONE_DWELL]:
                stage_3_queue_ids.add(v_id)
                
                cleaned_timestamp = event.timestamp.replace("Z", "+00:00")
                event_datetime = datetime.fromisoformat(cleaned_timestamp)
                
                if self.tx_engine._verify_transaction_proximity_fast(event_datetime, padding_minutes=5):
                    stage_4_purchase_ids.add(v_id)

        count_entry = len(stage_1_entry_ids)
        count_browse = len(stage_2_browse_ids)
        count_queue = len(stage_3_queue_ids)
        count_purchase = len(stage_4_purchase_ids)

        return {
            "store_id": store_id,
            "funnel_metrics": {
                "stage_1_store_entry": count_entry,
                "stage_2_product_browse": count_browse,
                "stage_3_checkout_queue": count_queue,
                "stage_4_completed_purchase": count_purchase
            },
            "conversion_efficiencies": {
                "entry_to_browse_rate": round(count_browse / count_entry, 4) if count_entry > 0 else 0.0,
                "browse_to_queue_rate": round(count_queue / count_browse, 4) if count_browse > 0 else 0.0,
                "queue_to_purchase_rate": round(count_purchase / count_queue, 4) if count_queue > 0 else 0.0,
                "net_store_conversion_rate": round(count_purchase / count_entry, 4) if count_entry > 0 else 0.0
            }
        }