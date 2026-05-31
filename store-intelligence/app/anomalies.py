from typing import List, Dict, Any
from app.models import StoreTrackingEvent, RetailEventType, StoreZoneID, CameraID

class RetailAnomalyDetector:
    def __init__(self, queue_alert_threshold: int = 10, abandon_alert_threshold: int = 5):
        """
        Initializes the operational anomaly engine with configurable retail risk thresholds.

        Inputs:
            queue_alert_threshold (int): Maximum customer depth allowed before triggering a checkout alert.
            abandon_alert_threshold (int): Upper bound of queue abandons allowed before raising structural alerts.
        """
        self.queue_alert_threshold = queue_alert_threshold
        self.abandon_alert_threshold = abandon_alert_threshold

    def scan_ledger_for_exceptions(self, active_events: List[StoreTrackingEvent]) -> List[Dict[str, Any]]:
        """
        Scans state event streams to catch operational bottlenecks and restricted zone violations.

        Inputs:
            active_events (List[StoreTrackingEvent]): Event list snapshot pulled from the storage engine.

        Outputs:
            List[Dict[str, Any]]: Array containing identified exceptions with severity indicators and remediation steps.
        """
        detected_exceptions = []
        queue_abandon_counter = 0
        current_queue_depth = 0

        if not active_events:
            return detected_exceptions

        for event in active_events:
            if event.camera_id == CameraID.CAM_4 and not event.is_staff:
                detected_exceptions.append({
                    "anomaly_type": "SECURITY_ZONE_VIOLATION",
                    "severity": "CRITICAL",
                    "camera_id": event.camera_id.value,
                    "visitor_id": event.visitor_id,
                    "timestamp": event.timestamp,
                    "remediation": "Deploy floor staff to storage perimeter to escort customer out of restricted zone."
                })

            if event.event_type == RetailEventType.QUEUE_ABANDONMENT:
                queue_abandon_counter += 1

            if event.metadata.queue_depth is not None:
                current_queue_depth = max(current_queue_depth, event.metadata.queue_depth)

        if current_queue_depth > self.queue_alert_threshold:
            detected_exceptions.append({
                "anomaly_type": "CHECKOUT_QUEUE_SPIKE",
                "severity": "HIGH",
                "current_depth": current_queue_depth,
                "timestamp": active_events[-1].timestamp,
                "remediation": "Open secondary registers immediately to clear the current checkout bottleneck."
            })

        if queue_abandon_counter > self.abandon_alert_threshold:
            detected_exceptions.append({
                "anomaly_type": "HIGH_QUEUE_ABANDONMENT_RATE",
                "severity": "WARN",
                "abandon_count": queue_abandon_counter,
                "timestamp": active_events[-1].timestamp,
                "remediation": "Inspect payment gateway terminals for connectivity drops or increase cashier allocations."
            })

        return detected_exceptions