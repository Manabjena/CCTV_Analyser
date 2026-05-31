from datetime import datetime
from typing import Dict, List, Any
from app.models import StoreTrackingEvent, CameraID

class StreamDiagnosticMonitor:
    def __init__(self, maximum_allowable_lag_seconds: float = 30.0):
        """
        Initializes the operational stream monitor to track feed staleness across cameras.

        Inputs:
            maximum_allowable_lag_seconds (float): Timeout threshold window defining feed dropout alerts.
        """
        self.max_lag = maximum_allowable_lag_seconds
        self.camera_heartbeat_registry: Dict[CameraID, datetime] = {}

    def update_diagnostic_heartbeats(self, active_events: List[StoreTrackingEvent]) -> None:
        """
        Parses tracking events to update the latest arrival timestamps for active cameras.

        Inputs:
            active_events (List[StoreTrackingEvent]): Live streaming arrays arriving from edge pipelines.
        """
        for event in active_events:
            cleaned_timestamp = event.timestamp.replace("Z", "+00:00")
            event_dt = datetime.fromisoformat(cleaned_str=cleaned_timestamp)
            
            current_latest = self.camera_heartbeat_registry.get(event.camera_id)
            if not current_latest or event_dt > current_latest:
                self.camera_heartbeat_registry[event.camera_id] = event_dt

    def evaluate_pipeline_health(self) -> Dict[str, Any]:
        """
        Compares host system times against camera heartbeats to identify stale data feeds.

        Outputs:
            Dict[str, Any]: Combined diagnostic payload showing the status of each tracking stream.
        """
        current_system_time = datetime.utcnow()
        unhealthy_camera_feeds = []
        camera_status_report = {}

        mandatory_cameras = [CameraID.CAM_1, CameraID.CAM_2, CameraID.CAM_3, CameraID.CAM_5]

        for cam in mandatory_cameras:
            last_seen = self.camera_heartbeat_registry.get(cam)
            
            if last_seen is None:
                camera_status_report[cam.value] = "NO_DATA_RECEIVED"
                unhealthy_camera_feeds.append(cam.value)
                continue

            drift_seconds = (current_system_time - last_seen.replace(tzinfo=None)).total_seconds()
            
            if drift_seconds > self.max_lag:
                camera_status_report[cam.value] = f"STALE_FEED_LAG_{round(drift_seconds, 1)}s"
                unhealthy_camera_feeds.append(cam.value)
            else:
                camera_status_report[cam.value] = "ACTIVE"

        is_healthy = len(unhealthy_camera_feeds) == 0
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "timestamp": current_system_time.isoformat() + "Z",
            "camera_status_matrix": camera_status_report,
            "unhealthy_feeds_count": len(unhealthy_camera_feeds),
            "rereoute_required": not is_healthy
        }