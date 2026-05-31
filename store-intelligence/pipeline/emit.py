import uuid
import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

class SpatialEventEmitter:
    def __init__(self, store_id: str = "ST1008", frame_rate: float = 15.0, tracking_timeout_frames: int = 150):
        """
        Initializes the state-space spatial event emission engine with persistent historical 
        tracking tables and leakage protection structures.

        Inputs:
            store_id (str): The unique corporate profile code for the active physical store.
            frame_rate (float): Operating frame cadence frequency of the camera network streams.
            tracking_timeout_frames (int): Target frame window to maintain lost tracks before forcing session eviction.
        """
        self.store_id = store_id
        self.frame_rate = frame_rate
        self.tracking_timeout_frames = tracking_timeout_frames
        
        self.track_zone_history: Dict[int, str] = {}
        self.track_camera_history: Dict[int, str] = {}
        self.store_entry_frames: Dict[int, int] = {}
        self.zone_entry_frames: Dict[int, int] = {}
        self.track_last_seen_frames: Dict[int, int] = {}
        self.track_dwell_emission_counter: Dict[int, int] = {}
        self.track_visitor_sequence_counter: Dict[int, int] = {}
        self.track_last_centroid_y: Dict[int, float] = {}
        self.staff_global_ids: Set[int] = set()

    def calculate_centroid(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Calculates the 2D geometric center coordinate point of an entity boundary array.

        Inputs:
            bbox (List[float]): Bounding box coordinates ordered as [xmin, ymin, xmax, ymax].

        Outputs:
            Tuple[float, float]: Position coordinates for the center (x, y) point.
        """
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    def resolve_spatial_zone(self, camera_id: str, x: float, y: float) -> str:
        """
        Maps raw spatial coordinates to explicit retail category zones.

        Inputs:
            camera_id (str): Identifier of the camera stream capturing the target.
            x (float): Horizontal center coordinate pixel location.
            y (float): Vertical center coordinate pixel location.

        Outputs:
            str: Assigned retail product zone or functional category zone label string.
        """
        if camera_id == "CAM_3":
            return "ENTRANCE_ZONE"
        if camera_id == "CAM_5":
            return "BILLING_ZONE"
        if camera_id == "CAM_4":
            return "STORAGE_ROOM"
        if camera_id == "CAM_1":
            return "MAKEUP" if x < 960 else "FRAGRANCE"
        if camera_id == "CAM_2":
            return "SKIN" if x < 960 else "HAIR"
        return "MAIN_FLOOR"

    def _get_next_sequence(self, global_id: int) -> int:
        """
        Generates and logs a strictly monotonic, sequential event sequence number for a specific target.

        Inputs:
            global_id (int): Unique identifier token for the tracking target.

        Outputs:
            int: The incremented sequential index position parameter.
        """
        self.track_visitor_sequence_counter[global_id] = self.track_visitor_sequence_counter.get(global_id, 0) + 1
        return self.track_visitor_sequence_counter[global_id]

    def _evict_identity_state(self, global_id: int, frame_index: int) -> None:
        """
        Purges an active visitor identity across all mapping caches to prevent memory leaks.

        Inputs:
            global_id (int): Absolute global tracking identity slot.
            frame_index (int): Execution index line tracker.
        """
        if global_id in self.track_zone_history:
            del self.track_zone_history[global_id]
        if global_id in self.track_camera_history:
            del self.track_camera_history[global_id]
        if global_id in self.store_entry_frames:
            del self.store_entry_frames[global_id]
        if global_id in self.zone_entry_frames:
            del self.zone_entry_frames[global_id]
        if global_id in self.track_last_seen_frames:
            del self.track_last_seen_frames[global_id]
        if global_id in self.track_dwell_emission_counter:
            del self.track_dwell_emission_counter[global_id]
        if global_id in self.track_visitor_sequence_counter:
            del self.track_visitor_sequence_counter[global_id]
        if global_id in self.track_last_centroid_y:
            del self.track_last_centroid_y[global_id]
        if global_id in self.staff_global_ids:
            self.staff_global_ids.remove(global_id)

    def compute_dynamic_queue_depth(self, active_tracks_snapshot: Dict[str, List[Dict[str, Any]]]) -> int:
        """
        Scans active tracking coordinates inside the checkout view, filtering out persistent staff IDs.

        Inputs:
            active_tracks_snapshot (Dict[str, List[Dict[str, Any]]]): Unified frame logs map.

        Outputs:
            int: Total count of active consumer shoppers waiting in the checkout queue lines.
        """
        billing_counter = 0
        billing_records = active_tracks_snapshot.get("CAM_5", [])
        for track in billing_records:
            global_id = track["track_id"]
            if global_id in self.staff_global_ids or track.get("class_id") == 1:
                continue
            x, y = self.calculate_centroid(track["bbox"])
            if self.resolve_spatial_zone("CAM_5", x, y) == "BILLING_ZONE":
                billing_counter += 1
        return billing_counter

    def execute_garbage_collection_sweep(self, active_frame_gids: Set[int], frame_index: int) -> List[Dict[str, Any]]:
        """
        Sweeps inactive memory caches to clean out broken tracks and generate missing termination records.

        Inputs:
            active_frame_gids (Set[int]): Current frame active tracking IDs collection array.
            frame_index (int): Absolute system frame loop index point.

        Outputs:
            List[Dict[str, Any]]: Validated session termination structures created during cache cleanup.
        """
        eviction_logs = []
        cached_gids = list(self.track_zone_history.keys())
        current_timestamp_str = datetime.datetime.utcnow().isoformat() + "Z"

        for gid in cached_gids:
            if gid in active_frame_gids:
                self.track_last_seen_frames[gid] = frame_index
                continue
                
            last_seen = self.track_last_seen_frames.get(gid, frame_index)
            if (frame_index - last_seen) > self.tracking_timeout_frames:
                if gid in self.staff_global_ids:
                    self._evict_identity_state(gid, frame_index)
                    continue

                entry_fr = self.store_entry_frames.get(gid, frame_index)
                total_visit_duration_ms = int(((frame_index - entry_fr) / self.frame_rate) * 1000)
                last_camera = self.track_camera_history.get(gid, "CAM_3")
                last_zone = self.track_zone_history.get(gid, "ENTRANCE_ZONE")
                
                seq = self._get_next_sequence(gid)
                eviction_logs.append(self.construct_json_payload(
                    last_camera, gid, "SESSION_END", current_timestamp_str, last_zone, 
                    total_visit_duration_ms, False, 1.0, seq, frame_index, queue_depth=0
                ))
                
                self._evict_identity_state(gid, frame_index)

        return eviction_logs

    def process_global_tracking_frame(
        self, 
        global_tracks_map: Dict[str, List[Dict[str, Any]]], 
        frame_index: int
    ) -> List[Dict[str, Any]]:
        """
        Processes multi-camera tracking streams to construct transition logs and run validation gates.

        Inputs:
            global_tracks_map (Dict[str, List[Dict[str, Any]]]): Tracker outputs keyed by source cameras.
            frame_index (int): Absolute runtime tracking execution frame index counter.

        Outputs:
            List[Dict[str, Any]]: Structured event payloads matching production data contracts.
        """
        emitted_events_batch = []
        current_timestamp_str = datetime.datetime.utcnow().isoformat() + "Z"
        active_frame_gids = set()
        
        current_queue_depth = self.compute_dynamic_queue_depth(global_tracks_map)

        for camera_id, tracks in global_tracks_map.items():
            for track in tracks:
                global_id = track["track_id"]
                bbox = track["bbox"]
                confidence = track["confidence"]
                is_staff = (track["class_id"] == 1 or global_id in self.staff_global_ids)
                active_frame_gids.add(global_id)

                if is_staff and global_id not in self.staff_global_ids:
                    self.staff_global_ids.add(global_id)

                x, y = self.calculate_centroid(bbox)
                assigned_zone = self.resolve_spatial_zone(camera_id, x, y)
                
                previous_zone = self.track_zone_history.get(global_id)
                previous_camera = self.track_camera_history.get(global_id)

                if previous_zone is None:
                    self.track_zone_history[global_id] = assigned_zone
                    self.track_camera_history[global_id] = camera_id
                    self.store_entry_frames[global_id] = frame_index
                    self.zone_entry_frames[global_id] = frame_index
                    self.track_dwell_emission_counter[global_id] = 1
                    
                    if camera_id == "CAM_3":
                        self.track_last_centroid_y[global_id] = y
                        if y > 540 and not is_staff:
                            seq = self._get_next_sequence(global_id)
                            emitted_events_batch.append(self.construct_json_payload(
                                camera_id, global_id, "ENTRY", current_timestamp_str, None, 0, False, confidence, seq, frame_index, current_queue_depth
                            ))
                    else:
                        if not is_staff:
                            seq = self._get_next_sequence(global_id)
                            emitted_events_batch.append(self.construct_json_payload(
                                camera_id, global_id, "ZONE_ENTER", current_timestamp_str, assigned_zone, 0, False, confidence, seq, frame_index, current_queue_depth
                            ))
                    continue

                if is_staff:
                    if camera_id == "CAM_4" and previous_zone != "STORAGE_ROOM":
                        seq = self._get_next_sequence(global_id)
                        emitted_events_batch.append(self.construct_json_payload(
                            camera_id, global_id, "STAFF_BACKROOM_LOG", current_timestamp_str, "STORAGE_ROOM", 0, True, confidence, seq, frame_index, current_queue_depth
                        ))
                        self.track_zone_history[global_id] = "STORAGE_ROOM"
                    continue

                if camera_id == "CAM_3":
                    prev_y = self.track_last_centroid_y.get(global_id)
                    self.track_last_centroid_y[global_id] = y
                    if prev_y is not None:
                        if prev_y <= 540 and y > 540:
                            seq = self._get_next_sequence(global_id)
                            emitted_events_batch.append(self.construct_json_payload(
                                camera_id, global_id, "ENTRY", current_timestamp_str, None, 0, False, confidence, seq, frame_index, current_queue_depth
                            ))
                        elif prev_y >= 540 and y < 540:
                            store_entry = self.store_entry_frames.get(global_id, frame_index)
                            total_duration = int(((frame_index - store_entry) / self.frame_rate) * 1000)
                            
                            seq_exit = self._get_next_sequence(global_id)
                            emitted_events_batch.append(self.construct_json_payload(
                                camera_id, global_id, "EXIT", current_timestamp_str, None, total_duration, False, confidence, seq_exit, frame_index, current_queue_depth
                            ))
                            
                            seq_end = self._get_next_sequence(global_id)
                            emitted_events_batch.append(self.construct_json_payload(
                                camera_id, global_id, "SESSION_END", current_timestamp_str, previous_zone or "ENTRANCE_ZONE", total_duration, False, confidence, seq_end, frame_index, current_queue_depth
                            ))
                            
                            self._evict_identity_state(global_id, frame_index)
                            continue

                if previous_zone != assigned_zone:
                    zone_entry = self.zone_entry_frames.get(global_id, frame_index)
                    zone_dwell_ms = int(((frame_index - zone_entry) / self.frame_rate) * 1000)
                    
                    exit_camera = previous_camera if previous_camera else camera_id
                    seq_exit = self._get_next_sequence(global_id)
                    emitted_events_batch.append(self.construct_json_payload(
                        exit_camera, global_id, "ZONE_EXIT", current_timestamp_str, previous_zone, zone_dwell_ms, False, confidence, seq_exit, frame_index, current_queue_depth
                    ))

                    if previous_zone == "BILLING_ZONE" and assigned_zone != "ENTRANCE_ZONE" and zone_dwell_ms < 60000:
                        seq_abandon = self._get_next_sequence(global_id)
                        emitted_events_batch.append(self.construct_json_payload(
                            exit_camera, global_id, "QUEUE_ABANDONMENT", current_timestamp_str, previous_zone, zone_dwell_ms, False, confidence, seq_abandon, frame_index, current_queue_depth
                        ))

                    self.track_zone_history[global_id] = assigned_zone
                    self.track_camera_history[global_id] = camera_id
                    self.zone_entry_frames[global_id] = frame_index
                    self.track_dwell_emission_counter[global_id] = 1

                    seq_enter = self._get_next_sequence(global_id)
                    emitted_events_batch.append(self.construct_json_payload(
                        camera_id, global_id, "ZONE_ENTER", current_timestamp_str, assigned_zone, 0, False, confidence, seq_enter, frame_index, current_queue_depth
                    ))

                else:
                    zone_entry = self.zone_entry_frames.get(global_id, frame_index)
                    total_zone_dwell_ms = int(((frame_index - zone_entry) / self.frame_rate) * 1000)
                    emission_count = self.track_dwell_emission_counter.get(global_id, 1)

                    if total_zone_dwell_ms >= (emission_count * 30000):
                        seq_dwell = self._get_next_sequence(global_id)
                        emitted_events_batch.append(self.construct_json_payload(
                            camera_id, global_id, "ZONE_DWELL", current_timestamp_str, assigned_zone, total_zone_dwell_ms, False, confidence, seq_dwell, frame_index, current_queue_depth
                        ))
                        if assigned_zone == "BILLING_ZONE" and emission_count == 1:
                            seq_join = self._get_next_sequence(global_id)
                            emitted_events_batch.append(self.construct_json_payload(
                                camera_id, global_id, "BILLING_QUEUE_JOIN", current_timestamp_str, assigned_zone, total_zone_dwell_ms, False, confidence, seq_join, frame_index, current_queue_depth
                            ))
                        self.track_dwell_emission_counter[global_id] += 1

        eviction_logs = self.execute_garbage_collection_sweep(active_frame_gids, frame_index)
        emitted_events_batch.extend(eviction_logs)

        return emitted_events_batch

    def construct_json_payload(
        self, 
        camera_id: str, 
        global_id: int, 
        event_type: str, 
        timestamp: str, 
        zone_id: Optional[str], 
        dwell_ms: int, 
        is_staff: bool, 
        confidence: float, 
        seq_num: int,
        frame_index: int,
        queue_depth: int
    ) -> Dict[str, Any]:
        """
        Constructs a structured JSON event payload containing multi-temporal metrics.

        Inputs:
            camera_id (str): Operating camera identity token.
            global_id (int): Continuous cross-camera global person tracking ID.
            event_type (str): Categorized classification event label.
            timestamp (str): ISO-8601 string processing time.
            zone_id (Optional[str]): Target store layout department zone name string.
            dwell_ms (int): Target tracking frame duration elapsed in milliseconds.
            is_staff (bool): Uniform detection workforce flag assignment status.
            confidence (float): Object detection score precision scalar value.
            seq_num (int): Ordinal trajectory sequence tracking index.
            frame_index (int): Synchronized video frame timeline step index pointer.
            queue_depth (int): Dynamic evaluation count of people inside the checkout zone.

        Outputs:
            Dict[str, Any]: Standardized event payload structure matching production data contracts.
        """
        video_time_seconds = round(frame_index / self.frame_rate, 2)
        return {
            "event_id": str(uuid.uuid4()),
            "store_id": self.store_id,
            "camera_id": camera_id,
            "visitor_id": f"VIS_GLOBAL_{global_id}",
            "event_type": event_type,
            "timestamp": timestamp,
            "zone_id": zone_id,
            "dwell_ms": dwell_ms,
            "is_staff": is_staff,
            "confidence": round(float(confidence), 2),
            "metadata": {
                "queue_depth": queue_depth if event_type in ["BILLING_QUEUE_JOIN", "ZONE_DWELL"] and zone_id == "BILLING_ZONE" else None,
                "sku_zone": zone_id if zone_id not in ["ENTRANCE_ZONE", "STORAGE_ROOM"] else None,
                "session_seq": seq_num,
                "frame_index": frame_index,
                "video_time_seconds": video_time_seconds
            }
        }