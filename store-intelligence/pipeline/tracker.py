import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Any, Tuple, Optional

class FeatureExtractionBackbone:
    def __init__(self, embedding_dim: int = 512):
        """
        Initializes a structural image patch descriptor processing pipeline.

        Inputs:
            embedding_dim (int): Vector length for the high-dimensional appearance space.
        """
        self.embedding_dim = embedding_dim

    def extract_appearance_embedding(self, frame_patch: np.ndarray) -> List[float]:
        """
        Extracts a unit-normalized appearance vector directly from the color channel 
        distributions of a physical bounding box frame patch.

        Inputs:
            frame_patch (np.ndarray): Cropped sub-matrix array representing a person.

        Outputs:
            List[float]: A unique, L2 normalized visual appearance feature vector.
        """
        if frame_patch is None or frame_patch.size == 0:
            raw_vector = np.zeros(self.embedding_dim, dtype=np.float32)
            raw_vector[0] = 1.0
            return raw_vector.tolist()

        resized_patch = float(np.mean(frame_patch)) * np.ones(self.embedding_dim, dtype=np.float32)
        
        channels = [frame_patch[:, :, c] for c in range(min(3, frame_patch.shape[2]))] if len(frame_patch.shape) == 3 else [frame_patch]
        for idx, channel in enumerate(channels):
            hist, _ = np.histogram(channel, bins=min(64, self.embedding_dim // 4), range=(0, 256))
            start_pos = idx * len(hist)
            if start_pos + len(hist) <= self.embedding_dim:
                resized_patch[start_pos:start_pos + len(hist)] = hist

        normalized_vector = resized_patch / (np.linalg.norm(resized_patch) + 1e-6)
        return normalized_vector.tolist()


class KalmanBoxTracker:
    def __init__(self, bbox: List[float]):
        """
        Initializes an internal state-space tracking system for bounding box trajectories.

        Inputs:
            bbox (List[float]): Bounding box coordinates formatted as [xmin, ymin, xmax, ymax].
        """
        xmin, ymin, xmax, ymax = bbox
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        h = ymax - ymin
        r = (xmax - xmin) / float(h) if h > 0 else 0.0

        self.mean = np.zeros(8)
        self.mean[:4] = [cx, cy, r, h]

        self.covariance = np.diag([
            2 * 1e-2 * h, 2 * 1e-2 * h, 1e-2, 2 * 1e-2 * h,
            10 * 1e-5 * h, 10 * 1e-5 * h, 1e-5, 10 * 1e-5 * h
        ]) ** 2

        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.eye(4, 8)

    def predict(self) -> None:
        """
        Advances the internal motion tracking matrix state using linear kinematic velocity parameters.
        """
        h = self.mean[3]
        motion_noise = np.diag([
            1e-2 * h, 1e-2 * h, 1e-2, 1e-2 * h,
            1e-5 * h, 1e-5 * h, 1e-5, 1e-5 * h
        ]) ** 2

        self.mean = np.dot(self.F, self.mean)
        self.covariance = np.dot(np.dot(self.F, self.covariance), self.F.T) + motion_noise

    def update(self, bbox: List[float]) -> None:
        """
        Corrects internal velocity vectors based on real-world observation matrices.

        Inputs:
            bbox (List[float]): Measured bounding box coordinates formatted as [xmin, ymin, xmax, ymax].
        """
        xmin, ymin, xmax, ymax = bbox
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        h = ymax - ymin
        r = (xmax - xmin) / float(h) if h > 0 else 0.0
        measurement = np.array([cx, cy, r, h])

        measurement_noise = np.diag([
            1e-1 * h, 1e-1 * h, 1e-1, 1e-1 * h
        ]) ** 2

        innovation = measurement - np.dot(self.H, self.mean)
        innovation_covariance = np.dot(np.dot(self.H, self.covariance), self.H.T) + measurement_noise
        kalman_gain = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(innovation_covariance))
        
        self.mean = self.mean + np.dot(kalman_gain, innovation)
        self.covariance = self.covariance - np.dot(np.dot(kalman_gain, self.H), self.covariance)

    def to_xyxy(self) -> List[float]:
        """
        Converts the state representation back to standard pixel boundary coordinate formats.

        Outputs:
            List[float]: Output coordinates formatted as [xmin, ymin, xmax, ymax].
        """
        cx, cy, r, h = self.mean[:4]
        w = r * h
        return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


class STrack:
    def __init__(self, bbox: List[float], score: float, is_staff: bool, embedding: List[float], frame_id: int):
        """
        Manages identity assignments, performance counters, and appearance embeddings for individual tracks.

        Inputs:
            bbox (List[float]): Initial coordinate boundary array.
            score (float): Model object detection confidence metric.
            is_staff (bool): Workforce uniform tracking identification parameter.
            embedding (List[float]): Raw tracking descriptors.
            frame_id (int): Absolute system execution index counter.
        """
        self.bbox = bbox
        self.score = score
        self.is_staff = is_staff
        
        raw_emb = np.array(embedding, dtype=np.float32)
        self.embedding = raw_emb / (np.linalg.norm(raw_emb) + 1e-6)
        
        self.track_id = 0
        self.state = "Unconfirmed"
        self.kalman_filter = KalmanBoxTracker(bbox)
        self.start_frame = frame_id
        self.frame_id = frame_id
        self.time_since_update = 0

    def predict(self) -> None:
        """
        Triggers projection updates within the underlying Kalman architecture.
        """
        self.kalman_filter.predict()
        self.bbox = self.kalman_filter.to_xyxy()
        self.time_since_update += 1

    def update(self, new_track: "STrack", frame_id: int) -> None:
        """
        Synchronizes track parameters and merges appearance features using an exponential moving average.

        Inputs:
            new_track (STrack): Incoming target representation block containing updated metrics.
            frame_id (int): Absolute system execution index counter.
        """
        self.kalman_filter.update(new_track.bbox)
        self.bbox = self.kalman_filter.to_xyxy()
        self.score = new_track.score
        self.is_staff = new_track.is_staff
        
        self.embedding = (0.90 * self.embedding) + (0.10 * new_track.embedding)
        self.embedding /= (np.linalg.norm(self.embedding) + 1e-6)
        
        self.state = "Tracked"
        self.frame_id = frame_id
        self.time_since_update = 0

    def mark_lost(self) -> None:
        """
        Updates the track state when an object is temporarily obscured or unassigned.
        """
        self.state = "Lost"

    def mark_removed(self) -> None:
        """
        Permanently deactivates and flags tracking data layers for system removal.
        """
        self.state = "Removed"


class SingleCamByteTracker:
    def __init__(self, det_thresh: float = 0.60, match_thresh: float = 0.80, max_time_lost: int = 45):
        """
        Implements a localized dual-threshold object tracking pipeline for a single camera stream.

        Inputs:
            det_thresh (float): Minimum score separating high-confidence primary targets.
            match_thresh (float): Maximum allowed distance penalty to accept an intersection match.
            max_time_lost (int): Frame limit to preserve lost targets before extraction removal.
        """
        self.det_thresh = det_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.unconfirmed_stracks: List[STrack] = []
        self.next_id = 1
        self.frame_id = 0

    def calculate_iou_distance_matrix(self, tracks: List[STrack], detections: List[STrack]) -> np.ndarray:
        """
        Computes an assignment distance penalty matrix based on physical intersection metrics.

        Inputs:
            tracks (List[STrack]): Active tracking trajectories.
            detections (List[STrack]): Raw coordinate inputs found in the active frame step.

        Outputs:
            np.ndarray: Evaluated intersection distance costs mapped as a 2D float tracking table grid.
        """
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        if cost_matrix.size == 0:
            return cost_matrix

        for t_idx, track in enumerate(tracks):
            for d_idx, det in enumerate(detections):
                box_a = track.bbox
                box_b = det.bbox
                xmin_i = max(box_a[0], box_b[0])
                ymin_i = max(box_a[1], box_b[1])
                xmax_i = min(box_a[2], box_b[2])
                ymax_i = min(box_a[3], box_b[3])
                iw = max(0.0, xmax_i - xmin_i)
                ih = max(0.0, ymax_i - ymin_i)
                intersection = iw * ih
                area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
                area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
                union = area_a + area_b - intersection
                iou = intersection / union if union > 0 else 0.0
                cost_matrix[t_idx, d_idx] = 1.0 - iou

        return cost_matrix

    def remove_duplicate_tracks(self, pool_a: List[STrack], pool_b: List[STrack]) -> Tuple[List[STrack], List[STrack]]:
        """
        Suppresses duplicate tracks monitoring the same physical space by verifying true trajectory age longevity.

        Inputs:
            pool_a (List[STrack]): Active tracks array.
            pool_b (List[STrack]): Target evaluation tracks array.

        Outputs:
            Tuple[List[STrack], List[STrack]]: Cleaned lists with overlapping tracks eliminated.
        """
        if not pool_a or not pool_b:
            return pool_a, pool_b
        cost_matrix = self.calculate_iou_distance_matrix(pool_a, pool_b)
        dupes_a = set()
        dupes_b = set()
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                if cost_matrix[i, j] < 0.15:
                    age_a = pool_a[i].frame_id - pool_a[i].start_frame
                    age_b = pool_b[j].frame_id - pool_b[j].start_frame
                    if age_a > age_b:
                        dupes_b.add(j)
                    else:
                        dupes_a.add(i)
        clean_a = [pool_a[i] for i in range(len(pool_a)) if i not in dupes_a]
        clean_b = [pool_b[j] for j in range(len(pool_b)) if j not in dupes_b]
        return clean_a, clean_b

    def track_frame_detections(self, detections_list: List[Dict[str, Any]]) -> List[STrack]:
        """
        Executes multi-tier global assignment computations across all active track states for a single camera view.

        Inputs:
            detections_list (List[Dict[str, Any]]): Raw bounding box configurations and features found by detection layers.

        Outputs:
            List[STrack]: Finalized tracking outputs bound to locally optimized tracking tracks.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks_pool = []
        high_dets = []
        low_dets = []

        for det in detections_list:
            strack = STrack(det["bbox"], det["confidence"], (det["class_id"] == 1), det["embedding"], self.frame_id)
            if det["confidence"] >= self.det_thresh:
                high_dets.append(strack)
            elif det["confidence"] >= 0.10:
                low_dets.append(strack)

        active_track_pool: List[STrack] = []
        for strack in self.tracked_stracks:
            strack.predict()
            active_track_pool.append(strack)
        for strack in self.lost_stracks:
            strack.predict()
            active_track_pool.append(strack)

        cost_matrix_pass1 = self.calculate_iou_distance_matrix(active_track_pool, high_dets)
        matched_tracks_pass1 = set()
        matched_dets_pass1 = set()

        if cost_matrix_pass1.size > 0:
            t_indices, d_indices = linear_sum_assignment(cost_matrix_pass1)
            for t_idx, d_idx in zip(t_indices, d_indices):
                if cost_matrix_pass1[t_idx, d_idx] <= self.match_thresh:
                    track = active_track_pool[t_idx]
                    det = high_dets[d_idx]
                    if track.state == "Tracked":
                        track.update(det, self.frame_id)
                        activated_stracks.append(track)
                    else:
                        track.update(det, self.frame_id)
                        refind_stracks.append(track)
                    matched_tracks_pass1.add(t_idx)
                    matched_dets_pass1.add(d_idx)

        unmatched_tracks_pass1 = [active_track_pool[i] for i in range(len(active_track_pool)) if i not in matched_tracks_pass1 and active_track_pool[i].state == "Tracked"]
        cost_matrix_pass2 = self.calculate_iou_distance_matrix(unmatched_tracks_pass1, low_dets)
        matched_tracks_pass2 = set()

        if cost_matrix_pass2.size > 0:
            t_indices_p2, d_indices_p2 = linear_sum_assignment(cost_matrix_pass2)
            for t_idx, d_idx in zip(t_indices_p2, d_indices_p2):
                if cost_matrix_pass2[t_idx, d_idx] <= 0.50:
                    track = unmatched_tracks_pass1[t_idx]
                    det = low_dets[d_idx]
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                    matched_tracks_pass2.add(track)

        for track in unmatched_tracks_pass1:
            if track not in matched_tracks_pass2:
                track.mark_lost()
                lost_stracks_pool.append(track)

        unmatched_dets_pass1_list = [high_dets[i] for i in range(len(high_dets)) if i not in matched_dets_pass1]
        for track in self.unconfirmed_stracks:
            track.predict()
        
        cost_matrix_unconfirmed = self.calculate_iou_distance_matrix(self.unconfirmed_stracks, unmatched_dets_pass1_list)
        matched_tracks_uc = set()
        matched_dets_uc = set()

        if cost_matrix_unconfirmed.size > 0:
            t_indices_uc, d_indices_uc = linear_sum_assignment(cost_matrix_unconfirmed)
            for t_idx, d_idx in zip(t_indices_uc, d_indices_uc):
                if cost_matrix_unconfirmed[t_idx, d_idx] <= 0.70:
                    track = self.unconfirmed_stracks[t_idx]
                    det = unmatched_dets_pass1_list[d_idx]
                    track.update(det, self.frame_id)
                    track.state = "Tracked"
                    activated_stracks.append(track)
                    matched_tracks_uc.add(t_idx)
                    matched_dets_uc.add(d_idx)

        for t_idx, track in enumerate(self.unconfirmed_stracks):
            if t_idx not in matched_tracks_uc:
                track.mark_removed()

        final_unmatched_dets = [unmatched_dets_pass1_list[i] for i in range(len(unmatched_dets_pass1_list)) if i not in matched_dets_uc]
        for det in final_unmatched_dets:
            if det.score >= self.det_thresh:
                det.track_id = self.next_id
                self.next_id += 1
                det.state = "Unconfirmed"
                self.unconfirmed_stracks.append(det)

        self.tracked_stracks = [t for t in activated_stracks if t.state == "Tracked"]
        for t in refind_stracks:
            if t not in self.tracked_stracks:
                self.tracked_stracks.append(t)

        self.lost_stracks = [t for t in self.lost_stracks if t.state == "Lost"]
        for t in lost_stracks_pool:
            if t not in self.lost_stracks:
                self.lost_stracks.append(t)

        self.lost_stracks = [t for t in self.lost_stracks if t.time_since_update <= self.max_time_lost]
        self.unconfirmed_stracks = [t for t in self.unconfirmed_stracks if t.state == "Unconfirmed" and t.time_since_update <= 3]
        
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_tracks(self.tracked_stracks, self.lost_stracks)
        return self.tracked_stracks


class GlobalMultiCamReIDTracker:
    def __init__(self, reid_threshold: float = 0.75, max_memory_frames: int = 450):
        """
        Coordinates cross-camera tracking loops by evaluating visual embeddings and assigning global IDs.

        Inputs:
            reid_threshold (float): Minimum cosine similarity to merge cross-camera identities.
            max_memory_frames (int): Maximum history duration to preserve visual features for unassigned tracks.
        """
        self.reid_threshold = reid_threshold
        self.max_memory_frames = max_memory_frames
        self.camera_trackers = {
            "CAM_1": SingleCamByteTracker(), "CAM_2": SingleCamByteTracker(),
            "CAM_3": SingleCamByteTracker(), "CAM_4": SingleCamByteTracker(),
            "CAM_5": SingleCamByteTracker()
        }
        self.global_id_mapping: Dict[Tuple[str, int], int] = {}
        self.global_appearance_pool: Dict[int, Dict[str, Any]] = {}
        
        self.camera_transition_cost_matrix = {
            ("CAM_3", "CAM_1"): 0.0, ("CAM_3", "CAM_2"): 0.0, ("CAM_3", "CAM_5"): 0.8,
            ("CAM_1", "CAM_2"): 0.1, ("CAM_1", "CAM_5"): 0.3, ("CAM_2", "CAM_5"): 0.3,
            ("CAM_1", "CAM_4"): 0.5, ("CAM_2", "CAM_4"): 0.5, ("CAM_5", "CAM_4"): 0.9
        }
        
        self.next_global_id = 1
        self.frame_counter = 0

    def resolve_topology_penalty(self, last_camera: str, current_camera: str) -> float:
        """
        Retrieves directional penalty weights between store zones safely, protecting zero-value values.

        Inputs:
            last_camera (str): Name string of the source camera profile.
            current_camera (str): Name string of the destination camera profile.

        Outputs:
            float: Extracted penalty index bound between 0.0 and 1.0.
        """
        if (last_camera, current_camera) in self.camera_transition_cost_matrix:
            return self.camera_transition_cost_matrix[(last_camera, current_camera)]
        if (current_camera, last_camera) in self.camera_transition_cost_matrix:
            return self.camera_transition_cost_matrix[(current_camera, last_camera)]
        return 0.0

    def track_store_features_globally(self, multi_cam_detections: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fuses localized tracking streams into a unified cross-camera identity graph while validating spatial-temporal gating.

        Inputs:
            multi_cam_detections (Dict[str, List[Dict[str, Any]]]): Raw frame detections map keyed by camera labels.

        Outputs:
            Dict[str, List[Dict[str, Any]]]: Unified event stream outputs keyed by store camera labels.
        """
        self.frame_counter += 1
        active_local_tracks_map: Dict[Tuple[str, int], STrack] = {}
        unmapped_local_keys: List[Tuple[str, int]] = []

        for cam_id, detections in multi_cam_detections.items():
            local_tracker = self.camera_trackers[cam_id]
            confirmed_stracks = local_tracker.track_frame_detections(detections)
            
            for strack in confirmed_stracks:
                lookup_key = (cam_id, strack.track_id)
                active_local_tracks_map[lookup_key] = strack
                if lookup_key not in self.global_id_mapping:
                    unmapped_local_keys.append(lookup_key)

        dead_gids = set()
        for gid, profile in list(self.global_appearance_pool.items()):
            if (self.frame_counter - profile["last_seen_frame"]) > self.max_memory_frames:
                dead_gids.add(gid)
                del self.global_appearance_pool[gid]

        for lookup_key, gid in list(self.global_id_mapping.items()):
            if gid in dead_gids or lookup_key not in active_local_tracks_map:
                cam_tracker = self.camera_trackers[lookup_key[0]]
                is_dead = all(t.track_id != lookup_key[1] for t in cam_tracker.tracked_stracks + cam_tracker.lost_stracks)
                if is_dead or gid in dead_gids:
                    del self.global_id_mapping[lookup_key]

        if unmapped_local_keys and self.global_appearance_pool:
            pool_ids = []
            pool_vectors = []
            
            for gid, profile in self.global_appearance_pool.items():
                pool_ids.append(gid)
                pool_vectors.append(profile["embedding"])

            pool_matrix = np.array(pool_vectors, dtype=np.float32)
            current_matrix = np.array([active_local_tracks_map[k].embedding for k in unmapped_local_keys], dtype=np.float32)
            
            similarity_matrix = np.dot(pool_matrix, current_matrix.T)
            cost_matrix = 1.0 - similarity_matrix

            if cost_matrix.size > 0:
                p_indices, c_indices = linear_sum_assignment(cost_matrix)
                active_assigned_gids = set(self.global_id_mapping.values())

                for p_idx, c_idx in zip(p_indices, c_indices):
                    assigned_gid = pool_ids[p_idx]
                    if assigned_gid in active_assigned_gids:
                        continue

                    appearance_sim = float(similarity_matrix[p_idx, c_idx])
                    if appearance_sim >= self.reid_threshold:
                        target_key = unmapped_local_keys[c_idx]
                        profile = self.global_appearance_pool[assigned_gid]
                        
                        topology_penalty = self.resolve_topology_penalty(profile["last_camera"], target_key[0])
                        time_delta_frames = self.frame_counter - profile["last_seen_frame"]

                        if time_delta_frames == 0 and profile["last_camera"] != target_key[0]:
                            continue
                        if time_delta_frames < 15 and topology_penalty > 0.5:
                            continue

                        self.global_id_mapping[target_key] = assigned_gid
                        active_assigned_gids.add(assigned_gid)

        for key in unmapped_local_keys:
            if key not in self.global_id_mapping:
                self.global_id_mapping[key] = self.next_global_id
                self.next_global_id += 1

        global_tracking_output_map = {}
        for (cam_id, local_id), strack in active_local_tracks_map.items():
            global_id = self.global_id_mapping[(cam_id, local_id)]
            
            if global_id in self.global_appearance_pool:
                old_emb = self.global_appearance_pool[global_id]["embedding"]
                blended_emb = (0.90 * old_emb) + (0.10 * strack.embedding)
                blended_emb /= (np.linalg.norm(blended_emb) + 1e-6)
            else:
                blended_emb = strack.embedding

            self.global_appearance_pool[global_id] = {
                "embedding": blended_emb,
                "last_seen_frame": self.frame_counter,
                "last_camera": cam_id
            }
            
            global_tracking_output_map.setdefault(cam_id, []).append({
                "bbox": strack.bbox,
                "confidence": float(strack.score),
                "class_id": 1 if strack.is_staff else 0,
                "track_id": global_id
            })

        for cam_id in multi_cam_detections.keys():
            if cam_id not in global_tracking_output_map:
                global_tracking_output_map[cam_id] = []

        return global_tracking_output_map