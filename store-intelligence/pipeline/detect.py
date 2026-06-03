import cv2
import os
import re
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
from ultralytics import YOLO

CAMERA_FILE_PATTERN = re.compile(r"CAM\s*(\d+)", re.IGNORECASE)

class MultiCamStreamMultiplexer:
    def __init__(self, video_directory: str, frame_target_width: int = 1920, frame_target_height: int = 1080):
        """
        Initializes the multi-camera stream multiplexer by mapping the local folder files.

        Inputs:
            video_directory (str): The physical path to the folder holding cam1.mp4 to cam5.mp4.
            frame_target_width (int): Standardized scaling width for uniform matrix dimensions.
            frame_target_height (int): Standardized scaling height for uniform matrix dimensions.
        """

        self.directory = Path(video_directory)
        self.width = frame_target_width
        self.height = frame_target_height
        self.camera_ids: List[str] = []
        self.captures: Dict[str, cv2.VideoCapture] = {}

    def _discover_camera_files(self) -> Dict[str, Path]:
        camera_files: Dict[str, Path] = {}
        if not self.directory.exists():
            return camera_files

        for file_path in self.directory.glob("*.mp4"):
            match = CAMERA_FILE_PATTERN.search(file_path.name)
            if match:
                camera_files[f"CAM_{int(match.group(1))}"] = file_path

        if not camera_files:
            for file_path in self.directory.rglob("*.mp4"):
                match = CAMERA_FILE_PATTERN.search(file_path.name)
                if match:
                    camera_files[f"CAM_{int(match.group(1))}"] = file_path

        return camera_files

    def initialize_streams(self) -> None:
        """
        Opens file pointers for available camera streams and verifies source availability.

        Outputs:
            None. Raises FileNotFoundError if no valid camera video streams are found.
        """
        camera_files = self._discover_camera_files()
        if not camera_files:
            raise FileNotFoundError(f"No camera files found in directory: {self.directory}")

        self.camera_ids = sorted(camera_files.keys())
        for cam_id, video_path in camera_files.items():
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise FileNotFoundError(f"Unable to open video file: {video_path}")
            self.captures[cam_id] = cap

    def yield_synchronized_frames(self) -> Generator[Tuple[int, Dict[str, Optional[cv2.Mat]]], None, None]:
        """
        Generates frame-synchronized matrices across all video streams concurrently.

        Outputs:
            Generator yielding a tuple of:
                - int: Current synchronized frame index pointer.
                - Dict[str, Optional[cv2.Mat]]: Map of camera IDs to their respective frame image arrays.
        """

        frame_idx = 0 
        while True:
            frame_package = {}
            active_streams = 0

            for cam , cap in self.captures.items():
                sucess,frame =cap.read()
                if sucess and frame is not None:
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame,(self.width,self.height))
                    frame_package[cam] = frame
                    active_streams += 1
                else:
                    frame_package[cam] = None
            
            if active_streams == 0:
                break

            yield frame_idx , frame_package
            frame_idx += 1

    def close_all_streams(self) -> None:
        """
        Closes all video stream file pointers to release system resources.

        Outputs:
            None.
        """
        for cap in self.captures.values():
            cap.release()



class BatchStoreDetector:
    def __init__(self, model_weights_path :str = "yolob8x.pt", confidence_threshold: float = 0.25):
        """
        Initializes the batch store detector by loading the YOLOv8 model with specified weights.

        Inputs:
            model_weights_path (str): The file path to the pre-trained YOLOv8 model weights.
            confidence_threshold (float): The minimum confidence score for valid detections.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_weights_path).to(self.device)
        self.confidence_threshold = confidence_threshold
        self.target_classes = [0]

    def execute_bathc_infernece(self, synchronized_frame_package: Dict[str,cv2.Mat]) -> Dict[str,List[Dict[str, any]]]:
        """
        Executes batch inference across the synchronized frame package from all cameras.

        Inputs:
            synchronized_frame_package (Dict[str, cv2.Mat]): A mapping of camera IDs to their respective frame images.
        Outputs:
            Dict[str, List[Dict[str, any]]]: A mapping of camera IDs to their respective lists of detection results.
        """
        camera_order = []
        frame_list = []
        batch_detections_map = {}

        for cam_id, frame in synchronized_frame_package.items():
            if frame is not None:
                camera_order.append(cam_id)
                frame_list.append(frame)
            else :
                batch_detections_map[cam_id] = []
        
        if not frame_list:
            return batch_detections_map
        
        inference_results = self.model.predict(
            source = frame_list,
            device = self.device,
            conf = self.confidence_threshold,
            classes = self.target_classes,
            verbose = False
            )
        
        for idx,result in enumerate(inference_results):
            cam_id = camera_order[idx]
            batch_detections_map[cam_id] = []

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()


            for i in range(len(boxes)):
                detection_payload = {
                    "bbox":[float(coord) for coord in boxes[i]],
                    "confidence": float(scores[i]),
                    "class_id": int(classes[i])
                }
                batch_detections_map[cam_id].append(detection_payload)

        return batch_detections_map