import cv2
import os
from typing import Dict, List, Optional, Tuple ,Tupule, Generator
from ultralytics import YOLO

class MultiCamStreamMultiplexer:
    def __init__(self,video_directory:str,frame_target_width:int =1920, frame_target_height: int =1080):
        """
        Initializes the multi-camera stream multiplexer by mapping the local folder files.

        Inputs:
            video_directory (str): The physical path to the folder holding cam1.mp4 to cam5.mp4.
            frame_target_width (int): Standardized scaling width for uniform matrix dimensions.
            frame_target_height (int): Standardized scaling height for uniform matrix dimensions.
        """

        self.directory = video_directory
        self.width = frame_target_width
        self.height = frame_target_height
        self.camera_ids = [f"CAM {i}" for i in range(1, 6)]
        self.captures = Dict[str, cv2.VideoCapture] = {}

    def initialize_streams(self) -> None:
        """
        Opens file pointers for all 5 cameras and verifies system availability.

        Outputs:
            None. Raises FileNotFoundError if any of the mandatory 5 files are missing.
        """
        for cam in self.camera_ids:
            file_name = f"{cam}.mp4"
            full_path = os.path.join(self.directory, file_name)

            cap = cv2.VideoCapture(full_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Unable to open video file: {full_path}")  
            
            self.captures[cam] = cap

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