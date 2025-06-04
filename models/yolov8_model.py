from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import torch
from typing import List, Dict, Tuple, Optional
import logging

class TrafficDetector:
    def __init__(
        self,
        model_path: str = "models/weights/yolov8n.pt",
        confidence: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the traffic detector with YOLOv8 model.
        
        Args:
            model_path: Path to the YOLOv8 model weights
            confidence: Detection confidence threshold
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.confidence = confidence
        self.device = device
        self.model = self._load_model(model_path)
        self.classes = self.model.names
        logging.info(f"Model loaded successfully on {device}")
        
    def _load_model(self, model_path: str) -> YOLO:
        """Load the YOLOv8 model from the specified path."""
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
            
    def detect(
        self,
        frame: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Perform object detection on a single frame.
        
        Args:
            frame: Input image/frame (BGR format)
            classes: List of class IDs to detect (None for all classes)
            
        Returns:
            Tuple of (annotated_frame, detections)
            detections: List of dictionaries containing detection info
        """
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence,
                classes=classes,
                device=self.device
            )[0]
            
            # Process detections
            detections = []
            for box in results.boxes:
                detection = {
                    'class': self.classes[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].cpu().numpy(),
                    'track_id': int(box.id) if box.id is not None else None
                }
                detections.append(detection)
            
            # Draw detections on frame
            annotated_frame = results.plot()
            
            return annotated_frame, detections
            
        except Exception as e:
            logging.error(f"Error during detection: {e}")
            return frame, []
            
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        classes: Optional[List[int]] = None,
        show: bool = False
    ) -> None:
        """
        Process a video file for traffic detection.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save processed video (optional)
            classes: List of class IDs to detect (None for all classes)
            show: Whether to display the video while processing
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Initialize video writer if output path is provided
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Perform detection
                annotated_frame, detections = self.detect(frame, classes)
                
                # Write frame if output is specified
                if writer:
                    writer.write(annotated_frame)
                    
                # Display frame if show is True
                if show:
                    cv2.imshow('Traffic Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
                
        except Exception as e:
            logging.error(f"Error processing video: {e}")
            raise
            
    def get_traffic_stats(self, detections: List[Dict]) -> Dict:
        """
        Calculate traffic statistics from detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary containing traffic statistics
        """
        stats = {
            'total_vehicles': 0,
            'vehicle_types': {},
            'average_confidence': 0.0
        }
        
        if not detections:
            return stats
            
        # Count vehicles by type
        for det in detections:
            vehicle_type = det['class']
            stats['vehicle_types'][vehicle_type] = stats['vehicle_types'].get(vehicle_type, 0) + 1
            stats['total_vehicles'] += 1
            stats['average_confidence'] += det['confidence']
            
        # Calculate average confidence
        stats['average_confidence'] /= len(detections)
        
        return stats 