import cv2
import numpy as np
from pathlib import Path
import json
import logging
from typing import List, Dict, Tuple, Optional
import argparse
from datetime import datetime

class DataAnnotator:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        class_names: List[str] = None
    ):
        """
        Initialize the data annotator.
        
        Args:
            input_dir: Directory containing images to annotate
            output_dir: Directory to save annotations
            class_names: List of class names for annotation
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.class_names = class_names or ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian']
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Annotation state
        self.current_image = None
        self.current_annotations = []
        self.current_image_path = None
        self.drawing = False
        self.start_point = None
        self.current_class = 0
        
    def load_image(self, image_path: Path) -> bool:
        """
        Load an image for annotation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if image loaded successfully, False otherwise
        """
        try:
            self.current_image = cv2.imread(str(image_path))
            if self.current_image is None:
                raise ValueError(f"Could not read image: {image_path}")
                
            self.current_image_path = image_path
            self.current_annotations = []
            
            # Load existing annotations if any
            annotation_path = self.output_dir / f"{image_path.stem}.json"
            if annotation_path.exists():
                with open(annotation_path, 'r') as f:
                    self.current_annotations = json.load(f)
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return False
            
    def save_annotations(self) -> bool:
        """
        Save current annotations to a JSON file.
        
        Returns:
            True if annotations saved successfully, False otherwise
        """
        try:
            if not self.current_image_path:
                raise ValueError("No image loaded")
                
            annotation_path = self.output_dir / f"{self.current_image_path.stem}.json"
            with open(annotation_path, 'w') as f:
                json.dump({
                    'image_path': str(self.current_image_path),
                    'timestamp': datetime.now().isoformat(),
                    'annotations': self.current_annotations
                }, f, indent=2)
                
            self.logger.info(f"Annotations saved to: {annotation_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving annotations: {e}")
            return False
            
    def draw_annotations(self, image: np.ndarray) -> np.ndarray:
        """
        Draw current annotations on the image.
        
        Args:
            image: Input image
            
        Returns:
            Image with annotations drawn
        """
        # Create a copy of the image
        annotated = image.copy()
        
        # Draw existing annotations
        for ann in self.current_annotations:
            x1, y1, x2, y2 = map(int, ann['bbox'])
            class_name = self.class_names[ann['class_id']]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw class label
            cv2.putText(
                annotated,
                class_name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            
        # Draw current selection if drawing
        if self.drawing and self.start_point:
            cv2.rectangle(
                annotated,
                self.start_point,
                (self.current_x, self.current_y),
                (255, 0, 0),
                2
            )
            
        return annotated
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for annotation."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_x, self.current_y = x, y
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point:
                # Add annotation
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Ensure x1,y1 is top-left and x2,y2 is bottom-right
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                self.current_annotations.append({
                    'class_id': self.current_class,
                    'bbox': [x1, y1, x2, y2]
                })
                
    def annotate(self) -> None:
        """Run the annotation tool."""
        try:
            # Get list of images
            image_files = sorted([
                f for f in self.input_dir.glob('**/*')
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
            ])
            
            if not image_files:
                raise ValueError(f"No images found in {self.input_dir}")
                
            # Create window and set mouse callback
            cv2.namedWindow('Annotation Tool')
            cv2.setMouseCallback('Annotation Tool', self.mouse_callback)
            
            current_idx = 0
            while current_idx < len(image_files):
                # Load image
                if not self.load_image(image_files[current_idx]):
                    current_idx += 1
                    continue
                    
                while True:
                    # Draw annotations
                    display = self.draw_annotations(self.current_image)
                    
                    # Show class selection
                    class_text = f"Current class: {self.class_names[self.current_class]}"
                    cv2.putText(
                        display,
                        class_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                    
                    # Show instructions
                    cv2.putText(
                        display,
                        "Press 'n' for next, 'p' for previous, 'c' to change class, 's' to save, 'q' to quit",
                        (10, self.current_image.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1
                    )
                    
                    # Display image
                    cv2.imshow('Annotation Tool', display)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('n'):  # Next image
                        self.save_annotations()
                        current_idx += 1
                        break
                    elif key == ord('p'):  # Previous image
                        if current_idx > 0:
                            self.save_annotations()
                            current_idx -= 1
                            break
                    elif key == ord('c'):  # Change class
                        self.current_class = (self.current_class + 1) % len(self.class_names)
                    elif key == ord('s'):  # Save annotations
                        self.save_annotations()
                    elif key == ord('q'):  # Quit
                        self.save_annotations()
                        return
                        
            cv2.destroyAllWindows()
            
        except Exception as e:
            self.logger.error(f"Error during annotation: {e}")
            cv2.destroyAllWindows()
            raise

def main():
    parser = argparse.ArgumentParser(description="Annotate images for traffic detection")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for annotations")
    parser.add_argument("--classes", type=str, nargs='+', help="List of class names")
    
    args = parser.parse_args()
    
    # Initialize annotator
    annotator = DataAnnotator(
        input_dir=args.input,
        output_dir=args.output,
        class_names=args.classes
    )
    
    # Run annotation tool
    annotator.annotate()

if __name__ == "__main__":
    main() 