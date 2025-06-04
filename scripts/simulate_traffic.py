import numpy as np
import cv2
import random
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime, timedelta

class TrafficSimulator:
    def __init__(
        self,
        output_dir: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        duration: int = 60,
        traffic_density: float = 0.5
    ):
        """
        Initialize the traffic simulator.
        
        Args:
            output_dir: Directory to save simulation output
            width: Video width
            height: Video height
            fps: Frames per second
            duration: Simulation duration in seconds
            traffic_density: Traffic density (0.0 to 1.0)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.traffic_density = max(0.0, min(1.0, traffic_density))
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Vehicle types and their properties
        self.vehicle_types = {
            'car': {
                'color': (0, 0, 255),  # Red
                'size': (60, 30),
                'speed_range': (2, 5)
            },
            'truck': {
                'color': (0, 255, 0),  # Green
                'size': (80, 40),
                'speed_range': (1, 3)
            },
            'bus': {
                'color': (255, 0, 0),  # Blue
                'size': (70, 35),
                'speed_range': (1, 4)
            }
        }
        
    def create_road(self) -> np.ndarray:
        """Create a road background image."""
        # Create blank image
        road = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw road (gray)
        road[:] = (100, 100, 100)
        
        # Draw lane markings
        lane_width = self.width // 4
        for x in range(lane_width, self.width, lane_width):
            cv2.line(road, (x, 0), (x, self.height), (255, 255, 255), 2)
            
        return road
        
    def create_vehicle(
        self,
        vehicle_type: str,
        position: Tuple[int, int],
        direction: str
    ) -> Dict:
        """
        Create a vehicle with random properties.
        
        Args:
            vehicle_type: Type of vehicle ('car', 'truck', 'bus')
            position: Initial position (x, y)
            direction: Movement direction ('left', 'right')
            
        Returns:
            Dictionary containing vehicle properties
        """
        props = self.vehicle_types[vehicle_type]
        speed = random.uniform(*props['speed_range'])
        
        return {
            'type': vehicle_type,
            'position': list(position),
            'size': props['size'],
            'color': props['color'],
            'speed': speed,
            'direction': direction
        }
        
    def update_vehicle(self, vehicle: Dict) -> None:
        """Update vehicle position based on its speed and direction."""
        if vehicle['direction'] == 'right':
            vehicle['position'][0] += vehicle['speed']
        else:
            vehicle['position'][0] -= vehicle['speed']
            
    def draw_vehicle(self, frame: np.ndarray, vehicle: Dict) -> None:
        """Draw a vehicle on the frame."""
        x, y = map(int, vehicle['position'])
        w, h = vehicle['size']
        
        # Draw vehicle rectangle
        cv2.rectangle(
            frame,
            (x - w//2, y - h//2),
            (x + w//2, y + h//2),
            vehicle['color'],
            -1
        )
        
    def generate_traffic_data(self) -> List[Dict]:
        """Generate traffic data for annotation."""
        traffic_data = []
        timestamp = datetime.now()
        
        for _ in range(int(self.duration * self.fps * self.traffic_density)):
            # Randomly select vehicle type and properties
            vehicle_type = random.choice(list(self.vehicle_types.keys()))
            direction = random.choice(['left', 'right'])
            y = random.randint(self.height//4, 3*self.height//4)
            x = 0 if direction == 'right' else self.width
            
            vehicle = self.create_vehicle(vehicle_type, (x, y), direction)
            traffic_data.append({
                'timestamp': timestamp.isoformat(),
                'vehicle': vehicle
            })
            timestamp += timedelta(seconds=1/self.fps)
            
        return traffic_data
        
    def simulate(self, save_annotation: bool = True) -> Optional[Path]:
        """
        Run the traffic simulation.
        
        Args:
            save_annotation: Whether to save traffic data annotation
            
        Returns:
            Path to the output video file
        """
        try:
            # Create video writer
            output_path = self.output_dir / f"traffic_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
            
            # Generate traffic data
            traffic_data = self.generate_traffic_data()
            vehicles = []
            frame_count = 0
            
            while frame_count < self.duration * self.fps:
                # Create frame
                frame = self.create_road()
                
                # Add new vehicles
                if random.random() < self.traffic_density:
                    vehicle_type = random.choice(list(self.vehicle_types.keys()))
                    direction = random.choice(['left', 'right'])
                    y = random.randint(self.height//4, 3*self.height//4)
                    x = 0 if direction == 'right' else self.width
                    vehicles.append(self.create_vehicle(vehicle_type, (x, y), direction))
                    
                # Update and draw vehicles
                vehicles = [v for v in vehicles if 0 <= v['position'][0] <= self.width]
                for vehicle in vehicles:
                    self.update_vehicle(vehicle)
                    self.draw_vehicle(frame, vehicle)
                    
                # Write frame
                writer.write(frame)
                frame_count += 1
                
            # Cleanup
            writer.release()
            
            # Save annotation if requested
            if save_annotation:
                annotation_path = output_path.with_suffix('.json')
                with open(annotation_path, 'w') as f:
                    json.dump({
                        'simulation_info': {
                            'width': self.width,
                            'height': self.height,
                            'fps': self.fps,
                            'duration': self.duration,
                            'traffic_density': self.traffic_density
                        },
                        'traffic_data': traffic_data
                    }, f, indent=2)
                    
            self.logger.info(f"Simulation saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error during simulation: {e}")
            return None

def main():
    # Example usage
    simulator = TrafficSimulator(
        output_dir="data/raw",
        width=1280,
        height=720,
        fps=30,
        duration=60,
        traffic_density=0.5
    )
    simulator.simulate()

if __name__ == "__main__":
    main() 