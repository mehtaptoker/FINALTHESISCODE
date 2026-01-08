#####This code is debugged and adapted with help of G00GLE GEMINI############
import json
from typing import List
from common.data_models import Point, Boundary

class EnvironmentManager:
    """
    Manages the geometric environment including obstacles and shaft locations.
    Defines the input and output shaft positions for the system.
    """
    
    # Define fixed input and output shaft locations
    INPUT_SHAFT = Point(x=0.0, y=0.0)
    OUTPUT_SHAFT = Point(x=10.0, y=0.0)

    @staticmethod
    def load_obstacles_from_json(file_path: str) -> List[Boundary]:
        """
        Load obstacle data from a JSON file and convert it into Boundary objects.
        
        Args:
            file_path: Path to the JSON file containing obstacle definitions
            
        Returns:
            List of Boundary objects representing obstacles
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            ValueError: If the JSON structure is invalid
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Obstacle file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {str(e)}")
        
        # Validate JSON structure
        if not isinstance(data, dict) or 'obstacles' not in data:
            raise ValueError("Invalid JSON structure: missing 'obstacles' key")
        
        obstacles = data.get('obstacles', [])
        if not obstacles:
            return []
        
        # Convert JSON obstacles to Boundary objects
        boundary_list = []
        for obstacle in obstacles:
            if not isinstance(obstacle, list) or len(obstacle) < 3:
                raise ValueError("Each obstacle must be a list of at least 3 points")
            
            points = []
            for point in obstacle:
                if not isinstance(point, dict) or 'x' not in point or 'y' not in point:
                    raise ValueError("Invalid point format in obstacle")
                points.append(Point(x=point['x'], y=point['y']))
            
            boundary_list.append(Boundary(vertices=tuple(points)))
        
        return boundary_list