import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
#sys.path.append("../")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matplotlib.patches import Polygon, Circle
from typing import List, Tuple, Dict, Any
from common.data_models import SystemDefinition, Boundary, Point, Constraints, Gear
import numpy as np
import json


class Renderer:
    @staticmethod
    def render_system(system: SystemDefinition, output_path: str, 
                      path: List[Tuple[float, float]] = None, 
                      gears: List[Gear] = None) -> None:
        """
        Render a gear system, including outlines of generated gears and their IDs.
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot boundary polygon
        boundary_points = [(p.x, p.y) for p in system.boundary.points]
        boundary_poly = Polygon(boundary_points, closed=True, fill=False, color='black', linewidth=2)
        ax.add_patch(boundary_poly)
        
        # Plot generated gears if provided
        if gears:
            for gear in gears:
                # For compound gears, find the largest radius for annotation positioning
                radii = [d / 2 for d in gear.diameters]
                largest_radius = max(radii)
                
                # Draw a circle outline for each gear in the set
                for radius in radii:
                    gear_circle = Circle(
                        (gear.center.x, gear.center.y), radius,
                        fill=False, # Draw as an outline
                        edgecolor='darkblue', linewidth=1.5
                    )
                    ax.add_patch(gear_circle)
                
                # Add a dot for the center of the gear set
                ax.plot(gear.center.x, gear.center.y, 'ko', markersize=3)

                # Add the gear ID text annotation on the edge of the largest circle
                # Place text at a 45-degree angle from the center
                angle_rad = np.deg2rad(45)
                text_x = gear.center.x + (largest_radius + 1.0) * np.cos(angle_rad) # Small offset
                text_y = gear.center.y - (largest_radius + 1.0) * np.sin(angle_rad) # Y is inverted
                ax.text(text_x, text_y, gear.id, fontsize=9, ha='center', va='center', color='darkgreen')

        # Plot input and output shafts
        ax.plot(system.input_shaft.x, system.input_shaft.y, 'ro', markersize=10, label='Input Shaft')
        ax.plot(system.output_shaft.x, system.output_shaft.y, 'bo', markersize=10, label='Output Shaft')
        
        # Plot path if provided
        if path:
            # print(path)
            path_x = [p.x for p in path]
            path_y = [p.y for p in path]
            ax.plot(path_x, path_y, 'm-', linewidth=1, alpha=0.7, label='Path')
        
        # Set plot properties
        all_x = [p[0] for p in boundary_points]
        all_y = [p[1] for p in boundary_points]
        ax.set_xlim(min(all_x) - 10, max(all_x) + 10)
        ax.set_ylim(min(all_y) - 10, max(all_y) + 10)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        ax.set_title('Gear System Visualization')
        
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    @staticmethod
    def render_processed_data(processed_data_path: str, output_path: str, 
                              path: List[Tuple[float, float]] = None, 
                              gears: List[Gear] = None) -> None:
        """
        Render processed data from JSON, optionally including a path and generated gears.
        """
        with open(processed_data_path, 'r') as f:
            data = json.load(f)
            
        norm_space = data['normalized_space']
        
        # Create a minimal SystemDefinition from the processed data
        system = SystemDefinition(
            boundary=Boundary(points=[Point(x=p[0], y=p[1]) for p in norm_space['boundaries']]),
            input_shaft=Point(x=norm_space['input_shaft']['x'], y=norm_space['input_shaft']['y']),
            output_shaft=Point(x=norm_space['output_shaft']['x'], y=norm_space['output_shaft']['y']),
            constraints=None # Constraints are not needed for rendering
        )
        
        # Render the system with optional path and gears
        Renderer.render_system(system, output_path, path, gears)

    @staticmethod
    def render_path(processed_data_path: str, output_path: str, path: List[Tuple[float, float]]):
        """
        Renders only the normalized boundaries, shafts, and a given path.
        """
        with open(processed_data_path, 'r') as f:
            data = json.load(f)['normalized_space']

        boundary_points = data['boundaries']
        input_shaft = tuple(data['input_shaft'].values())
        output_shaft = tuple(data['output_shaft'].values())

        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot boundary polygon
        boundary_poly = Polygon(boundary_points, closed=True, fill=False, color='black', linewidth=2)
        ax.add_patch(boundary_poly)

        # Plot input and output shafts
        ax.plot(input_shaft[0], input_shaft[1], 'ro', markersize=10, label='Input Shaft')
        ax.plot(output_shaft[0], output_shaft[1], 'bo', markersize=10, label='Output Shaft')

        # print(path)
        # Plot path
        path_x = [p.x for p in path]
        path_y = [p.y for p in path]
        ax.plot(path_x, path_y, 'm-', linewidth=2, label='Path')

        # Set plot properties
        all_x = [p[0] for p in boundary_points]
        all_y = [p[1] for p in boundary_points]
        ax.set_xlim(min(all_x) - 10, max(all_x) + 10)
        ax.set_ylim(min(all_y) - 10, max(all_y) + 10)
        # ax.set_xlim(-60,60)
        # ax.set_ylim(-60,60)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        ax.set_title('Path Visualization')

        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    # In visualization/renderer.py -> class Renderer

    @staticmethod
    def render_clearance(processed_data_path, output_path, clearance_data, path):
        """Rendert de 'clearance corridor' op de afbeelding."""
        fig, ax = Renderer.setup_plot_from_processed_data(processed_data_path)

        # Teken het pad
        path_x = [p.x for p in path]
        path_y = [p.y for p in path]
        ax.plot(path_x, path_y, 'g--', linewidth=1, label='Optimal Path')

        # Teken de clearance cirkels
        for data in clearance_data:
            circle = plt.Circle(
                (data['point_x'], data['point_y']),
                data['max_radius'],
                color='cyan',
                alpha=0.3,
                fill=True,
                linewidth=0
            )
            ax.add_artist(circle)
        
        # Sla de afbeelding op
        plt.legend()
        plt.savefig(output_path, dpi=300)
        plt.close()
    # In visualization/renderer.py -> class Renderer

    @staticmethod
    def setup_plot_from_processed_data(processed_data_path):
        """
        Zet een matplotlib plot op door de achtergrondafbeelding, grenzen,
        en assen uit een processed JSON-bestand te laden.

        Returns:
            fig, ax: De Matplotlib figure en axes objecten.
        """
        with open(processed_data_path, 'r') as f:
            data = json.load(f)

        # Haal de benodigde data op
        img_path = data['background_image_path']
        normalized_space = data['normalized_space']
        boundaries = normalized_space['boundaries']
        input_shaft = normalized_space['input_shaft']
        output_shaft = normalized_space['output_shaft']
        img = mpimg.imread(img_path)
        
        # Zet de plot op
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, extent=[0, img.shape[1], 0, img.shape[0]])
        ax.set_aspect('equal')
        
        # Teken de grenzen
        boundary_x = [p[0] for p in boundaries] + [boundaries[0][0]]
        boundary_y = [p[1] for p in boundaries] + [boundaries[0][1]]
        ax.plot(boundary_x, boundary_y, 'b-', label='Boundary')

        # Teken de assen
        ax.plot(input_shaft['x'], input_shaft['y'], 'go', markersize=10, label='Input Shaft')
        ax.plot(output_shaft['x'], output_shaft['y'], 'ro', markersize=10, label='Output Shaft')

        return fig, ax

if __name__ == "__main__":
    fn = 'Example1'
    
    CONFIG = {
        "INPUT_DIR": "../data",
        "INTERMEDIATE_DIR": "../data/intermediate",
        "EXAMPLE_NAME": f"{fn}",
        "module": 1.0,
        "clearance_margin": 1.0,
        "initial_gear_teeth": 20,
        "OUTPUT_DIR": "../output",
    }
    processed_json_path = os.path.join(CONFIG["INTERMEDIATE_DIR"], f"{CONFIG['EXAMPLE_NAME']}_processed.json")
    path_json_path = os.path.join(CONFIG['OUTPUT_DIR'], 'path.json')
    path_image_path = os.path.join(CONFIG['OUTPUT_DIR'], 'path.png')
    output_dir = os.path.join(CONFIG["OUTPUT_DIR"], CONFIG["EXAMPLE_NAME"])
    output_image_path = os.path.join(output_dir, "gear_train_result.png")
    gear_layout_path = os.path.join(output_dir, "gear_layout.json")

    with open(gear_layout_path, 'r') as f:
        gears_data = json.load(f)

    gears_for_renderer = [Gear.from_json(g) for g in gears_data]

    Renderer.render_processed_data(
        processed_data_path=processed_json_path,
        output_path=output_image_path,
        gears=gears_for_renderer
    )