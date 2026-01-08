#####This code is debugged and adapted with help of G00GLE GEMINI############
mport json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches  # Voor de legenda
import sys
import os
import numpy as np 

def plot_gear_layout(config_data, layout_data, save_path=None):
    """
    Plots the gear train layout.
    AANGEPAST: Tekent nu ENKEL de STEEKCIRKELS (pitch circles).
    """
    
    # --- Setup Plot ---
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal') 
    
    # --- AANGEPAST: Gereduceerde kleuren dictionary ---
    colors = {
        'pitch_circle_1': 'blue',
        'pitch_circle_2': 'lightcoral',
        'center_point': 'black'
    }

    # --- 1. Load Boundaries and Shafts from Config Data ---
    try:
        problem_space = config_data['normalized_space']
        boundaries = problem_space['boundaries']
        input_shaft = problem_space['input_shaft']
        output_shaft = problem_space['output_shaft']
    except KeyError as e:
        print(f"Error: Config JSON is missing key: {e}. Check 'normalized_space' and its children.")
        return

    # --- 2. Plot Boundaries ---
    try:
        boundary_points = [(p[0], p[1]) for p in boundaries]
        
        if boundary_points:
            boundary_poly = patches.Polygon(boundary_points, closed=True, facecolor='none', edgecolor='black', linewidth=2, zorder=1)
            ax.add_patch(boundary_poly)
    except (TypeError, IndexError):
        print("Error: Boundary data is not in the expected [[x, y], ...] format.")
        return

    # --- 3. Plot Input/Output Shafts ---
    try:
        input_pos = (input_shaft['x'], input_shaft['y'])
        output_pos = (output_shaft['x'], output_shaft['y'])
    except KeyError:
        print("Error: Shaft data is not in the expected {'x': ..., 'y': ...} format.")
        return
    
    ax.plot(input_pos[0], input_pos[1], 'ro', markersize=10, label='Input Shaft', zorder=10) 
    ax.plot(output_pos[0], output_pos[1], 'bo', markersize=10, label='Output Shaft', zorder=10) 

    # --- 4. Load and Plot Placed Gears from Layout Data ---
    try:
        placed_gears = layout_data['gears']
        final_reward = layout_data.get('final_reward', 'N/A')
    except KeyError:
        print("Error: Layout JSON is missing 'placed_gears'.")
        return

    print(f"Plotting {len(placed_gears)} gears (pitch circles only) from layout file...")

    for gear in placed_gears:
        center = (gear['center']['x'], gear['center']['y'])
        
        module = gear.get('module', 1.0)
        teeth_count = gear['teeth_count']
        
        if not isinstance(teeth_count, list):
            teeth_count_list = [teeth_count]
        else:
            teeth_count_list = teeth_count

        driven_teeth = teeth_count_list[0]
        driving_teeth = teeth_count_list[-1]

        # --- BEREKENING RADII (Alleen Pitch) ---
        driven_pitch_radius = (module * driven_teeth) / 2.0
        driving_pitch_radius = (module * driving_teeth) / 2.0
        # --- EINDE BEREKENING ---
        
        # --- VERWIJDERD: Teken de BUITENSTE Cirkel (Addendum) ---
        # outer_radius = max(driven_outer_radius, driving_outer_radius)
        # circle_outer = plt.Circle(center, outer_radius, color=..., fill=False, lw=2)
        # ax.add_artist(circle_outer)
        
        # --- VERWIJDERD: Teken de binnenste addendum cirkel ---
        # if driving_teeth != driven_teeth:
        #     inner_radius = min(driven_outer_radius, driving_outer_radius)
        #     circle_inner = plt.Circle(center, inner_radius, color=..., fill=False, lw=1)
        #     ax.add_artist(circle_inner)

        # --- Teken de STEECIRKELS (Pitch Circles) ---
        pitch_radius_1 = max(driven_pitch_radius, driving_pitch_radius)
        circle_pitch_1 = plt.Circle(center, pitch_radius_1, color=colors['pitch_circle_1'], fill=False, lw=1, linestyle='solid')
        ax.add_artist(circle_pitch_1)
        
        if driving_teeth != driven_teeth:
            pitch_radius_2 = min(driven_pitch_radius, driving_pitch_radius)
            circle_pitch_2 = plt.Circle(center, pitch_radius_2, color=colors['pitch_circle_2'], fill=False, lw=1, linestyle='solid')
            ax.add_artist(circle_pitch_2)

        # --- VERWIJDERD: Teken de WORTELCIRKELS (Root Circles) ---
        # ...
            
        ax.plot(center[0], center[1], 'o', color=colors['center_point'], markersize=4) # Center punt
        ax.text(center[0] + 1, center[1] + 1, gear['id'], color='green', fontsize=9)

    # --- 6. Finalize Plot ---
    if boundary_points:
        all_x = [p[0] for p in boundary_points]
        all_y = [p[1] for p in boundary_points]
        padding = 1.0
        ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
        ax.set_ylim(max(all_y) + padding, min(all_y) - padding) 

    ax.set_title("Gear System Visualization (Pitch Circles Only)", fontsize=16)
    ax.set_xlabel('X Coordinate (mm)', fontsize=10)
    ax.set_ylabel('Y Coordinate (mm)', fontsize=10)
    
    # --- AANGEPAST: Gereduceerde legenda ---
    handles, labels = ax.get_legend_handles_labels()
    
    legend_patches = [
        mpatches.Patch(edgecolor=colors['pitch_circle_1'], facecolor='none', label='Pitch Circle (Largest)', linestyle='solid', linewidth=1),
        mpatches.Patch(edgecolor=colors['pitch_circle_2'], facecolor='none', label='Pitch Circle (Smallest)', linestyle='solid', linewidth=1),
        plt.Line2D([0], [0], marker='o', color='none', label='Center Point', markerfacecolor=colors['center_point'], markersize=4)
        
    ]
    
    #ax.legend(handles=handles + legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    ax.legend(handles=handles + legend_patches, loc='upper left', bbox_to_anchor=(1.00, 1.0), borderaxespad=0.)
    # --- EINDE AANPASSING ---
    
    ax.grid(True, linestyle='solid', alpha=0.5, color="#AAAAAA") 
    
    plt.tight_layout(rect=[0, 0, 0.8, 1]) 
    
    # --- 7. Save and Show Plot ---
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ Plot successfully saved to: {save_path}")
        except Exception as e:
            print(f"\n⚠️ Error saving plot: {e}")
            
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize a saved gear train layout (Pitch Circles Only).')
    
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the problem/environment config JSON file (e.g., Example4_processed.json)')
    parser.add_argument('--layout_path', type=str, required=True,
                        help='Path to the saved agent layout JSON file (e.g., best_gear_layout.json)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Optional: Path to save the plot (e.g., "results/best_layout.png")')
    
    args = parser.parse_args()

    # --- Load Data ---
    try:
        with open(args.config_path, 'r') as f:
            config_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.config_path}")
        sys.exit(1)
        
    try:
        with open(args.layout_path, 'r') as f:
            layout_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Layout file not found at {args.layout_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.layout_path}")
        sys.exit(1)

    # --- Plot ---
    plot_gear_layout(config_data, layout_data, args.save_path)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    main()