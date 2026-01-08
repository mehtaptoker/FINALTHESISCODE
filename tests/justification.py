#####This code is made with help of G00GLE GEMINI############
# compare_solution.py
#
# This script compares the AI-generated gear train solution against a
# predefined "ground truth" real-world gearbox shape to validate its performance.

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from matplotlib import patches
import sys
sys.path.append("../")
# --- CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EXAMPLE_NAME = "Example3"
STRATEGY = 'max_gears' # Make sure this matches the file you want to test

# Path to the AI-generated solution
AI_SOLUTION_PATH = os.path.join(BASE_DIR, "output", EXAMPLE_NAME, f"strategic_gear_layout_{STRATEGY}.json")
print("DEBUG — Loading from:", AI_SOLUTION_PATH)
print("File exists?", os.path.exists(AI_SOLUTION_PATH))


# Output path for the comparison visual
OUTPUT_IMAGE_PATH = os.path.join(BASE_DIR, "output", EXAMPLE_NAME, "comparison_visual.png")

class Gear:
    """Simple data class for a gear."""
    def __init__(self, id, center, driven_r, driving_r=None):
        self.id = id
        self.center = np.array(center)
        self.driven_radius = driven_r
        self.driving_radius = driving_r if driving_r is not None else driven_r

def define_real_gearbox():
    """
    Defines the 'ground truth' gearbox based on the provided example script.
    Returns a list of Gear objects.
    """
    print("Defining the ground truth 'real' gearbox layout...")
    # Radii from the example code
    r1, r2, r3, r4, r5 = 3.0, 2.2, 1.2, 2.5, 2.0
    r4_compound, r5_compound = 1.9, 1.1

    # Calculate positions based on tangency
    x1, y1 = 0, 0
    x2, y2 = -(r1 + r2), 0
    x3, y3 = x2, -(r2 + r3)
    hoek_g4 = np.deg2rad(135)
    dist_g1_g4 = r1 + r4
    x4 = x1 + dist_g1_g4 * np.cos(hoek_g4)
    y4 = y1 + dist_g1_g4 * np.sin(hoek_g4)
    x5, y5 = r1 + r5, 0

    # Create Gear objects
    # We assume G1 is the input gear for this comparison
    real_gears = [
        Gear("real_G1_input", [x1, y1], r1),
        Gear("real_G2", [x2, y2], r2),
        Gear("real_G3", [x3, y3], r3),
        Gear("real_G4", [x4, y4], r4, r4_compound),
        Gear("real_G5", [x5, y5], r5, r5_compound),
    ]
    return real_gears

def load_ai_solution(filepath):
    """Loads the AI-generated gear layout from a JSON file."""
    print(f"Loading AI solution from {filepath}...")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        ai_gears = [
            Gear(
                g['id'],
                [g['center']['x'], g['center']['y']],
                g['driven_radius'],
                g['driving_radius']
            ) for g in data
        ]
        return ai_gears
    except FileNotFoundError:
        print(f"ERROR: AI solution file not found at {filepath}")
        return None

def normalize_gears(gear_list):
    """Translates a list of gears so that the first gear is at the origin (0,0)."""
    if not gear_list:
        return []
    
    offset = gear_list[0].center
    for gear in gear_list:
        gear.center -= offset
    return gear_list

def calculate_total_ratio(gear_list):
    """Calculates the total gear ratio of a gear train."""
    if len(gear_list) < 2:
        return 1.0
    
    total_ratio = 1.0
    for i in range(len(gear_list) - 1):
        # Skip gears with zero driving radius to avoid division by zero
        if gear_list[i].driving_radius == 0.0:
            continue
        # Ratio is driven gear (i+1) divided by driving gear (i)
        stage_ratio = gear_list[i+1].driven_radius / gear_list[i].driving_radius
        total_ratio *= stage_ratio
    return total_ratio

def compare_solutions(real_gears, ai_gears):
    """Calculates and prints similarity metrics between two gear sets."""
    print("\n--- Similarity Report ---")

    # 1. Compare Gear Ratios
    real_ratio = calculate_total_ratio(real_gears)
    ai_ratio = calculate_total_ratio(ai_gears)
    ratio_diff = (abs(ai_ratio - real_ratio) / real_ratio) * 100
    print(f"Gear Ratio Comparison:")
    print(f"  - Real Gearbox Ratio: {real_ratio:.3f}")
    print(f"  - AI Generated Ratio: {ai_ratio:.3f}")
    print(f"  - Difference: {ratio_diff:.2f}%")

    # Normalize for fair comparison of size and position
    real_gears_norm = normalize_gears(real_gears[:]) # Use copies
    ai_gears_norm = normalize_gears(ai_gears[:])

    # 2. Compare Gear Sizes (Mean Absolute Error of radii)
    # We compare the sorted lists of radii to ignore the order
    real_radii = sorted([g.driven_radius for g in real_gears_norm])
    ai_radii = sorted([g.driven_radius for g in ai_gears_norm])
    
    # Pad the shorter list with NaNs for fair comparison
    max_len = max(len(real_radii), len(ai_radii))
    real_radii.extend([np.nan] * (max_len - len(real_radii)))
    ai_radii.extend([np.nan] * (max_len - len(ai_radii)))
    
    size_mae = np.nanmean(np.abs(np.array(real_radii) - np.array(ai_radii)))
    print(f"\nSize Similarity (Mean Absolute Error on Radii):")
    print(f"  - MAE: {size_mae:.3f} mm (Lower is better)")

    # 3. Compare Gear Positions (Mean Absolute Error of centers)
    real_centers = np.array([g.center for g in real_gears_norm])
    ai_centers = np.array([g.center for g in ai_gears_norm])
    
    # Cannot directly compare positions if number of gears is different
    if len(real_centers) == len(ai_centers):
        pos_mae = np.mean(np.abs(real_centers - ai_centers))
        print(f"\nPosition Similarity (Mean Absolute Error on Centers):")
        print(f"  - MAE: {pos_mae:.3f} mm (Lower is better)")
    else:
        print(f"\nPosition Similarity:")
        print(f"  - Cannot compare positions directly (Real: {len(real_gears)} gears, AI: {len(ai_gears)} gears)")

def visualize_comparison(real_gears, ai_gears, output_path):
    """Generates an overlay image of the two solutions."""
    print("\nGenerating visual comparison...")
    fig, ax = plt.subplots(figsize=(12, 12))

    # Normalize for visualization
    real_gears_norm = normalize_gears(real_gears[:])
    ai_gears_norm = normalize_gears(ai_gears[:])

    # Plot Real Gearbox (in gray)
    for gear in real_gears_norm:
        ax.add_artist(plt.Circle(gear.center, gear.driven_radius, facecolor='gray', edgecolor='black', alpha=0.5, zorder=5))
        if gear.driven_radius != gear.driving_radius:
            ax.add_artist(plt.Circle(gear.center, gear.driving_radius, facecolor='darkgray', edgecolor='black', alpha=0.6, zorder=6))

    # Plot AI Gearbox (in blue/skyblue)
    for gear in ai_gears_norm:
        ax.add_artist(plt.Circle(gear.center, gear.driven_radius, facecolor='skyblue', edgecolor='blue', alpha=0.7, zorder=10))
        if gear.driven_radius != gear.driving_radius:
            ax.add_artist(plt.Circle(gear.center, gear.driving_radius, facecolor='royalblue', edgecolor='blue', alpha=0.8, zorder=11))
        ax.text(gear.center[0], gear.center[1], gear.id, ha='center', va='center', fontsize=8, zorder=12)

    ax.set_title('Comparison: Real Gearbox (Gray) vs. AI-Generated (Blue)', fontsize=16)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    
    # Create a custom legend
    legend_elements = [
        patches.Patch(facecolor='gray', edgecolor='black', alpha=0.6, label='Real Gearbox'),
        patches.Patch(facecolor='skyblue', edgecolor='blue', alpha=0.7, label='AI-Generated Gearbox')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Comparison visual saved to: {output_path}")

# --- Main Execution ---
if __name__ == "__main__":
    real_solution = define_real_gearbox()
    ai_solution = load_ai_solution(AI_SOLUTION_PATH)

    if ai_solution:
        compare_solutions(real_solution, ai_solution)
        visualize_comparison(real_solution, ai_solution, OUTPUT_IMAGE_PATH)
