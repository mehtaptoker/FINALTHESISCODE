#####This code is made with help of G00GLE GEMINI############
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
file_path = 'models/thesis_success_log.csv' 
output_folder = 'thesis_plots'

# Create output folder
os.makedirs(output_folder, exist_ok=True)

try:
    # 1. Load Data
    df = pd.read_csv(file_path)
    print(f"Data loaded! Found {len(df)} successful designs.")

    # Set style
    sns.set_theme(style="whitegrid")
    
    # --- GRAPH 1: Total Reward Evolution (The Learning Curve) ---
    # This is the graph you asked for. It shows if the agent's score improves.
    plt.figure(figsize=(10, 6))
    # Scatter plot for individual episodes (dots)
    sns.scatterplot(data=df, x='Episode', y='Total_Reward', alpha=0.3, color='gray', label='Individual Episode')
    # Line plot for the trend (smooth line)
    sns.lineplot(data=df, x='Episode', y='Total_Reward', color='blue', label='Average Trend')
    plt.title("Evolution of Reward Score Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward Score")
    plt.legend()
    plt.savefig(f"{output_folder}/1_reward_evolution.png")
    plt.close()

    # --- GRAPH 2: Torque Ratio Error Convergence ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Episode', y='Ratio_Error', alpha=0.5, color='orange', label='Run Error')
    sns.lineplot(data=df, x='Episode', y='Ratio_Error', color='red', label='Trend')
    plt.title("Convergence of Torque Ratio Error (Lower is Better)")
    plt.xlabel("Episode")
    plt.ylabel("Absolute Deviation from Target")
    plt.legend()
    plt.savefig(f"{output_folder}/2_ratio_convergence.png")
    plt.close()

    # --- GRAPH 3: Safety Clearance Distribution ---
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Safety_Clearance', kde=True, color='green')
    plt.title("Distribution of Safety Clearance")
    plt.xlabel("Minimum Distance to Boundary (mm)")
    plt.ylabel("Count of Designs")
    plt.axvline(x=0, color='red', linestyle='--', label='Collision Limit')
    plt.legend()
    plt.savefig(f"{output_folder}/3_safety_clearance_dist.png")
    plt.close()

    # --- GRAPH 4: Mass/Space Ratio vs Reward ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Mass_Space_Ratio', y='Total_Reward', hue='Episode', palette='viridis')
    plt.title("Impact of Compactness (Mass/Space) on Reward")
    plt.xlabel("Mass / Space Ratio")
    plt.ylabel("Total Reward")
    plt.savefig(f"{output_folder}/4_mass_space_correlation.png")
    plt.close()

    print(f"✅ All 4 graphs saved in '{output_folder}':")
    print("   1. 1_reward_evolution.png (Your requested Reward Graph)")
    print("   2. 2_ratio_convergence.png")
    print("   3. 3_safety_clearance_dist.png")
    print("   4. 4_mass_space_correlation.png")

except FileNotFoundError:
    print(f"❌ Error: Could not find file at '{file_path}'")
except Exception as e:
    print(f"❌ An error occurred: {e}")