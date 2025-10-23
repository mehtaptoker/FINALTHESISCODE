import os
import json
import argparse
import sys
import numpy as np

# <--- FIX 1: Correctly add project root to path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_env.env import GearEnv
from rl_agent.agents.ppo_agent import PPOAgent
from visualization.renderer import Renderer
from common.data_models import Gear 

# <--- ADDED: Custom JSON Encoder to handle Numpy types ---
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
# <--- END OF ADDED SECTION ---

def evaluate_agent():
    """
    Loads a trained PPO agent and evaluates its performance on the GearEnv
    by running one full episode and visualizing the result.
    """
    parser = argparse.ArgumentParser(description='Evaluate a trained RL agent for gear generation.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained PPO model (.pt file).')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the environment config JSON file.')
    parser.add_argument('--output_dir', type=str, default='output_eval', help='Directory to save evaluation results.')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps to run in evaluation episode.')
    args = parser.parse_args()

    # --- Environment and Agent Setup ---
    print("--- Setting up Environment and Agent ---")
    with open(args.config_path, 'r') as f:
        env_config = json.load(f)

    # <--- FIX 2: THIS IS THE MISSING LINE ---
    env_config['json_path'] = args.config_path
    # <--- END OF FIX ---

    env = GearEnv(env_config)

    # Get state and action dimensions from the environment
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec

    # Instantiate the agent and load the trained weights
    agent = PPOAgent(state_dim, action_dims, lr=0, gamma=0, clip_epsilon=0) # Hyperparams don't matter for eval
    agent.load(args.model_path)
    print(f"Model loaded from {args.model_path}")

    # --- Run Evaluation Episode ---
    print("\n--- Running Evaluation Episode ---")
    state, _ = env.reset()
    done = False
    episode_reward = 0
    step_count = 0
    
    while not done and step_count < args.max_steps:
        # Agent selects the best action deterministically
        try:
            # Use deterministic=True to get the best action
            action, _ = agent.act(state, deterministic=True)
        except TypeError:
            # Fallback if the agent's act method doesn't support deterministic flag
            print("Warning: agent.act() may not support deterministic flag. Using default action.")
            action, _ = agent.act(state)
        
        # Environment executes the action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        state = next_state
        episode_reward += reward
        step_count += 1
        
        print(f"Step {step_count}: Action={action}, Reward={reward:.2f}")

    print(f"\n--- Episode Finished ---")
    if info.get("success"):
        print(f"Result: SUCCESS - {info['success']}")
    elif info.get("error"):
        print(f"Result: FAILED - {info['error']}")
    elif not done and step_count >= args.max_steps:
        print(f"Result: FAILED - Episode timed out at {args.max_steps} steps.")
    else:
        print("Result: Episode finished due to step limit or other reason.")
    print(f"Total Reward: {episode_reward:.2f}")

    # --- Save and Visualize the Result ---
    example_name = os.path.basename(env_config['json_path']).replace('_processed.json', '')
    eval_output_dir = os.path.join(args.output_dir, f"{example_name}_eval")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    gear_layout_path = os.path.join(eval_output_dir, "evaluation_gear_layout.json")
    
    gears_list = []
    if hasattr(env, 'simulator') and hasattr(env.simulator, 'gears'):
        gears_list = env.simulator.gears
    elif hasattr(env, 'gears'): # Fallback if gears are stored on env
        gears_list = env.gears
    else:
        print("Warning: Could not find 'env.simulator.gears' or 'env.gears'. Gear layout JSON may be empty.")

    gears_json_data = [gear.to_json() for gear in gears_list]

    with open(gear_layout_path, 'w') as f:
        # <--- MODIFIED: Use the custom NumpyEncoder ---
        json.dump(gears_json_data, f, indent=4, cls=NumpyEncoder)
        # <--- END OF MODIFICATION ---
    print(f"\nGenerated gear layout saved to: {gear_layout_path}")

    # Render the final gear train from the saved files
    output_image_path = os.path.join(eval_output_dir, "evaluation_result.png")
    
    guidance_path = []
    if hasattr(env, 'simulator') and hasattr(env.simulator, 'path'):
        guidance_path = env.simulator.path
    elif hasattr(env, 'path'): # Fallback if path is stored on env
        guidance_path = env.path
    else:
        print("Warning: Could not Sfind 'env.simulator.path' or 'env.path'. Visualization may not show the path.")

    # <--- MODIFIED: Removed the problematic keyword argument ---
    Renderer.render_processed_data(
        processed_data_path=env_config['json_path'],
        output_path=output_image_path,
        path=guidance_path
        # gear_layout_path=gear_layout_path  <-- This was causing the TypeError
    )
    # <--- END OF MODIFICATION ---
    print(f"Final visualization saved to: {output_image_path}")
    
    env.close()

if __name__ == "__main__":
    evaluate_agent()

