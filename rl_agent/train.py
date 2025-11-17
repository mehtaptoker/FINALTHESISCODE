import time
import os
import json # <- REQUIRED: For saving the gear placement data
import argparse
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_env.env import GearEnv
from rl_agent.agents.ppo_agent import PPOAgent
# --- PLAK DEZE KLASSE HIER ---
class NumpyEncoder(json.JSONEncoder):
    """ Speciale json encoder voor numpy types """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super(NumpyEncoder, self).default(obj)
# --- EINDE VAN DE KLASSE ---

def plot_learning_curve(episode_rewards):
    """
    Plots the learning curve from the collected episode rewards.
    """
    # --- Data for Plotting ---
    # Focus on the final phase of training as requested
    episodes_to_plot = min(len(episode_rewards), 100) # Plot last 100 episodes or all if less
    last_rewards = episode_rewards[-episodes_to_plot:]
    episode_numbers = range(len(episode_rewards) - episodes_to_plot, len(episode_rewards))

    # --- Calculate Moving Average ---
    # A 5-episode moving average helps to see the trend
    window_size = 5
    moving_avg = np.convolve(last_rewards, np.ones(window_size), 'valid') / window_size
    # Adjust x-axis for the moving average
    moving_avg_episodes = episode_numbers[window_size-1:]

    # --- Create Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot raw episode rewards
    ax.plot(episode_numbers, last_rewards, label='Episode Reward', color='lightgray', alpha=0.8)

    # Plot moving average
    ax.plot(moving_avg_episodes, moving_avg, label=f'{window_size}-Episode Moving Average', color='blue', linewidth=2)

    # --- Formatting ---
    ax.set_title('Agent Learning Curve (Final Phase)', fontsize=16)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()

    print("\nDisplaying learning curve plot...")
    # --- Save the plot to a file ---
    file_name = "learning_curve.png"
    plt.savefig(file_name)
    print(f"\nLearning curve plot saved as '{file_name}'")

    plt.show()


def main():
    
    parser = argparse.ArgumentParser(description='Train RL agent for gear train generation')
    # --- Arguments ---
    parser.add_argument('--config_path', type=str, required=True, help='Path to the environment config JSON file.')
    parser.add_argument('--episodes', type=int, default=20000, help='Number of training episodes')
    parser.add_argument('--max_steps_per_episode', type=int, default=25, help='Maximum steps per episode')
    parser.add_argument('--update_timestep', type=int, default=4096, help='Number of steps to collect before updating the policy')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for optimizer')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--log_interval', type=int, default=1, help='Log progress every N episodes')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--constraints_path', type=str, required=True, help='Path to the *original constraints* JSON file (e.g., Example1_constraints.json).')


    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # 1. Laad de geometrie (processed file)
    with open(args.config_path, 'r') as f:
        env_config = json.load(f)
    
    # 2. Laad de constraints
    with open(args.constraints_path, 'r') as f:
        constraints_config = json.load(f)

    # 3. Voeg de constraints samen in de hoofdconfig
    env_config.update(constraints_config)
    
    # 4. Voeg de paden toe die de env nodig heeft
    env_config['json_path'] = args.config_path
    env_config['constraints_path'] = args.constraints_path
    # --- EINDE VAN DE FIX ---

    env = GearEnv(env_config)

    state_dim = env.observation_space.shape[0]

    # --- Environment and Agent Setup ---
    action_dims = env.action_space.nvec
    agent = PPOAgent(state_dim=state_dim, action_dims=action_dims, lr=args.learning_rate, gamma=args.gamma, clip_epsilon=args.clip_epsilon)

    print("--- Starting Agent Training ---")
    start_time = time.time()

    time_step_counter = 0
    all_episode_rewards = []
    
    # --- ADDED: Tracking variables for the best results ---
    #best_episode_reward = -float('inf')
    # Start op 10.0. Hierdoor worden "crashes" (die nu negatief scoren) NOOIT opgeslagen.
    best_episode_reward = 10.0
    best_model_path = os.path.join(args.output_dir, "ppo_gear_placer_best.pt")
    # This is the dedicated path for your visualization JSON:
    best_layout_path = os.path.join(args.output_dir, "best_gear_layout.json") 
    best_layout_data = None
    # -----------------------------------------------------

    # --- Training Loop ---
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(args.max_steps_per_episode):
            time_step_counter += 1

            action, log_prob = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.memory.add(state, action, reward, done, log_prob)

            if time_step_counter % args.update_timestep == 0:
                loss = agent.update()
                # print(f"Episode {episode} | Timestep {time_step_counter} | Policy Updated | Loss: {loss:.4f}")

            state = next_state
            episode_reward += reward
            if done:
                break

        all_episode_rewards.append(episode_reward)

        # <--- MODIFIED/ADDED: Check for new best model and save BOTH model and layout ---
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            print(f"🎉 New best reward! {best_episode_reward:.2f} at episode {episode}. Saving best results to {args.output_dir}. 🎉")

            # 1. Save the best model weights
            agent.save(best_model_path)

            # 2. Get the current gear layout data from the environment
            # NOTE: THIS REQUIRES YOU TO IMPLEMENT env.get_current_layout_data() IN GearEnv!
            try:
                best_layout_data = env.get_current_layout_data()
                
                # 3. Save the layout data to the dedicated JSON file
                with open(best_layout_path, 'w') as f:
                    json.dump(best_layout_data, f, indent=4, cls=NumpyEncoder)
                print(f"✅ Saved best gear layout to {best_layout_path}")
            except AttributeError:
                print("⚠️ WARNING: GearEnv is missing the required method 'get_current_layout_data()'. Gear layout was NOT saved.")
            except Exception as e:
                print(f"⚠️ Error saving gear layout: {e}")
        # <--- END OF MODIFIED/ADDED SECTION ---

        if episode % args.log_interval == 0:
            avg_reward = np.mean(all_episode_rewards[-args.log_interval:])
            print(f"Episode {episode} | Last Reward: {episode_reward:.2f} | Avg Reward (last {args.log_interval}): {avg_reward:.2f}")

    end_time = time.time()
    print(f"--- Training Finished in {end_time - start_time:.2f}s ---")

    # --- Final saving steps ---
    print("\nSaving final model...")
    model_path = os.path.join(args.output_dir, "ppo_gear_placer_final.pt")
    agent.save(model_path)
    print(f"Final model saved to {model_path}")
    print(f"Best model saved to {best_model_path} (Reward: {best_episode_reward:.2f})")

    env.close()

    # --- Plotting ---
    plot_learning_curve(all_episode_rewards)


if __name__ == "__main__":
    main()