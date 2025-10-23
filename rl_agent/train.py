import time
import os
import json
import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt # <-- ADDED: Import matplotlib

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_env.env import GearEnv
from rl_agent.agents.ppo_agent import PPOAgent 

def plot_learning_curve(episode_rewards): # <-- ADDED: Plotting function
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
    plt.savefig(file_name) # <-- CHANGED: Replaced plt.show() with plt.savefig()
    print(f"\nLearning curve plot saved as '{file_name}'")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train RL agent for gear train generation')
    # --- Arguments ---
    parser.add_argument('--config_path', type=str, required=True, help='Path to the environment config JSON file.')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--max_steps_per_episode', type=int, default=10, help='Maximum steps per episode')
    parser.add_argument('--update_timestep', type=int, default=2048, help='Number of steps to collect before updating the policy')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for optimizer')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--log_interval', type=int, default=1, help='Log progress every N episodes')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save models')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Environment and Agent Setup ---
    with open(args.config_path, 'r') as f:
        env_config = json.load(f)
    env_config['json_path'] = args.config_path
    env = GearEnv(env_config)

    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec
    agent = PPOAgent(state_dim=state_dim, action_dims=action_dims, lr=args.learning_rate, gamma=args.gamma, clip_epsilon=args.clip_epsilon)

    print("--- Starting Agent Training ---")
    start_time = time.time()
    
    time_step_counter = 0
    all_episode_rewards = []
    # ---added
    best_episode_reward = -float('inf') # Initialize with a very low number
    best_model_path = os.path.join(args.output_dir, "ppo_gear_placer_best.pt")
    #ended

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
                # This print can be verbose, so let's comment it out for the test
                # print(f"Episode {episode} | Timestep {time_step_counter} | Policy Updated | Loss: {loss:.4f}")

            state = next_state
            episode_reward += reward
            if done:
                break
        
        all_episode_rewards.append(episode_reward)
        # <--- ADDED: Check for new best model ---
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            print(f"🎉 New best reward! {best_episode_reward:.2f} at episode {episode}. Saving best model. 🎉")
            agent.save(best_model_path)
        # <--- END OF ADDED SECTION ---
        if episode % args.log_interval == 0:
            avg_reward = np.mean(all_episode_rewards[-args.log_interval:])
            print(f"Episode {episode} | Last Reward: {episode_reward:.2f} | Avg Reward (last {args.log_interval}): {avg_reward:.2f}")

    end_time = time.time()
    print(f"--- Training Finished in {end_time - start_time:.2f}s ---")
    
    # --- DEBUGGING PRINTS ADDED BELOW ---
    
    print("\nDEBUG: Preparing to save model...")
    model_path = os.path.join(args.output_dir, "ppo_gear_placer_final.pt")
    agent.save(model_path)
    print(f"DEBUG: Model saving process finished. Model should be at {model_path}")
    # <--- MODIFIED: Ensure "best model" message is clear ---
    print(f"Best model saved to {best_model_path} (Reward: {best_episode_reward:.2f})")
    
    env.close()
    
    
 
    # --- Plotting ---
    plot_learning_curve(all_episode_rewards)


if __name__ == "__main__":
    main()