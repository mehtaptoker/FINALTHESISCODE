import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import sys
import os
sys.path.append('../')

from geometry_env.simulator import GearTrainSimulator
from gear_generator.factory import GearFactory
from pathfinding.finder import Pathfinder
from common.data_models import Gear, Point

class GearEnv(gym.Env):
    """
    A Gymnasium environment for the gear train generation problem.
    The agent learns to place simple and compound gears to connect two shafts.
    """
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, config: dict):
        """
        Initializes the environment, including pathfinding.

        Args:
            config (dict): A configuration dictionary.
        """
        super().__init__()
        self.config = config
        
        # --- Run Pathfinding to get the optimal path for the simulator ---
        processed_json_path = config["json_path"]
        pathfinder = Pathfinder()
        self.optimal_path = pathfinder.find_path(processed_json_path, margin=0.5)
        
        if not self.optimal_path:
            raise RuntimeError(f"Pathfinder failed to find a path for {processed_json_path}")
            
        # --- Define Action and Observation Spaces ---
        self.min_teeth = config.get("min_gear_teeth", 8)
        self.max_teeth = config.get("max_gear_teeth", 40)
        num_choices = self.max_teeth - self.min_teeth + 1

        # Action Space: [driven_teeth, driving_teeth] for a compound gear.
        # For a simple gear, the agent can learn to select the same value for both.
        self.action_space = spaces.MultiDiscrete([num_choices, num_choices])
        
        # Observation Space: [last_gear_x, last_gear_y, last_gear_teeth, last_gear_radius, dist_to_target]
        low_bounds = np.array([-500, -500, self.min_teeth, 0, 0], dtype=np.float32)
        high_bounds = np.array([500, 500, self.max_teeth, 500, 1000], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        # --- Initialize the Simulation Engine ---
        gear_factory = GearFactory(module=config.get("module", 1.0))
        
        with open(processed_json_path, 'r') as f:
            data = json.load(f)['normalized_space']

        self.simulator = GearTrainSimulator(
            path=self.optimal_path,  # Use the optimal path found earlier
            input_shaft=tuple(data['input_shaft'].values()),
            output_shaft=tuple(data['output_shaft'].values()),
            boundaries=data['boundaries'],
            gear_factory=gear_factory,
            clearance_margin=config.get("clearance_margin", 0.5)
        )

    def _state_to_observation(self, state: dict) -> np.ndarray:
        """Converts the simulator's state dictionary to a NumPy array."""
        if state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        return np.array([
            state["last_gear_center_x"],
            state["last_gear_center_y"],
            state["last_gear_teeth"],
            state["last_gear_radius"],
            state["distance_to_target"]
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        
        initial_teeth = self.config.get("initial_gear_teeth", 20)
        state, _, _, info = self.simulator.reset(initial_gear_teeth=initial_teeth)
        
        observation = self._state_to_observation(state)
        return observation, info

    def step(self, action: np.ndarray):
        """Executes one time step within the environment."""
        # Map the MultiDiscrete action array to the tooth count tuple
        driven_teeth = self.min_teeth + action[0]
        driving_teeth = self.min_teeth + action[1]
        action_tuple = (driven_teeth, driving_teeth)
        
        # Pass the action tuple to the simulator
        state, reward, done, info = self.simulator.step(action_tuple)
        
        observation = self._state_to_observation(state)
        
        terminated = done
        truncated = False # No time limit truncation

        return observation, reward, terminated, truncated, info

    def close(self):
        """Performs any necessary cleanup."""
        pass