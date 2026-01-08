# test_known_solution.py
#####This code is made with help of G00GLE GEMINI############
#
# Dit script valideert de RL-methode door te testen of de agent
# een specifieke, vooraf gedefinieerde en correcte tandwiel-oplossing kan leren.

# Stap 1: Installeer de benodigde bibliotheken in een Colab-cel
# Verwijder de '#' hieronder en voer deze cel eenmalig uit
# !pip install stable-baselines3[extra] torch gymnasium matplotlib shapely

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from IPython.display import display, clear_output

# --- CONFIGURATIE VAN DE BEKENDE OPLOSSING ---
TARGET_GEAR_RATIO = 2
DESIRED_INTERMEDIATE_GEARS = 4

class Gear:
    """Dataklasse voor een tandwiel."""
    def __init__(self, id, center, driven_r, driving_r):
        self.id = id
        self.center = np.array(center)
        self.driven_radius = driven_r
        self.driving_radius = driving_r

class KnownSolutionEnv(gym.Env):
    """Een RL-omgeving die is ontworpen om te testen of de agent een specifieke,
       bekende oplossing kan leren."""
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(KnownSolutionEnv, self).__init__()
        
        # Omgeving is een simpele rechthoek
        rect = [0, 0, 120, 50]
        self.boundaries = [[rect[0], rect[1]], [rect[0]+rect[2], rect[1]], [rect[0]+rect[2], rect[1]+rect[3]], [rect[0], rect[1]+rect[3]]]
        self.boundary_polygon = Polygon(self.boundaries)
        self.input_shaft_pos = np.array([15.0, 25.0])
        self.output_shaft_pos = np.array([105.0, 25.0])
        
        # --- ACTIES: De correcte keuzes + een paar "foute" keuzes ---
        self.possible_actions = [
            (15.0, 10.0),  # Correcte Actie 0
            (12.5, 10.0),  # Correcte Actie 1
            (10.0, 10.0),  # Foute Actie 2
            (8.0, 8.0),    # Foute Actie 3
        ]
        self.action_space = spaces.Discrete(len(self.possible_actions))

        # Observaties
        self.observation_space = spaces.Box(
            low=np.array([-120, -50, 0, 0]), 
            high=np.array([120, 50, 10, 10]), 
            dtype=np.float32
        )
        
        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        # Houd de correcte sequentie bij om de agent te valideren
        self.correct_sequence = [0, 1] 
        self.action_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gears = []
        self.total_ratio = 1.0
        self.action_history = []
        
        input_gear = Gear("input", self.input_shaft_pos, 10.0, 10.0)
        self.gears.append(input_gear)
        self.last_gear = input_gear
        
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        vector_to_target = self.output_shaft_pos - self.last_gear.center
        return np.array([vector_to_target[0], vector_to_target[1], self.total_ratio, len(self.gears)], dtype=np.float32)

    def _get_info(self):
        return {"gears_placed": len(self.gears), "current_ratio": self.total_ratio}

    def step(self, action_index):
        # Sla de actie op als een integer
        self.action_history.append(int(action_index))
        
        if len(self.gears) > DESIRED_INTERMEDIATE_GEARS + 1:
            return self._get_obs(), -500, True, False, self._get_info()

        driven_radius, driving_radius = self.possible_actions[action_index]
        
        direction = self.output_shaft_pos - self.last_gear.center
        direction /= np.linalg.norm(direction)
        
        meshing_distance = self.last_gear.driving_radius + driven_radius
        next_center = self.last_gear.center + direction * meshing_distance
        
        new_gear_circle = Point(next_center).buffer(driven_radius)
        if not self.boundary_polygon.contains(new_gear_circle):
            return self._get_obs(), -200, True, False, self._get_info()

        new_gear = Gear(f"gear_{len(self.gears)}", next_center, driven_radius, driving_radius)
        self.gears.append(new_gear)
        self.total_ratio *= new_gear.driven_radius / self.last_gear.driving_radius
        self.last_gear = new_gear
        
        # *** NIEUW: REWARD SHAPING ***
        # Geef een hint als de EERSTE stap correct is
        reward = -1 # Standaard straf voor een stap
        if len(self.action_history) == 1:
            if self.action_history[0] == self.correct_sequence[0]:
                reward += 100 # Bonus voor de juiste eerste zet
            else:
                reward -= 100 # Straf voor de foute eerste zet

        # --- FINALE REWARD BEREKENING ---
        dist_to_output = np.linalg.norm(self.last_gear.center - self.output_shaft_pos)
        final_gear_radius = dist_to_output - self.last_gear.driving_radius
        final_gear_circle = Point(self.output_shaft_pos).buffer(final_gear_radius)
        
        if len(self.gears) - 1 == DESIRED_INTERMEDIATE_GEARS and self.boundary_polygon.contains(final_gear_circle):
            if self.action_history == self.correct_sequence:
                reward += 5000 # Enorme bonus voor de perfecte oplossing
            else:
                reward -= 1000 # Grote straf voor een foute oplossing
            
            final_gear = Gear("output", self.output_shaft_pos, final_gear_radius, final_gear_radius)
            self.gears.append(final_gear)
            return self._get_obs(), reward, True, False, self._get_info()
        
        return self._get_obs(), reward, False, False, self._get_info()

    def render(self):
        self.ax.clear()
        self.ax.plot(*zip(*self.boundaries, self.boundaries[0]), 'k-', linewidth=1)
        for gear in self.gears:
            self.ax.add_artist(plt.Circle(gear.center, gear.driven_radius, fc='skyblue', ec='blue', alpha=0.6))
            if gear.driven_radius != gear.driving_radius:
                self.ax.add_artist(plt.Circle(gear.center, gear.driving_radius, fc='royalblue', ec='blue'))
        self.ax.add_artist(plt.Circle(self.output_shaft_pos, 2, color='red'))
        self.ax.set_aspect('equal'); self.ax.grid(True)
        clear_output(wait=True); display(self.fig)

    def close(self):
        plt.close(self.fig)

if __name__ == "__main__":
    env = KnownSolutionEnv()
    check_env(env)

    # *** NIEUW: MEER EXPLORATIE ***
    # ent_coef moedigt de agent aan om meer te proberen
    model = PPO('MlpPolicy', env, verbose=1, ent_coef=0.01)
    
    # *** MEER TRAININGSTIJD ***
    TIMESTEPS = 100000
    print(f"\n--- Starting training for {TIMESTEPS} timesteps ---")
    model.learn(total_timesteps=TIMESTEPS)
    print("--- Training complete ---")

    print("\n--- Evaluating trained agent to see if it learned the correct sequence ---")
    total_successes = 0
    for episode in range(10): # Evalueer 10 keer
        obs, info = env.reset()
        done = False
        print(f"\n--- Episode {episode + 1} ---")
        for i in range(DESIRED_INTERMEDIATE_GEARS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            if done: break
        
        if env.action_history == env.correct_sequence:
            print(f"SUCCESS! Agent chose the correct sequence: {env.action_history}")
            total_successes += 1
        else:
            print(f"FAILURE. Agent chose sequence: {env.action_history}, expected: {env.correct_sequence}")
        
        env.render()
    
    print(f"\nValidation complete. Agent found the correct solution in {total_successes}/10 episodes.")
    env.close()