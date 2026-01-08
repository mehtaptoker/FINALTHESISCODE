#JUSTIFICATION##
#####This code is made with help of G00GLE GEMINI############
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
# We definiëren een correcte oplossing die de agent moet leren.
# Doel: een totale ratio van ~1.88 met 2 tussentandwielen.
# Input Gear: r=10
# Gear 1 (Correcte Actie 0): driven=15, driving=10 -> Ratio = 15/10 = 1.5
# Gear 2 (Correcte Actie 1): driven=12.5, driving=10 -> Ratio = 12.5/10 = 1.25
# Finale Ratio = 1.5 * 1.25 = 1.875
TARGET_GEAR_RATIO = 1.875
DESIRED_INTERMEDIATE_GEARS = 2

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
        self.action_history.append(action_index)
        
        # Als de agent te veel stappen zet, is het een mislukking
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
        
        # --- FINALE REWARD BEREKENING ---
        dist_to_output = np.linalg.norm(self.last_gear.center - self.output_shaft_pos)
        final_gear_radius = dist_to_output - self.last_gear.driving_radius
        final_gear_circle = Point(self.output_shaft_pos).buffer(final_gear_radius)
        
        # Controleer of we de output kunnen bereiken
        if len(self.gears) - 1 == DESIRED_INTERMEDIATE_GEARS and self.boundary_polygon.contains(final_gear_circle):
            # We hebben het juiste aantal tandwielen. Nu controleren we de sequentie.
            if self.action_history == self.correct_sequence:
                # De agent heeft de perfecte oplossing gevonden!
                reward = 5000 
            else:
                # De agent heeft een oplossing gevonden, maar niet de juiste.
                reward = -1000
            
            final_gear = Gear("output", self.output_shaft_pos, final_gear_radius, final_gear_radius)
            self.gears.append(final_gear)
            return self._get_obs(), reward, True, False, self._get_info()
        
        return self._get_obs(), -1, False, False, self._get_info()

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

    model = PPO('MlpPolicy', env, verbose=1)
    
    TIMESTEPS = 50000
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
        
        # Controleer of de agent de juiste acties heeft gekozen
        if env.action_history == env.correct_sequence:
            print(f"SUCCESS! Agent chose the correct sequence: {env.action_history}")
            total_successes += 1
        else:
            print(f"FAILURE. Agent chose sequence: {env.action_history}, expected: {env.correct_sequence}")
        
        env.render() # Toon het eindresultaat
    
    print(f"\nValidation complete. Agent found the correct solution in {total_successes}/10 episodes.")
    env.close()
