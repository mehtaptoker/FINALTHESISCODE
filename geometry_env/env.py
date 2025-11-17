# ADD THIS LINE AT THE VERY TOP OF geometry_env/env.py
import os; print(f"--- LOADING ENV.PY FROM THIS EXACT FILE: {os.path.abspath(__file__)} ---")
print("--- LOADING THE NEW, ADAPTED ENV.PY (VERSION 123) ---")
# --- End of test line ---
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
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
        """
        super().__init__()
        self.config = config
        
        # --- 1. Pad configuratie ---
        self.processed_json_path = config["json_path"]
        self.constraints_json_path = config.get("constraints_path")
        if not self.constraints_json_path:
             raise ValueError("Config mist 'constraints_path'. Geef dit argument mee aan train.py.")

        # --- 2. CONSTRAINTS LADEN (MET CORRECTE TRY/EXCEPT) ---
        try:
            with open(self.constraints_json_path, 'r') as f:
                constraints_data = json.load(f)
            
            # Lees alle parameters
            self.min_teeth = int(constraints_data.get("min_gear_size", 8))
            self.max_teeth = int(constraints_data.get("max_gear_size", 40))
            self.max_intermediate_gears = int(constraints_data.get("num_intermediate_gears", 10))
            self.boundary_margin = float(constraints_data.get("boundary_margin", 1.0))

            # Converteer de "1:2" string naar een float 2.0
            ratio_str = constraints_data.get("torque_ratio", "1:1")
            try:
                t_in, t_out = map(float, ratio_str.split(':'))
                self.target_torque_ratio = t_out / t_in if t_in != 0 else 1.0
            except Exception:
                self.target_torque_ratio = 1.0
            print(f"INFO: Doel koppelverhouding ingesteld op: {self.target_torque_ratio:.2f}")

        # --- DIT IS DE ONTBREKENDE 'EXCEPT' VOOR DE BUITENSTE 'TRY' ---
        except Exception as e:
            # Vang fouten op zoals FileNotFoundError
            raise ValueError(f"Fout bij lezen van constraints in {self.constraints_json_path}: {e}") from e
        # --- EINDE FIX ---

        # --- 3. Padvinder uitvoeren ---
        pathfinder = Pathfinder()
        #self.optimal_path = pathfinder.find_path(self.processed_json_path, margin=self.boundary_margin)
        self.optimal_path = pathfinder.find_centerline_path(self.processed_json_path, margin=self.boundary_margin)
        
        if not self.optimal_path:
            print(f"WAARSCHUWING: Kon geen pad vinden met marge {self.boundary_margin}. Probeert opnieuw met minimale marge (0.1)...")
            self.optimal_path = pathfinder.find_path(self.processed_json_path, margin=0.1)
            if not self.optimal_path:
                raise RuntimeError(f"Pathfinder failed to find a path for {self.processed_json_path}")
                
        # --- 4. Definieer Actie- en Observatieruimte ---
        num_choices = self.max_teeth - self.min_teeth + 1
        self.action_space = spaces.MultiDiscrete([num_choices, num_choices])
        
        # Observatieruimte (inclusief 'consecutive_failures')
        # [last_x, last_y, last_teeth, last_radius, dist_to_target, consecutive_failures]
        low_bounds = np.array([-500, -500, self.min_teeth, 0, 0, 0, 0], dtype=np.float32)
        high_bounds = np.array([500, 500, self.max_teeth, 500, 1000, 10, 500], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        # --- 5. Initialiseer de Simulatie Engine ---
        gear_factory = GearFactory(module=config.get("module", 1.0))
        
        with open(self.processed_json_path, 'r') as f:
            data = json.load(f)['normalized_space']

        self.simulator = GearTrainSimulator(
            path=self.optimal_path,
            input_shaft=tuple(data['input_shaft'].values()),
            output_shaft=tuple(data['output_shaft'].values()),
            boundaries=data['boundaries'],
            gear_factory=gear_factory,
            clearance_margin=self.boundary_margin,
            target_torque_ratio=self.target_torque_ratio, # Geef ratio door
            # obstacles=... (Deze wordt correct afgehandeld in de simulator)
            min_teeth=self.min_teeth
        )
        
        # Interne staat voor de omgeving
        self.consecutive_failures = 0
        self.intermediate_gears_placed = 0
        self._current_step = 0

    def _parse_torque_ratio(self, ratio_str: str):
        try:
            t_in, t_out = map(float, ratio_str.split(':'))
            if t_in == 0: return 1.0, 0.1
            return t_out / t_in, 0.1 # Standaard tolerantie
        except Exception:
            return 1.0, 0.1 # Default naar 1:1

    def _state_to_observation(self, state: dict, consecutive_failures: int) -> np.ndarray:
        """Converteert de simulator's state dictionary naar een NumPy array."""
        clearance = self.simulator.get_available_clearance() if hasattr(self.simulator, 'get_available_clearance') else 0.0

        if state is None:
            # Maak een dummy state als de simulator faalt bij reset
            return np.array([
                getattr(self.simulator.input_shaft, 'x', 0),
                getattr(self.simulator.input_shaft, 'y', 0),
                0, 0, 
                getattr(self.simulator, '_distance', lambda p1, p2: 100)(
                    self.simulator.input_shaft, self.simulator.output_shaft
                ),
                consecutive_failures,
                0.0
            ], dtype=np.float32)
            
        return np.array([
            state["last_gear_center_x"],
            state["last_gear_center_y"],
            state["last_gear_teeth"],
            state["last_gear_radius"],
            state["distance_to_target"],
            consecutive_failures,
            clearance
        ], dtype=np.float32)
        
    # VERVANG DE HUIDIGE FUNCTIE DOOR DEZE:
    # VERVANG DE HUIDIGE FUNCTIE DOOR DEZE:
    def get_current_layout_data(self):
        """
        Haalt de gestructureerde data op voor de huidige tandwielplaatsing
        uit de simulator.
        """
        if not hasattr(self.simulator, 'gears'):
             raise ValueError("Simulator object heeft geen 'gears' attribuut.")
             
        try:
            # Haal tandwiel data op
            layout_data = [gear.to_json() for gear in self.simulator.gears]
            
            # --- *** DIT IS DE CORRECTIE *** ---
            # Converteer de Point-objecten handmatig naar dictionaries
            # in plaats van .to_dict() aan te roepen.
            
            boundaries_list = [{"x": p.x, "y": p.y} for p in self.simulator.boundaries]
            input_shaft_dict = {"x": self.simulator.input_shaft.x, "y": self.simulator.input_shaft.y}
            output_shaft_dict = {"x": self.simulator.output_shaft.x, "y": self.simulator.output_shaft.y}

            return {
                "gears": layout_data,
                "path": self.optimal_path,
                "boundaries": boundaries_list,     
                "input_shaft": input_shaft_dict,   
                "output_shaft": output_shaft_dict  
            }
           
            
        except Exception as e:
            print(f"FOUT bij ophalen layout data: {e}")
            return {"error": str(e), "gears": []}
        
    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        
        # Reset interne counters
        self.consecutive_failures = 0
        self.intermediate_gears_placed = 0
        self._current_step = 0
        
        initial_teeth = self.config.get("initial_gear_teeth", 20)
        
        # Probeer de simulator te resetten
        try:
            state, _, terminated, info = self.simulator.reset(initial_gear_teeth=initial_teeth)
            if state is None: 
                raise RuntimeError("Simulator.reset() gaf None state terug.")
            if terminated:
                 print(f"WAARSCHUWING: Simulator rapporteerde 'terminated=True' direct na reset. Info: {info}")
                 # Dit is een fatale fout, de state is waarschijnlijk een error state
                 info["error"] = info.get("error", "Onbekende reset fout")
                 
        except Exception as e:
            print(f"FATALE FOUT tijdens simulator.reset(): {e}")
            state = None # Forceer gebruik van _get_state_on_error
            info = {"error": f"Simulator reset mislukt: {e}"}

        observation = self._state_to_observation(state, self.consecutive_failures)
        return observation, info

    def step(self, action: np.ndarray):
        """Voert een stap uit (CORRECTE, SIMPELE VERSIE)."""
        self._current_step += 1
        info = {}

        # --- 1. Actie Verwerken ---
        try:
            driven_teeth = self.min_teeth + action[0]
            driving_teeth = self.min_teeth + action[1]
            action_tuple = (driven_teeth, driving_teeth)
        except (TypeError, IndexError) as e:
            observation = self._state_to_observation(None, self.consecutive_failures)
            return observation, -50.0, True, False, {"error": f"Ongeldige actie-structuur: {e}"}

        # --- 2. Simulator Aanroepen ---
        try:
            new_state, sim_reward, terminated, sim_info = self.simulator.step(action_tuple)
            info.update(sim_info if sim_info is not None else {})
            reward = float(sim_reward) # Zorg ervoor dat het een float is
            
        except Exception as e:
            print(f"FATALE FOUT tijdens simulator.step(): {e}")
            new_state = None # Forceer error state
            reward = -500.0 # Zeer zware straf voor runtime error
            terminated = True
            info["error"] = f"Simulator runtime error: {e}"

        # --- 3. Staat en Counters Bijwerken ---
        # Controleer OF DEZE STAP een fout was
        is_placement_error = terminated and "error" in info and any(
            keyword in info["error"].lower() for keyword in ["boundary", "collision", "placement", "grens", "fout"])

        if is_placement_error:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
            if not terminated: # Verhoog alleen als het een succesvolle *tussenstap* was
                 self.intermediate_gears_placed += 1
        
        observation = self._state_to_observation(new_state, self.consecutive_failures)
        
        # --- 4. Finale Controle op Episode Einde ---
        truncated = False
        
        # Regel A: "Three Strikes, You're Out"
        if self.consecutive_failures >= 3 and not terminated:
            terminated = True
            reward = -50.0  # Zware straf voor vastlopen
            info["error"] = info.get("error", "") + "; Agent is vastgelopen (3+ fouten)."

        # Regel B: Limiet op aantal tandwielen
        if self.intermediate_gears_placed >= self.max_intermediate_gears and not terminated:
            terminated = True
            reward = -20.0  # Straf voor inefficiëntie
            info["error"] = info.get("error", "") + f"; Limiet {self.max_intermediate_gears} tandwielen overschreden."
        
        # Regel C: Tijdslimiet (Truncation)
        max_env_steps = self.config.get("max_steps_per_episode", 20)
        if self._current_step >= max_env_steps and not terminated:
            truncated = True
            reward -= 5.0  # Kleine straf voor tijd op
            info["TimeLimit.truncated"] = True
            terminated = True # In PPO wordt truncation vaak als termination behandeld

        return observation, reward, terminated, truncated, info

    def close(self):
        """Performs any necessary cleanup."""
        pass