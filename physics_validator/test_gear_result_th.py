import json
import sys
import numpy as np
import os
import torch

# ---------------------------------------------------------------------
# Global CPU-only mode to avoid CUDA surprises in evaluation
# ---------------------------------------------------------------------
torch.cuda.is_available = lambda: False
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Make project root importable (tests/ -> project root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Project imports
from preprocessing.processor import Processor
from gear_generator.factory import GearFactory
from geometry_env.env import GearEnv
from rl_agent.agents.ppo_agent import PPOAgent
from common.data_models import Gear, Point
from visualization.renderer import Renderer
from pathfinding.finder import Pathfinder


# ---------------------------------------------------------------------
# 1) Monkey-patch Point so code that uses p[0]/p[1] also works.
#    We do NOT modify simulator.py; we adapt here as requested.
# ---------------------------------------------------------------------
try:
    if not hasattr(Point, "__getitem__"):
        def __getitem__(self, idx):
            if idx == 0:
                return float(self.x)
            if idx == 1:
                return float(self.y)
            raise IndexError("Point only supports indices 0 (x) and 1 (y)")
        Point.__getitem__ = __getitem__  # type: ignore[attr-defined]
except Exception as e:
    print("Warning: could not patch Point for subscriptability:", e)


# ---------------------------------------------------------------------
# 2) Utilities: JSON-safe dumping and geometry normalization
# ---------------------------------------------------------------------
def _json_np_default(o):
    """Convert NumPy / torch types to native Python for json.dump."""
    import numpy as _np
    if isinstance(o, _np.integer):
        return int(o)
    if isinstance(o, _np.floating):
        return float(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        t = o.detach().cpu()
        return t.item() if t.ndim == 0 else t.tolist()
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _to_point_list(path_like):
    """Normalize a generic path into a list[Point]."""
    pts = []
    for p in path_like:
        if isinstance(p, dict) and "x" in p and "y" in p:
            pts.append(Point(x=float(p["x"]), y=float(p["y"])))
        elif isinstance(p, (list, tuple)) and len(p) == 2:
            pts.append(Point(x=float(p[0]), y=float(p[1])))
        elif hasattr(p, "x") and hasattr(p, "y"):
            pts.append(Point(x=float(p.x), y=float(p.y)))
        else:
            continue
    return pts


def _to_point(p):
    """Coerce (x,y) tuple/list or {'x','y'} or Point into a Point."""
    if hasattr(p, "x") and hasattr(p, "y"):
        # Ensure numeric
        return Point(x=float(p.x), y=float(p.y))
    if isinstance(p, dict) and "x" in p and "y" in p:
        return Point(x=float(p["x"]), y=float(p["y"]))
    # assume sequence
    return Point(x=float(p[0]), y=float(p[1]))


def _normalize_sim_geometry(sim):
    """
    Ensure the simulator sees canonical Point objects for everything.
    This keeps .x/.y access working, while our monkey-patch enables [0]/[1].
    """
    try:
        sim.input_shaft = _to_point(sim.input_shaft)
    except Exception:
        pass
    try:
        sim.output_shaft = _to_point(sim.output_shaft)
    except Exception:
        pass
    try:
        sim.boundaries = [_to_point(b) for b in sim.boundaries]
    except Exception:
        pass
    try:
        sim.path = [_to_point(p) for p in sim.path]
    except Exception:
        pass


# ---------------------------------------------------------------------
# 3) Main: RL-driven gear generation
# ---------------------------------------------------------------------
def run_rl_gear_generation(example_name="simple_rect", model_path=None):
    """
    Use trained RL agent to generate gear layout, save JSON, and export a PNG visualization.
    """

    # ---------- Paths & config ----------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG = {
        "INPUT_DIR": os.path.join(BASE_DIR, "data"),
        "INTERMEDIATE_DIR": os.path.join(BASE_DIR, "data", "intermediate"),
        "EXAMPLE_NAME": example_name,
        "OUTPUT_DIR": os.path.join(BASE_DIR, "output"),
    }

    # Default model path
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "models", "ppo_gear_placer_final.pt")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using train.py")
        return None, 0

    # ---------- Load environment and constraints from your files ----------
    print(f"--- Loading data for {CONFIG['EXAMPLE_NAME']} ---")
    processed_json_path = os.path.join(CONFIG["INTERMEDIATE_DIR"], f"{CONFIG['EXAMPLE_NAME']}.json")
    constraints_path = os.path.join(CONFIG["INTERMEDIATE_DIR"], f"{CONFIG['EXAMPLE_NAME']}_constraints.json")

    if not os.path.exists(processed_json_path) or not os.path.exists(constraints_path):
        print(f"Error: Ensure '{processed_json_path}' and '{constraints_path}' exist.")
        return None, 0

    with open(constraints_path, 'r') as f:
        constraints = json.load(f)

    # ---------- Pathfinding ----------
    print("\n--- Pathfinding ---")
    path_json_path = os.path.join(CONFIG["OUTPUT_DIR"], example_name, "path.json")
    os.makedirs(os.path.dirname(path_json_path), exist_ok=True)

    if os.path.exists(path_json_path):
        with open(path_json_path, "r") as f:
            optimal_path = json.load(f)
        print(f"Loaded existing path with {len(optimal_path)} points")
    else:
        finder = Pathfinder()
        optimal_path = finder.find_path(processed_json_path, margin=0.5)
        if not optimal_path:
            raise RuntimeError("Could not find path")
        with open(path_json_path, "w") as f:
            json.dump(optimal_path, f, indent=2, default=_json_np_default)
        print(f"Generated new path with {len(optimal_path)} points")

    # ---------- Environment config ----------
    env_config = {
        "json_path": processed_json_path,
        "path": optimal_path,
        "module": 2.0,
        "clearance_margin": 5,
        "initial_gear_teeth": 20,
        "target_torque": constraints.get("torque_ratio", "free"), # Use torque from your file
        "torque_weight": 0.6,
        "space_weight": 0.3,
        "weight_penalty": 0.1,
    }
    print(f"\nEnvironment configured with a target torque of: {env_config['target_torque']}")


    # ---------- Setup Environment & PPO Agent (CPU) ----------
    print("\n--- Setting up Environment and RL Agent ---")
    env = GearEnv(env_config)

    # Normalize geometry *before* first reset so simulator sees Point everywhere
    _normalize_sim_geometry(env.simulator)

    # First reset to get state shape
    state, _ = env.reset()

    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec
    print(f"State dimension: {state_dim}")
    print(f"Action dimensions: {action_dims}")

    # Force PPOAgent to CPU temporarily by wrapping its __init__
    original_init = PPOAgent.__init__

    def cpu_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.device = torch.device("cpu")
        self.policy = self.policy.to(self.device)
        self.policy_old = self.policy_old.to(self.device)

    PPOAgent.__init__ = cpu_init

    agent = PPOAgent(
        state_dim=state_dim,
        action_dims=action_dims,
        lr=0.0,        # eval only
        gamma=0.0,     # eval only
        clip_epsilon=0 # eval only
    )

    # Restore original init
    PPOAgent.__init__ = original_init

    print(f"Loading trained model from: {model_path}")
    agent.policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.policy_old.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.policy.eval()
    agent.policy_old.eval()

    # ---------- Rollout ----------
    print("\n--- Running RL Agent for Gear Generation ---")
    # Ensure geometry still normalized in case env mutated after first reset
    _normalize_sim_geometry(env.simulator)
    state, _ = env.reset()

    done = False
    episode_reward = 0.0
    step_count = 0
    # Set max_steps to 1 since we expect 1 intermediate gear (total 3 gears = 1 step after input)
    max_steps = constraints.get("num_intermediate_gears", 1)


    action_history = []
    reward_history = []

    while not done and step_count < max_steps:
        try:
            with torch.no_grad():
                action, log_prob = agent.act(state)

            # Example: decode discrete actions to teeth counts (offset by 8)
            teeth_driver = int(action[0]) + 8
            teeth_driven = int(action[1]) + 8

            print(f"\nStep {step_count + 1}:")
            print(f"  Agent selected: driver_teeth={teeth_driver}, driven_teeth={teeth_driven}")

            # Some environments expect the raw action vector; if your env needs
            # decoded teeth, adapt here (e.g., pass via info or env wrapper).
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += float(reward)
            action_history.append(action)
            reward_history.append(float(reward))

            print(f"  Reward: {float(reward):.3f}")
            print(f"  Cumulative reward: {episode_reward:.3f}")

            if info.get("error"):
                print(f"  Error: {info['error']}")
            elif info.get("success"):
                print(f"  Success: {info['success']}")

            state = next_state
            step_count += 1

        except Exception as e:
            # If anything in step() touched Point by [] or .x/.y, our patch + normalization covers it.
            # If something else explodes, log and stop the episode gracefully.
            print(f"Error during step {step_count + 1}: {e}")
            break

    print(f"\n--- RL Generation Complete ---")
    print(f"Episode finished after {step_count} steps")
    print(f"Total reward: {episode_reward:.3f}")

    # ---------- Save JSON ----------
    output_dir = os.path.join(CONFIG["OUTPUT_DIR"], CONFIG["EXAMPLE_NAME"])
    os.makedirs(output_dir, exist_ok=True)

    gear_layout_path = os.path.join(output_dir, "rl_generated_layout.json")
    gears_json_data = [gear.to_json() for gear in env.simulator.gears]
    with open(gear_layout_path, "w") as f:
        json.dump(gears_json_data, f, indent=4, default=_json_np_default)
    print(f"\nGenerated gear layout JSON saved to: {gear_layout_path}")

    # ---------- Render PNG ----------
    png_path = os.path.join(output_dir, "rl_generated_layout.png")
    rendered = False
    try:
        # The line converting the path to Point objects has been removed.
        # We now pass the simple 'optimal_path' list directly.
        Renderer.render_processed_data(
            processed_data_path=processed_json_path,
            output_path=png_path,
            path=optimal_path, # <-- This is the corrected line
            gears=env.simulator.gears,
        )
        rendered = True
    except Exception as e1:
        print(f"render_processed_data failed: {e1}")
        # Fallback if your renderer exposes a simpler API
        try:
            # This fallback likely doesn't exist, but we keep the structure
            Renderer.render_gears(env.simulator.gears, save_path=png_path)
            rendered = True
        except Exception as e2:
            print(f"render_gears fallback failed: {e2}")

    if rendered:
        print(f"Gear layout PNG saved to: {png_path}")
    else:
        print("Failed to render PNG. Please verify the Renderer API/signature.")
    # ---------- Wrap up ----------
    print(f"\n--- Analysis ---")
    print(f"Total gears placed: {len(env.simulator.gears)}")

    env.close()
    return episode_reward, len(env.simulator.gears)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Use trained RL agent for gear generation")
    parser.add_argument("--example", type=str, default="simple_rect", help="Example name to process")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model")

    args = parser.parse_args()

    # Force CPU default tensor type
    torch.set_default_tensor_type("torch.FloatTensor")

    run_rl_gear_generation(args.example, args.model_path)