import math
import sys
import os
import numpy as np
from typing import List, Tuple, Union, Optional, Dict
# Verwijder sys.path manipulatie als common, physics_validator, etc. correct geïnstalleerd zijn

# Probeer imports relatief te maken als ze binnen hetzelfde package zitten
try:
    from ..common.data_models import Point, GearSet
    from ..gear_generator.factory import GearFactory
except ImportError:
    # Fallback voor als de relatieve imports falen (bv. bij direct uitvoeren)
    from common.data_models import Point, GearSet
    from gear_generator.factory import GearFactory

# Bovenaan in simulator.py
try:
    from ..rl_agent.reward import compute_terminal_reward # Pas pad aan indien nodig
except ImportError:
    # Fallback of dummy functie als bestand niet gevonden wordt
    pass

class GearTrainSimulator:
    def __init__(self, path: Union[List[List[float]], List[Point]],
                 input_shaft: Union[Tuple[float, float], Point, Dict],
                 output_shaft: Union[Tuple[float, float], Point, Dict],
                 boundaries: Union[List[List[float]], List[Point], List[Dict]],
                 gear_factory: GearFactory, clearance_margin: float = 1.0,
                 target_torque_ratio: float = 1.0,
                 min_teeth: int = 8):

        # Normaliseer alle geometrie inputs naar Point objecten
        self.path = self._normalize_path(path)
        self.input_shaft = self._normalize_point(input_shaft)
        self.output_shaft = self._normalize_point(output_shaft)
        self.boundaries = self._normalize_boundaries(boundaries)
        self.min_teeth = min_teeth

        # Factories / params
        self.gear_factory = gear_factory
        self.clearance_margin = max(0.1, clearance_margin)
        print(f"DEBUG: Simulator geïnitialiseerd met clearance_margin = {self.clearance_margin}")
        self.target_torque_ratio = target_torque_ratio


        # State - Initialiseer met None
        self.gears: List[GearSet] = []
        self.last_gear: Optional[GearSet] = None
        self.input_gear: Optional[GearSet] = None
        self.output_gear: Optional[GearSet] = None
        self.distance_on_path: float = 0.0
        self.current_path_index = 0
        self._prev_s: float = 0.0
        # Variabele voor reward shaping (toegevoegd)
        self._prev_dist_to_target: float = 0.0 

    def _normalize_point(self, p) -> Point:
        """Convert any point format to Point object."""
        if isinstance(p, Point): return Point(x=float(p.x), y=float(p.y))
        if isinstance(p, dict) and 'x' in p and 'y' in p: return Point(x=float(p['x']), y=float(p['y']))
        if isinstance(p, (list, tuple)) and len(p) >= 2: return Point(x=float(p[0]), y=float(p[1]))
        if hasattr(p, 'x') and hasattr(p, 'y'): return Point(x=float(p.x), y=float(p.y))
        raise ValueError(f"Kan {type(p)} niet converteren naar Point: {p}")

    def _normalize_path(self, path):
        """Converts the input path to a standardized list of Point objects."""
        if path is None:
            return []
        if (isinstance(path, np.ndarray) and path.size == 0) or (not isinstance(path, np.ndarray) and not path):
            return []
        normalized_path = []
        for p in path:
            try:
                if isinstance(p, Point):
                    normalized_path.append(p)
                elif hasattr(p, '__len__') and len(p) == 2:
                    normalized_path.append(Point(x=p[0], y=p[1]))
                else:
                    print(f"WAARSCHUWING: Kon padpunt {p} niet converteren naar Point object.")
            except Exception as e:
                print(f"FOUT bij converteren van padpunt {p}: {e}")
        return normalized_path

    def _normalize_boundaries(self, boundaries) -> List[Point]:
        """Convert boundaries to list of Point objects."""
        if not boundaries: return []
        normalized = []
        for i, b in enumerate(boundaries):
            try: normalized.append(self._normalize_point(b))
            except Exception as e: print(f"WAARSCHUWING: Kon grenspunt {i} ({b}) niet normaliseren: {e}"); continue
        return normalized

    # ------------------ Public API ------------------

    def _is_inside(self, point: Point, boundaries: List[Point]) -> bool:
        """Controleert of een punt binnen een polygoon ligt."""
        x, y = point.x, point.y
        n = len(boundaries)
        if n < 3: return False
        inside = False
        p1 = boundaries[0]
        for i in range(1, n + 1):
            p2 = boundaries[i % n]
            if y > min(p1.y, p2.y):
                if y <= max(p1.y, p2.y):
                    if x <= max(p1.x, p2.x):
                        if p1.y != p2.y:
                            x_intersection = (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
                        if p1.x == p2.x or x <= x_intersection:
                            inside = not inside
            p1 = p2
        return inside
        
    def reset(self, initial_gear_teeth: int = 20):
        """Resets the simulator state."""
        self.gears = []
        self.last_gear = None
        self.input_gear = None
        self.output_gear = None
        self.distance_on_path = 0.0
        self.current_path_index = 0
        self._prev_s = 0.0
        print(f"DEBUG Reset: Start reset...")

        try:
             max_input_radius = self._distance_to_boundary(self.input_shaft, self.boundaries) - self.clearance_margin
             max_output_radius = self._distance_to_boundary(self.output_shaft, self.boundaries) - self.clearance_margin
             print(f"DEBUG Reset: Max radii berekend: Input={max_input_radius:.2f}, Output={max_output_radius:.2f}")
        except Exception as e:
             print(f"FOUT Reset: Kon afstanden tot grens niet berekenen: {e}")
             return self._get_state_on_error(), 0.0, True, {"error": "Fout bij berekenen grensafstanden."}

        if max_input_radius <= 0.1 or max_output_radius <= 0.1:
            print(f"FOUT Reset: Input/Output as te dicht bij grens. Max Radii: Input={max_input_radius:.2f}, Output={max_output_radius:.2f}")
            return self._get_state_on_error(), 0.0, True, {"error": "Input/Output as te dicht bij grens."}

        try:
            self.input_gear = self.gear_factory.create_gear_from_diameter(
                gear_id='gear_input',
                center=(self.input_shaft.x, self.input_shaft.y),
                desired_diameter=max(0.1, max_input_radius * 2.0)
            )
            self.output_gear = self.gear_factory.create_gear_from_diameter(
                gear_id='gear_output',
                center=(self.output_shaft.x, self.output_shaft.y),
                desired_diameter=max(0.1, max_output_radius * 2.0)
            )
            if not self.input_gear or not self.output_gear:
                 raise ValueError("Kon input of output tandwiel niet aanmaken (diameter te klein?).")
            print(f"DEBUG Reset: Vaste tandwielen gemaakt: Input={self.input_gear.teeth_count}, Output={self.output_gear.teeth_count}")

        except Exception as e:
            print(f"FOUT Reset: Kon input/output tandwielen niet aanmaken: {e}")
            return self._get_state_on_error(), 0.0, True, {"error": f"Fout bij aanmaken input/output tandwielen: {e}"}

        self.gears = [self.input_gear, self.output_gear]
        self.last_gear = self.input_gear
        
        try:
            self.distance_on_path = self._get_path_distance_of_point(self.input_shaft)
            self._prev_s = self.distance_on_path
            self.current_path_index = self._find_segment_index(self.distance_on_path)
            
            # --- START AANPASSING (voor Reward Shaping) ---
            # Sla de initiële afstand tot het doel op
            self._prev_dist_to_target = self._distance(self.input_shaft, self.output_shaft)
            # --- EINDE AANPASSING ---
            
            print(f"DEBUG Reset: Start s={self.distance_on_path:.2f} op segment {self.current_path_index}")
        except Exception as e:
            print(f"FOUT Reset: Kon startafstand op pad niet bepalen: {e}")
            return self._get_state_on_error(), 0.0, True, {"error": f"Fout bij bepalen startpositie: {e}"}

        return self._get_state(), 0.0, False, {}

    # VERVANG UW HUIDIGE step() METHODE IN simulator.py MET DEZE:

    # VERVANG UW HUIDIGE step() METHODE IN simulator.py MET DEZE:

    # VERVANG UW HUIDIGE step() METHODE IN simulator.py MET DEZE:

    def step(self, action: tuple):
        """Plaatst een nieuw tussenliggend tandwiel."""
        driven_teeth, driving_teeth = action

        if driven_teeth < self.min_teeth or driving_teeth < self.min_teeth:
            error_msg = f"Gear teeth count too small (min {self.min_teeth})"
            return self._get_state(), -10.0, True, {"error": error_msg}

        if not self.last_gear:
             return self._get_state_on_error(), 0.0, True, {"error": "Simulator niet correct gereset (geen last_gear)."}

        # --- 1. Probeer Tangentieel te Plaatsen (Ongewijzigd) ---
        new_gear: Optional[GearSet] = None
        s_star: Optional[float] = None
        try:
             meshing_distance = self.gear_factory.get_meshing_distance(
                 self.last_gear.teeth_count[-1], driven_teeth
             )
             s_guess = max(self.distance_on_path + 0.1, self.distance_on_path + meshing_distance)
             intermediate_gear_index = len(self.gears) - 1
             new_gear_id = f'gear_{intermediate_gear_index}'

             placement_result = self._place_tangent_to_prev(
                 new_gear_id=new_gear_id,
                 driven_teeth=driven_teeth,
                 driving_teeth=driving_teeth,
                 s_guess=s_guess
             )
             if placement_result and placement_result[0] is not None:
                 new_gear, s_star, snap_error = placement_result
             else:
                 new_gear = None
        except Exception as e:
             new_gear = None

        # --- 2. Valideer de Plaatsing (Ongewijzigd) ---
        if new_gear:
            is_valid, validation_msg, details = self.validate_placement(new_gear, check_output_collision=False)
            if not is_valid:
                return self._get_state(), 0.0, True, {"error": validation_msg, "details": details}
        else:
            return self._get_state(), -10.0, True, {"error": "Placement calculation failed (no valid tangent position found)"}

        # --- 3. Plaatsing is Geldig: Update Staat (Ongewijzigd) ---
        self.distance_on_path = s_star
        self.current_path_index = self._find_segment_index(s_star)
        self.gears.insert(-1, new_gear) # Voeg in *voor* de output gear
        self.last_gear = new_gear
        
        # --- 4. Bereken Reward & Check Succesconditie ---
        reward = 0.0
        done = False
        info = {}

        current_state = self._get_state()
        if current_state is None:
             return self._get_state_on_error(), -500, True, {"error": "Kon staat niet ophalen."}

        # (Beloning voor vooruitgang op het pad)
        progress = self.distance_on_path - self._prev_s
        if progress > 0.1:
            reward += 1.0 * progress
        self._prev_s = self.distance_on_path
        
        # (Beloning voor dichterbij komen)
        current_dist_to_target = current_state["distance_to_target"]
        dist_reduction = self._prev_dist_to_target - current_dist_to_target
        reward += 5.0 * dist_reduction
        self._prev_dist_to_target = current_dist_to_target

        # --- START CORRECTIE: SUCCES CHECK ---
        
        # Bereken de AFSTAND tot de output gear
        dist_to_output_center = self._distance(self.last_gear.center, self.output_gear.center)
        
        # Bereken de VEREISTE AFSTAND voor een perfecte mesh
        required_mesh_distance = self.last_gear.driving_radius + self.output_gear.driven_radius
        
        # Bepaal de fout (Positief = opening, Negatief = overlap)
        error = dist_to_output_center - required_mesh_distance
        #Gewerkt met precieze tolleranties om een nauwkeurig resultaat te krijgen
        # Tolerantie voor een succesvolle mesh (alleen een OPENING)
        MESH_TOLERANCE = 0.005 # Max 0.005mm opening is succes
        
        # Tolerantie voor een botsing (elke OVERLAP)
        COLLISION_TOLERANCE = -0.001 # Max 0.001mm overlap is toegestaan

        # 1. Check op BOTSING (error is te negatief)
        if error < COLLISION_TOLERANCE:
            reward -= 1000.0  # Zware straf voor het botsen met het einddoel
            done = True
            info["error"] = f"Collision with output gear (overlap: {error:.2f})"

        # 2. Check op MESH (error is een grote getal of klein afhankelijk van de complexiteit van de figuur)
        elif error <= MESH_TOLERANCE:
            reward -= 500.0  # Zware straf voor het op elkaar liggen van de tandwielen
            # SUCCES!
            done = True
            info["success"] = f"Finale mesh bereikt (opening: {error:.2f})."
            
            # --- DATA VERZAMELEN VOOR REWARD.PY ---
            current_ratio = self.calculate_current_torque_ratio()
            
            # Bereken massa/oppervlakte (simpele benadering)
            total_mass = self.calculate_total_weight() # U heeft deze methode al in simulator.py
            total_area = self.calculate_space_efficiency() # U heeft deze ook
            
            # --- ROEP DE EXTERNE FUNCTIE AAN ---
            # Dit vervangt de hardcoded "100 * quality" logica
            try:
                final_reward = compute_terminal_reward(
                    current_ratio=current_ratio,
                    target_ratio=self.target_torque_ratio,
                    total_mass=total_mass,
                    total_area=total_area,
                    torque_weight=1.0,       # Pas deze gewichten aan naar wens
                    space_weight=0.05,       # Klein gewicht om geometrie niet te verstoren
                    weight_penalty_coef=0.05
                )
            except NameError:
                # Fallback als import mislukte: gebruik de oude logica
                torque_error = abs(current_ratio - self.target_torque_ratio)
                final_reward = 100.0 * math.exp(-5.0 * torque_error)
            
            reward += final_reward
            
            print(f"DEBUG MESH: Ratio={current_ratio:.2f}, Reward via reward.py={final_reward:.2f}")

        if not done and self.distance_on_path >= self._path_total_length() - (self.output_gear.driven_radius * 0.5):
            reward -= 50.0
            done = True
            info["error"] = "Reached end of path without successful mesh to output gear."

        return current_state, reward, done, info

    def validate_placement(self, gear_to_validate: GearSet, check_output_collision: bool = True) -> Tuple[bool, str, Dict]:
        """Valideert of een tandwiel geldig geplaatst kan worden."""
        try:
            if not self._is_inside(gear_to_validate.center, self.boundaries):
                return False, "Invalid placement - center is outside the boundaries", {"violation": "outside"}

            # 1. Grens Controle
            max_radius = max(gear_to_validate.radii) if gear_to_validate.radii else 0
            if max_radius <= 1e-6: 
                return False, "Invalid gear - zero radius", {"violation": "radius"}

            dist_to_boundary = self._distance_to_boundary(gear_to_validate.center, self.boundaries)
            required_dist_boundary = max_radius + self.clearance_margin

            if dist_to_boundary < required_dist_boundary - 1e-6:
                overlap = required_dist_boundary - dist_to_boundary
                msg = f"Invalid placement - too close to boundary (dist={dist_to_boundary:.3f}, req={required_dist_boundary:.3f})"
                details = {"violation": "boundary", "overlap": overlap}
                return False, msg, details

            # 2. Botsing Controle
            for existing_gear in self.gears:
                if existing_gear.id == gear_to_validate.id: continue
                
                # Sla check met 'last_gear' over (daar moeten we aan meshen)
                if self.last_gear and existing_gear.id == self.last_gear.id:
                    continue
                    
                # Sla check met 'output_gear' over, TENZIJ check_output_collision True is
                if not check_output_collision and existing_gear.id == 'gear_output':
                    continue

                dist_centers = self._distance(gear_to_validate.center, existing_gear.center)
                existing_max_radius = max(existing_gear.radii) if existing_gear.radii else 0
                
                # Gebruik pitch radius + margin voor botsing check
                required_dist_collision = max_radius + existing_max_radius + (self.clearance_margin * 0.5)

                if dist_centers < required_dist_collision - 1e-6:
                    overlap = required_dist_collision - dist_centers
                    msg = f"Invalid placement - collision with {existing_gear.id} (dist={dist_centers:.3f}, req={required_dist_collision:.3f})"
                    details = {"violation": "collision", "overlap": overlap}
                    return False, msg, details
            
            return True, "Valid placement", {}

        except Exception as e:
            return False, f"Exception during validation: {e}", {"violation": "exception"}

    def check_final_mesh(self, tol: float = 0.2) -> bool:
         """Controleert of 'self.last_gear' mesht met 'self.output_gear'."""
         if not self.last_gear or self.last_gear.id == 'gear_input' or not self.output_gear:
             return False
         try:
             dist_centers = self._distance(self.last_gear.center, self.output_gear.center)
             required_dist = self.last_gear.driving_radius + self.output_gear.driven_radius
             is_meshed = abs(required_dist - dist_centers) <= tol
             return is_meshed
         except Exception as e:
              print(f"WAARSCHUWING: Fout tijdens check_final_mesh: {e}")
              return False

    # ------------------ State & helpers ------------------
    
    def _get_state(self) -> Optional[Dict]:
        """Retourneert de huidige staat gebaseerd op self.last_gear."""
        current_gear_for_state = self.last_gear if (self.last_gear and self.last_gear.id != 'gear_input') else self.input_gear

        if not current_gear_for_state:
             print("FOUT: Kan staat niet ophalen, geen input_gear beschikbaar na reset.")
             return None

        target_x = self.output_shaft.x
        target_y = self.output_shaft.y
        dist_to_target = self._distance(current_gear_for_state.center, self.output_shaft)

        state = {
            "last_gear_center_x": float(current_gear_for_state.center.x),
            "last_gear_center_y": float(current_gear_for_state.center.y),
            "last_gear_teeth": int(current_gear_for_state.teeth_count[-1]),
            "last_gear_radius": float(current_gear_for_state.driving_radius),
            "target_gear_center_x": float(target_x),
            "target_gear_center_y": float(target_y),
            "distance_to_target": float(dist_to_target),
        }
        return state

    def _get_state_on_error(self) -> Dict:
        """ Geeft een dummy state terug bij een fatale fout. """
        print("WAARSCHUWING: _get_state_on_error aangeroepen.")
        center_x = getattr(self.input_shaft, 'x', 0)
        center_y = getattr(self.input_shaft, 'y', 0)
        target_x = getattr(self.output_shaft, 'x', 100)
        target_y = getattr(self.output_shaft, 'y', 0)
        dist = self._distance(Point(x=center_x, y=center_y), Point(x=target_x, y=target_y))

        return {
            "last_gear_center_x": float(center_x),
            "last_gear_center_y": float(center_y),
            "last_gear_teeth": 0,
            "last_gear_radius": 0.0,
            "target_gear_center_x": float(target_x),
            "target_gear_center_y": float(target_y),
            "distance_to_target": float(dist),
        }

    def _place_tangent_to_prev(self, new_gear_id: str, driven_teeth: int, driving_teeth: int, s_guess: float) -> Tuple[Optional[GearSet], Optional[float], Optional[float]]:
        """Probeert een nieuw tandwiel te plaatsen dat tangentieel is aan self.last_gear."""
        if not self.last_gear:
            print("FOUT: Kan niet tangentieel plaatsen zonder 'last_gear'.")
            return None, None, None
        try:
            temp_gear = self.gear_factory.create_gear(
                gear_id='gear_temp', center=(0, 0),
                num_teeth=[driven_teeth, driving_teeth] if driven_teeth != driving_teeth else [driven_teeth]
            )
            if not temp_gear or not temp_gear.radii:
                return None, None, None

            required_dist = self.last_gear.driving_radius + temp_gear.driven_radius
            if required_dist < 1e-6:
                 print(f"WAARSCHUWING PlaceTangent: Vereiste afstand is nul ({required_dist:.3f})")
                 return None, None, None

            # Verhoogde tolerantie voor de snap
            snap_result = self._snap_along_path_to_distance(
                s_guess=s_guess, target_point=self.last_gear.center,
                required_dist=required_dist, tol=0.5 # RUIMERE TOLERANTIE
            )

            if snap_result is None or snap_result[0] is None:
                return None, None, None

            s_star, p_star, err = snap_result

            final_gear = self.gear_factory.create_gear(
                gear_id=new_gear_id,
                center=(p_star.x, p_star.y),
                num_teeth=temp_gear.teeth_count
            )
            if not final_gear:
                 return None, None, None

            return final_gear, s_star, err
        except Exception as e:
            print(f"WAARSCHUWING: Fout tijdens _place_tangent_to_prev: {e}")
            return None, None, None

    # --- Andere hulpfuncties (onveranderd) ---
    def _distance(self, p1: Union[Point, Tuple, List, Dict], p2: Union[Point, Tuple, List, Dict]) -> float:
        try:
             if hasattr(p1,'x') and hasattr(p2,'x'):
                  dx = p1.x - p2.x; dy = p1.y - p2.y
                  return math.hypot(dx, dy)
             else:
                  np1 = self._normalize_point(p1); np2 = self._normalize_point(p2)
                  return math.hypot(np1.x - np2.x, np1.y - np2.y)
        except Exception as e:
             return float('inf')

    def _point_to_segment_distance(self, p: Point, v: Point, w: Point) -> float:
        try:
            l2 = self._distance(v, w) ** 2
            if l2 < 1e-12: return self._distance(p, v)
            dot_product = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y))
            t = max(0.0, min(1.0, dot_product / l2 if l2 > 1e-9 else 0.0))
            projection = Point(x=v.x + t * (w.x - v.x), y=v.y + t * (w.y - v.y))
            return self._distance(p, projection)
        except Exception as e:
            return float('inf')

    def _distance_to_boundary(self, point: Point, boundaries: List[Point]) -> float:
        if not boundaries or len(boundaries) < 2: return float('inf')
        min_dist = float('inf'); num_points = len(boundaries)
        for i in range(num_points):
            try:
                p1 = boundaries[i]
                p2 = boundaries[(i + 1) % num_points]
                dist = self._point_to_segment_distance(point, p1, p2)
                min_dist = min(min_dist, dist)
            except Exception as e:
                continue
        return min_dist if min_dist != float('inf') else 0.0

    def _find_point_on_path(self, target_distance: float) -> Optional[Point]:
        if not self.path or len(self.path) < 2: return None
        target_distance = max(0.0, min(target_distance, self._path_total_length()))
        cumulative_dist = 0.0
        for i in range(len(self.path) - 1):
            p1 = self.path[i]; p2 = self.path[i+1]; segment_len = self._distance(p1, p2)
            if segment_len < 1e-9: continue
            if cumulative_dist + segment_len >= target_distance - 1e-9:
                ratio = (target_distance - cumulative_dist) / segment_len
                ratio = max(0.0, min(1.0, ratio))
                return Point(x=p1.x + ratio * (p2.x - p1.x), y=p1.y + ratio * (p2.y - p1.y))
            cumulative_dist += segment_len
        return self.path[-1]

    def _get_path_distance_of_point(self, point: Point) -> float:
        if not self.path or len(self.path) < 2: return 0.0
        min_dist_sq_to_segment = float('inf'); cumulative_dist_at_projection = 0.0; current_cumulative_dist = 0.0
        for i in range(len(self.path) - 1):
            p1 = self.path[i]; p2 = self.path[i+1]; segment_len = self._distance(p1, p2)
            if segment_len < 1e-9: continue
            l2 = segment_len ** 2
            dot_product = ((point.x - p1.x) * (p2.x - p1.x) + (point.y - p1.y) * (p2.y - p1.y))
            t = max(0.0, min(1.0, dot_product / l2 if l2 > 1e-9 else 0.0))
            projection = Point(x=p1.x + t * (p2.x - p1.x), y=p1.y + t * (p2.y - p1.y))
            dist_sq = (point.x - projection.x)**2 + (point.y - projection.y)**2
            if dist_sq < min_dist_sq_to_segment:
                min_dist_sq_to_segment = dist_sq
                dist_along_segment = t * segment_len
                cumulative_dist_at_projection = current_cumulative_dist + dist_along_segment
            current_cumulative_dist += segment_len
        return cumulative_dist_at_projection

    def _find_segment_index(self, s: float) -> int:
         if not self.path or len(self.path) < 2: return 0
         s = max(0.0, min(s, self._path_total_length()))
         cumulative_dist = 0.0
         for i in range(len(self.path) - 1):
              try: segment_len = self._distance(self.path[i], self.path[i+1])
              except Exception: segment_len = 0
              if cumulative_dist + segment_len >= s - 1e-9: return i
              cumulative_dist += segment_len
         return max(0, len(self.path) - 2)

    def _path_total_length(self) -> float:
        if not hasattr(self, '_cached_path_length'):
             if not self.path or len(self.path) < 2: self._cached_path_length = 0.0
             else:
                  length = 0.0
                  for i in range(len(self.path) - 1):
                       try: length += self._distance(self.path[i], self.path[i+1])
                       except Exception: continue
                  self._cached_path_length = length
        return self._cached_path_length

    def _distance_to_point_at(self, s: float, pt: Point) -> float:
        point_on_path = self._find_point_on_path(s)
        if point_on_path is None: return float('inf')
        return self._distance(point_on_path, pt)

    def _snap_along_path_to_distance(self, s_guess: float, target_point: Point, required_dist: float,
                                    search_half_window: float = 25.0, tol: float = 0.15, iters: int = 15):
        path_len = self._path_total_length()
        if path_len < 1e-6: return None, None, None
        s_min = max(self.distance_on_path, s_guess - search_half_window)
        s_max = min(path_len, s_guess + search_half_window)
        if s_max <= s_min + 1e-6: s_max = s_min + 1.0

        f = lambda s_val: self._distance_to_point_at(s_val, target_point) - required_dist
        try: fa = f(s_min); fb = f(s_max)
        except Exception: return None, None, None

        if fa * fb >= 0:
            best_s, best_p, best_err = None, None, float('inf')
            for s_check in [s_min, s_max, s_guess]:
                 if s_min <= s_check <= s_max:
                      try:
                           f_check = f(s_check)
                           if abs(f_check) < abs(best_err):
                                p_check = self._find_point_on_path(s_check)
                                if p_check: best_s, best_p, best_err = s_check, p_check, f_check
                      except Exception: continue
            return (best_s, best_p, best_err) if best_p and abs(best_err) <= tol else (None, None, None)

        a, b = s_min, s_max; s_star, p_star, err = None, None, float('inf')
        for _ in range(iters):
            m = (a + b) / 2.0
            if abs(b - a) < 1e-4: break
            try: fm = f(m)
            except Exception: break
            current_p = self._find_point_on_path(m)
            if current_p and abs(fm) < abs(err): s_star, p_star, err = m, current_p, fm
            if abs(fm) <= tol: break
            if fa * fm < 0: b = m
            else: a, fa = m, fm
        return (s_star, p_star, err) if p_star and abs(err) <= tol else (None, None, None)

    # === Methoden voor torque/space/weight (placeholder) ===
    # Deze worden momenteel niet aangeroepen door de env, 
    # maar kunnen worden gebruikt voor complexere beloningen.

    def calculate_current_torque_ratio(self) -> float:
        """Berekent de cumulatieve koppelverhouding."""
        if not self.gears or len(self.gears) < 3:
            return 1.0
        total_ratio = 1.0
        is_complete = self.check_final_mesh(tol=0.5)
        end_gear_index = len(self.gears) -1 if is_complete else len(self.gears) - 2
        if not self.input_gear or not self.input_gear.teeth_count:
             print("WAARSCHUWING Torque Ratio: Input gear ongeldig.")
             return 1.0
        current_driving_teeth = self.input_gear.teeth_count[-1]
        for i in range(1, end_gear_index + 1):
             current_gear = self.gears[i]
             if not current_gear or not current_gear.teeth_count:
                  print(f"WAARSCHUWING Torque Ratio: Tandwiel op index {i} is ongeldig.")
                  current_driving_teeth = 0
                  continue
             current_driven_teeth = current_gear.teeth_count[0]
             if current_driving_teeth <= 0 or current_driven_teeth <= 0:
                  print(f"WAARSCHUWING Torque Ratio: Nul tanden gedetecteerd bij paar {i-1}-{i}.")
                  current_driving_teeth = current_gear.teeth_count[-1] if current_gear.teeth_count else 0
                  continue
             pair_ratio = float(current_driven_teeth) / float(current_driving_teeth)
             total_ratio *= pair_ratio
             current_driving_teeth = current_gear.teeth_count[-1]
        if abs(total_ratio) < 1e-6 or abs(total_ratio) > 1e6:
             print(f"WAARSCHUWING Torque Ratio: Extreem ({total_ratio:.2e}), retourneert 1.0.")
             return 1.0
        return total_ratio

    def calculate_space_efficiency(self) -> float:
        """Placeholder: Berekent hoe efficiënt de ruimte wordt gebruikt."""
        total_interm_area = 0.0
        for i in range(1, len(self.gears) - 1):
             gear = self.gears[i]
             if gear and hasattr(gear, 'radii') and gear.radii:
                  try:
                     max_r = max(gear.radii)
                     total_interm_area += math.pi * (max_r ** 2)
                  except ValueError: pass
                  except Exception as e:
                       print(f"WAARSCHUWING Space Efficiency: Fout: {e}")
        return total_interm_area * 0.001

    def calculate_total_weight(self) -> float:
        """Placeholder: Berekent gewicht van tussenliggende tandwielen."""
        total_interm_area = 0.0
        for i in range(1, len(self.gears) - 1):
             gear = self.gears[i]
             if gear and hasattr(gear, 'radii') and gear.radii:
                  try:
                       max_r = max(gear.radii)
                       total_interm_area += math.pi * (max_r ** 2)
                  except ValueError: pass
                  except Exception as e:
                       print(f"WAARSCHUWING Weight Calc: Fout: {e}")
        return total_interm_area * 0.01
    

    def get_available_clearance(self) -> float:
    #Calculates the distance from the active gear center to the nearest boundary.
    #Returns 0.0 if no active gear exists yet (start of episode).

        target_point = None
        
        # If we have a last gear, use its center
        if self.last_gear:
            target_point = self.last_gear.center
        # If not, use the input shaft (start of chain)
        elif self.input_shaft:
            target_point = self.input_shaft
            
        if target_point:
            dist = self._distance_to_boundary(target_point, self.boundaries)
            # Subtract the radius of the current gear to get "remaining" space, 
            # or just return the distance to center. 
            # Returning distance to center is safer/simpler for the AI to learn from.
            return float(dist)
            
        return 0.0