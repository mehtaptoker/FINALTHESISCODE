import math
from common.data_models import ValidationReport, Constraints
def compute_reward(
    report: ValidationReport, 
    constraints: Constraints, 
    target_torque: float,
    torque_weight: float,
    space_weight: float,
    weight_penalty_coef: float
) -> float:
    """
    Calculate reward based on validation report and constraints
    
    Args:
        report: Validation report from physics validator
        constraints: Design constraints
        target_torque: Desired torque ratio
        torque_weight: Weight for torque component
        space_weight: Weight for space usage component
        weight_penalty_coef: Weight penalty coefficient
        
    Returns:
        float: Calculated reward scalar
    """
    # Heavy penalty for invalid designs (collisions, etc)
    if not report.is_valid:
        return -10.0  # Significant penalty for invalid designs
    
    # Calculate torque reward (exponential decay for closeness to target)
    torque_diff = abs(report.torque_ratio - target_torque)
    torque_reward = math.exp(-torque_diff)  # [0,1] range
    
    # Calculate space usage reward (higher = better)
    space_reward = report.space_usage
    
    # Calculate weight penalty (lower mass = better)
    weight_penalty = report.total_mass * 0.01  # Scale mass penalty
    
    # Weighted sum of components
    reward = (
        torque_weight * torque_reward + 
        space_weight * space_reward - 
        weight_penalty_coef * weight_penalty
    )
    return reward

def compute_terminal_reward(
    current_ratio: float,
    target_ratio: float,
    total_mass: float,
    total_area: float,
    torque_weight: float = 1.0,
    space_weight: float = 0.1,
    weight_penalty_coef: float = 0.1
) -> float:
    """
    Berekent de eindscore alleen bij een succesvolle mesh.
    """
    # 1. Torque Component (De belangrijkste)
    # Gebruik een scherpe exponentiële straf voor afwijkingen
    torque_diff = abs(current_ratio - target_ratio)
    # Strenger dan voorheen: -5.0 factor zorgt voor precisie
    torque_quality = math.exp(-5.0 * torque_diff) 
    torque_reward = 100.0 * torque_quality * torque_weight

    # 2. Space Component (Optioneel: beloon compactheid)
    # Hoe kleiner de oppervlakte, hoe beter? Of juist hoe efficiënter gevuld?
    # Laten we aannemen: minder oppervlakte gebruik is beter (kostenbesparing)
    space_penalty = total_area * space_weight

    # 3. Weight Component (Straf voor zwaar design)
    weight_penalty = total_mass * weight_penalty_coef

    # Totale Formule
    # Basis succesbonus (+100) zit verwerkt in de torque_reward (als ratio perfect is)
    final_reward = torque_reward - space_penalty - weight_penalty
    
    return final_reward