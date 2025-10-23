import unittest
import math
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from common.data_models import ValidationReport, Constraints, Boundary, Point
from rl_agent.train import compute_reward

class TestRewardFunction(unittest.TestCase):
    def setUp(self):
        # Create minimal valid constraints
        boundary = Boundary(vertices=(Point(0,0), Point(10,0), Point(10,10), Point(0,10)))
        self.constraints = Constraints(boundary=boundary)
        
    def test_invalid_design(self):
        report = ValidationReport(is_valid=False)
        reward = compute_reward(
            report, 
            self.constraints,
            target_torque=2.0,
            torque_weight=0.6,
            space_weight=0.3,
            weight_penalty_coef=0.1
        )
        self.assertEqual(reward, -10.0)
        
    def test_perfect_match(self):
        report = ValidationReport(
            is_valid=True,
            torque_ratio=2.0,  # Exactly matches target
            space_usage=1.0,    # Max space usage
            total_mass=0.0      # Zero mass
        )
        reward = compute_reward(
            report, 
            self.constraints,
            target_torque=2.0,
            torque_weight=0.6,
            space_weight=0.3,
            weight_penalty_coef=0.1
        )
        # Should be: 0.6*1 + 0.3*1 - 0.1*0 = 0.9
        self.assertAlmostEqual(reward, 0.9, places=5)
        
    def test_torque_penalty(self):
        report = ValidationReport(
            is_valid=True,
            torque_ratio=3.0,  # Diff = 1.0
            space_usage=1.0,
            total_mass=0.0
        )
        reward = compute_reward(
            report, 
            self.constraints,
            target_torque=2.0,
            torque_weight=0.6,
            space_weight=0.3,
            weight_penalty_coef=0.1
        )
        # torque_reward = exp(-1.0) ≈ 0.367879
        expected = 0.6*0.367879 + 0.3*1.0 - 0.1*0
        self.assertAlmostEqual(reward, expected, places=5)
        
    def test_mass_penalty(self):
        report = ValidationReport(
            is_valid=True,
            torque_ratio=2.0,
            space_usage=1.0,
            total_mass=100.0  # High mass
        )
        reward = compute_reward(
            report, 
            self.constraints,
            target_torque=2.0,
            torque_weight=0.6,
            space_weight=0.3,
            weight_penalty_coef=0.1
        )
        # weight_penalty = 100 * 0.01 = 1.0
        # reward = 0.6*1 + 0.3*1 - 0.1*1.0 = 0.8
        self.assertAlmostEqual(reward, 0.8, places=5)
        
    def test_space_usage(self):
        report = ValidationReport(
            is_valid=True,
            torque_ratio=2.0,
            space_usage=0.5,  # Half space used
            total_mass=0.0
        )
        reward = compute_reward(
            report, 
            self.constraints,
            target_torque=2.0,
            torque_weight=0.6,
            space_weight=0.3,
            weight_penalty_coef=0.1
        )
        # Should be: 0.6*1 + 0.3*0.5 - 0.1*0 = 0.75
        self.assertAlmostEqual(reward, 0.75, places=5)
        
    def test_all_penalties(self):
        report = ValidationReport(
            is_valid=True,
            torque_ratio=3.0,  # Diff = 1.0
            space_usage=0.5,    # Half space
            total_mass=50.0     # Medium mass
        )
        reward = compute_reward(
            report, 
            self.constraints,
            target_torque=2.0,
            torque_weight=0.6,
            space_weight=0.3,
            weight_penalty_coef=0.1
        )
        torque_reward = math.exp(-1.0)
        weight_penalty = 50.0 * 0.01
        expected = (
            0.6 * torque_reward + 
            0.3 * 0.5 - 
            0.1 * weight_penalty
        )
        self.assertAlmostEqual(reward, expected, places=5)

if __name__ == '__main__':
    unittest.main()
