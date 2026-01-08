#####This code is debugged and adapted with help of G00GLE GEMINI############
import json
import math
from heapq import heappop, heappush
from typing import List, Tuple, Dict, Any

class Pathfinder:
    def find_path(self, processed_data_path, margin=5.0, step_size=0.5):
        # Your working shortest path method
        with open(processed_data_path, 'r') as f:
            data = json.load(f)['normalized_space']
        boundaries, start_node, end_node = self._load_data(data)
        if not self._is_valid(start_node, boundaries, margin): return None
        if not self._is_valid(end_node, boundaries, margin): return None
        open_set = [(0, start_node)]
        came_from = {}
        start_node_key = (round(start_node[0], 4), round(start_node[1], 4))
        g_score = {start_node_key: 0}
        while open_set:
            _, current = heappop(open_set)
            current_key = (round(current[0], 4), round(current[1], 4))
            if self._distance(current, end_node) < step_size:
                return self._reconstruct_path(came_from, current)
            for neighbor in self._get_neighbors(current, boundaries, end_node, step_size, margin):
                tentative_g_score = g_score.get(current_key, float('inf')) + self._distance(current, neighbor)
                neighbor_key = (round(neighbor[0], 4), round(neighbor[1], 4))
                if tentative_g_score < g_score.get(neighbor_key, float('inf')):
                    came_from[neighbor_key] = current
                    g_score[neighbor_key] = tentative_g_score
                    f_score = tentative_g_score + self._distance(neighbor, end_node)
                    heappush(open_set, (f_score, neighbor))
        return None

    def find_centerline_path(self, processed_data_path: str, margin=5.0,  step_size: float = 0.1, smoothing_iterations: int = 500, smoothing_amount: float = 0.2) -> Dict[str, Any] | None:
        """
        Finds a centered path using path smoothing.
        
        Args:
            processed_data_path: Path to the processed JSON file.
            smoothing_iterations: Number of times to push the path away from walls.
            smoothing_amount: How much to move the path in each iteration.
        """
        # Step 1: Get the shortest path as a starting point. Use a tiny margin.
        shortest_path_list = self.find_path(processed_data_path, margin=margin,step_size=step_size)
        if not shortest_path_list:
            return None

        with open(processed_data_path, 'r') as f:
            data = json.load(f)['normalized_space']
        boundaries, _, _ = self._load_data(data)
        
        path = [tuple(p) for p in shortest_path_list]

        # Step 2: Iteratively smooth the path by pushing points away from boundaries
        for _ in range(smoothing_iterations):
            new_path = [path[0]] # Keep the start point fixed
            
            # Iterate over the interior points of the path
            for i in range(1, len(path) - 1):
                point = path[i]
                closest_boundary_point = self._get_closest_boundary_point(point, boundaries)
                
                # Calculate the repulsion vector (from boundary point to path point)
                repulsion_vec = (point[0] - closest_boundary_point[0], point[1] - closest_boundary_point[1])
                
                # Normalize the vector
                magnitude = self._distance(repulsion_vec, (0,0))
                if magnitude < 1e-6:
                    unit_vec = (0, 0)
                else:
                    unit_vec = (repulsion_vec[0] / magnitude, repulsion_vec[1] / magnitude)
                
                # Move the point along the repulsion vector
                new_point = (point[0] + unit_vec[0] * smoothing_amount,
                             point[1] + unit_vec[1] * smoothing_amount)
                
                # Only update if the new point remains inside the boundary
                if self._is_inside(new_point, boundaries):
                    new_path.append(new_point)
                else:
                    new_path.append(point) # Otherwise, keep the old point

            new_path.append(path[-1]) # Keep the end point fixed
            path = new_path
        
        # Convert path back to list of lists for consistent output format
        final_path = [list(p) for p in path]
        min_clearance = self._calculate_path_clearance(final_path, boundaries)
        return path

    # --- HELPER METHODS ---
    def _load_data(self, data):
        boundaries = [tuple(p) for p in data['boundaries']]
        start_node = tuple(data['input_shaft'].values())
        end_node = tuple(data['output_shaft'].values())
        return boundaries, start_node, end_node

    def _get_closest_boundary_point(self, point, boundaries):
        min_dist = float('inf')
        closest_point = None
        for i in range(len(boundaries)):
            p1 = boundaries[i]
            p2 = boundaries[(i + 1) % len(boundaries)]
            proj_point, dist = self._get_projection_and_distance(point, p1, p2)
            if dist < min_dist:
                min_dist = dist
                closest_point = proj_point
        return closest_point

    def _get_projection_and_distance(self, p, v, w):
        l2 = self._distance(v, w)**2
        if l2 == 0.0: return v, self._distance(p, v)
        t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
        projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
        dist = self._distance(p, projection)
        return projection, dist

    def _calculate_path_clearance(self, path, boundaries):
        min_dist = float('inf')
        for point in path:
            dist = self._distance_to_boundary(point, boundaries)
            if dist < min_dist: min_dist = dist
        return min_dist
    def _reconstruct_path(self, came_from, current):
        path = [list(current)]
        current_key = (round(current[0], 4), round(current[1], 4))
        while current_key in came_from:
            current_tuple = came_from[current_key]
            path.append(list(current_tuple))
            current_key = (round(current_tuple[0], 4), round(current_tuple[1], 4))
        path.reverse()
        return path
    def _is_valid(self, point, boundaries, margin):
        if not self._is_inside(point, boundaries): return False
        if self._distance_to_boundary(point, boundaries) < margin: return False
        return True
    def _get_neighbors(self, point, boundaries, goal, step, margin):
        neighbors = []
        if self._has_line_of_sight(point, goal, boundaries, step, margin):
            neighbors.append(goal); return neighbors
        for dx in [-step, 0, step]:
            for dy in [-step, 0, step]:
                if dx == 0 and dy == 0: continue
                neighbor = (point[0] + dx, point[1] + dy)
                if self._is_valid(neighbor, boundaries, margin):
                    neighbors.append(neighbor)
        return neighbors
    def _has_line_of_sight(self, p1, p2, boundaries, resolution, margin):
        dist = self._distance(p1, p2)
        if dist < resolution: return True
        num_checks = int(dist / resolution)
        if num_checks == 0: return True
        dx, dy = (p2[0] - p1[0]) / num_checks, (p2[1] - p1[1]) / num_checks
        for i in range(1, num_checks):
            intermediate_point = (p1[0] + i * dx, p1[1] + i * dy)
            if not self._is_valid(intermediate_point, boundaries, margin): return False
        return True
    def _distance_to_boundary(self, point, boundaries):
        min_dist = float('inf')
        for i in range(len(boundaries)):
            dist = self._point_to_segment_distance(point, boundaries[i], boundaries[(i + 1) % len(boundaries)])
            if dist < min_dist: min_dist = dist
        return min_dist
    def _point_to_segment_distance(self, p, v, w):
        l2 = self._distance(v, w)**2
        if l2 == 0.0: return self._distance(p, v)
        t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
        projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
        return self._distance(p, projection)
    def _is_inside(self, point, boundaries):
        x, y = point; n = len(boundaries); inside = False
        p1 = boundaries[0]
        for i in range(1, n + 1):
            p2 = boundaries[i % n]; p1x, p1y = p1; p2x, p2y = p2
            if p1y == p2y:
                p1 = p2; continue
            if min(p1y, p2y) < y <= max(p1y, p2y):
                x_intersection = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if x_intersection > x: inside = not inside
            p1 = p2
        return inside
    def _distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)