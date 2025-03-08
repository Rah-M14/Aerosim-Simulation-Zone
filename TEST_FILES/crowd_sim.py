import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import cv2
from typing import List, Dict, Tuple, Optional, Union
import time
from enum import Enum
import math
from dataclasses import dataclass
import random

class CrowdModelType(Enum):
    SFM = "social_force_model"
    ORCA = "optimal_reciprocal_collision_avoidance"

class BehaviorType(Enum):
    GOAL_DIRECTED = "goal_directed"
    WANDERING = "wandering"
    FOLLOWING = "following"
    GROUPED = "grouped"
    STATIONARY = "stationary"

class AgentType(Enum):
    NORMAL = "normal"
    FAST = "fast"
    SLOW = "slow"
    LARGE = "large"
    SMALL = "small"

@dataclass
class AgentParams:
    """Parameters for different agent types"""
    radius: float
    max_speed: float
    preferred_speed: float
    max_acceleration: float
    color: Tuple[float, float, float]  # RGB tuple (0-1)

# Parameters for different agent types
AGENT_PARAMS = {
    AgentType.NORMAL: AgentParams(
        radius=0.3, 
        max_speed=1.5, 
        preferred_speed=1.2, 
        max_acceleration=1.0,
        color=(0.2, 0.6, 1.0)  # Blue
    ),
    AgentType.FAST: AgentParams(
        radius=0.25, 
        max_speed=2.5, 
        preferred_speed=2.0, 
        max_acceleration=1.5,
        color=(1.0, 0.2, 0.2)  # Red
    ),
    AgentType.SLOW: AgentParams(
        radius=0.35, 
        max_speed=0.8, 
        preferred_speed=0.5, 
        max_acceleration=0.5,
        color=(0.1, 0.8, 0.2)  # Green
    ),
    AgentType.LARGE: AgentParams(
        radius=0.5, 
        max_speed=1.2, 
        preferred_speed=0.9, 
        max_acceleration=0.8,
        color=(0.8, 0.4, 0.0)  # Orange
    ),
    AgentType.SMALL: AgentParams(
        radius=0.2, 
        max_speed=1.8, 
        preferred_speed=1.3, 
        max_acceleration=1.2,
        color=(0.8, 0.8, 0.0)  # Yellow
    ),
}

# Behavior parameters
BEHAVIOR_PARAMS = {
    BehaviorType.GOAL_DIRECTED: {
        "goal_weight": 1.0,
        "obstacle_weight": 2.0,
        "agent_weight": 1.5,
        "path_change_probability": 0.01
    },
    BehaviorType.WANDERING: {
        "goal_weight": 0.3,
        "obstacle_weight": 2.0,
        "agent_weight": 1.0,
        "path_change_probability": 0.05,
        "wander_angle_change": 0.5
    },
    BehaviorType.FOLLOWING: {
        "goal_weight": 0.5,
        "obstacle_weight": 2.0,
        "agent_weight": 0.8,
        "follow_distance": 1.0,
        "follow_weight": 2.0
    },
    BehaviorType.GROUPED: {
        "goal_weight": 0.8,
        "obstacle_weight": 2.0,
        "agent_weight": 0.5,
        "group_cohesion_weight": 1.0,
        "group_separation": 0.8
    },
    BehaviorType.STATIONARY: {
        "goal_weight": 0.1,
        "obstacle_weight": 3.0,
        "agent_weight": 1.0,
        "movement_probability": 0.005
    }
}

class Agent:
    """Base class for a crowd agent"""
    def __init__(
        self,
        id: int,
        position: np.ndarray,
        goal: Optional[np.ndarray] = None,
        agent_type: AgentType = AgentType.NORMAL,
        behavior_type: BehaviorType = BehaviorType.GOAL_DIRECTED,
        model_type: CrowdModelType = CrowdModelType.SFM,
        world_limits: np.ndarray = np.array([[-10, 10], [-8, 8]]),
    ):
        self.id = id
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        
        # Set goal or random position within world limits if not provided
        if goal is None:
            self.goal = self._generate_random_goal(world_limits)
        else:
            self.goal = np.array(goal, dtype=np.float32)
            
        self.world_limits = world_limits
        self.agent_type = agent_type
        self.behavior_type = behavior_type
        self.model_type = model_type
        
        # Get parameters based on agent type
        params = AGENT_PARAMS[agent_type]
        self.radius = params.radius
        self.max_speed = params.max_speed
        self.preferred_speed = params.preferred_speed
        self.max_acceleration = params.max_acceleration
        self.color = params.color
        
        # Behavior parameters
        self.behavior_params = BEHAVIOR_PARAMS[behavior_type].copy()
        
        # For wandering behavior
        self.wander_angle = np.random.uniform(0, 2*np.pi)
        
        # For following behavior
        self.leader_id = None
        
        # For grouped behavior
        self.group_id = None
        self.group_members = []
        
        # Initialize time
        self.last_update_time = time.time()
        
        # For stationary agents that occasionally move
        self.is_moving = False
        self.stationary_timer = 0
        self.move_duration = 0
        
        # Path history for visualization
        self.path_history = [self.position.copy()]
        self.max_path_history = 50
        
    def _generate_random_goal(self, world_limits: np.ndarray) -> np.ndarray:
        """Generate a random goal position within world limits"""
        x = np.random.uniform(world_limits[0][0] + 1, world_limits[0][1] - 1)
        y = np.random.uniform(world_limits[1][0] + 1, world_limits[1][1] - 1)
        return np.array([x, y], dtype=np.float32)
    
    def _limit_speed(self, velocity: np.ndarray) -> np.ndarray:
        """Limit velocity to max_speed"""
        speed = np.linalg.norm(velocity)
        if speed > self.max_speed:
            velocity = (velocity / speed) * self.max_speed
        return velocity
    
    def _keep_in_bounds(self):
        """Keep agent within world limits"""
        # Check x-bounds
        if self.position[0] < self.world_limits[0][0] + self.radius:
            self.position[0] = self.world_limits[0][0] + self.radius
            self.velocity[0] = abs(self.velocity[0])  # Bounce
        elif self.position[0] > self.world_limits[0][1] - self.radius:
            self.position[0] = self.world_limits[0][1] - self.radius
            self.velocity[0] = -abs(self.velocity[0])  # Bounce
            
        # Check y-bounds
        if self.position[1] < self.world_limits[1][0] + self.radius:
            self.position[1] = self.world_limits[1][0] + self.radius
            self.velocity[1] = abs(self.velocity[1])  # Bounce
        elif self.position[1] > self.world_limits[1][1] - self.radius:
            self.position[1] = self.world_limits[1][1] - self.radius
            self.velocity[1] = -abs(self.velocity[1])  # Bounce
    
    def _update_goal(self, nearby_agents=None):
        """Update goal based on behavior type"""
        if self.behavior_type == BehaviorType.WANDERING:
            if np.random.random() < self.behavior_params["path_change_probability"]:
                self.wander_angle += np.random.uniform(-self.behavior_params["wander_angle_change"], 
                                                       self.behavior_params["wander_angle_change"])
                direction = np.array([np.cos(self.wander_angle), np.sin(self.wander_angle)])
                self.goal = self.position + direction * 3.0
                
                # Keep goal in bounds
                self.goal[0] = np.clip(self.goal[0], 
                                      self.world_limits[0][0] + self.radius,
                                      self.world_limits[0][1] - self.radius)
                self.goal[1] = np.clip(self.goal[1], 
                                      self.world_limits[1][0] + self.radius,
                                      self.world_limits[1][1] - self.radius)
        
        elif self.behavior_type == BehaviorType.FOLLOWING and nearby_agents is not None:
            if self.leader_id is None:
                # Find a leader to follow
                potential_leaders = [a for a in nearby_agents if a.id != self.id]
                if potential_leaders:
                    self.leader_id = np.random.choice(potential_leaders).id
            else:
                # Follow the leader
                leader = next((a for a in nearby_agents if a.id == self.leader_id), None)
                if leader:
                    # Follow but maintain distance
                    direction = leader.position - self.position
                    distance = np.linalg.norm(direction)
                    follow_distance = self.behavior_params["follow_distance"]
                    
                    if distance > follow_distance + leader.radius + self.radius:
                        # Move toward leader
                        self.goal = leader.position
                    else:
                        # Keep current position as goal to maintain distance
                        self.goal = self.position
                else:
                    # Lost leader, find a new one
                    self.leader_id = None
        
        elif self.behavior_type == BehaviorType.STATIONARY:
            # Occasionally move to a new spot
            if np.random.random() < self.behavior_params["movement_probability"] and not self.is_moving:
                self.is_moving = True
                self.goal = self._generate_random_goal(self.world_limits)
                self.move_duration = np.random.randint(5, 20)  # Move for 5-20 seconds
                self.stationary_timer = 0
            
            if self.is_moving:
                self.stationary_timer += 1
                if self.stationary_timer > self.move_duration:
                    self.is_moving = False
                    self.stationary_timer = 0
        
        # For goal-directed agents, check if goal reached and generate new one
        dist_to_goal = np.linalg.norm(self.position - self.goal)
        if dist_to_goal < self.radius:
            # Goal reached, generate new goal
            if self.behavior_type != BehaviorType.STATIONARY or self.is_moving:
                self.goal = self._generate_random_goal(self.world_limits)
    
    def update(self, dt: float, binary_img: np.ndarray, all_agents: List["Agent"]):
        """Update agent position based on its behavior and model"""
        # Store current time for velocity calculation
        current_time = time.time()
        
        # Find nearby agents
        nearby_agents = [a for a in all_agents if a.id != self.id and 
                         np.linalg.norm(a.position - self.position) < 5.0]
        
        # Update goal based on behavior
        self._update_goal(nearby_agents)
        
        # Calculate new velocity based on model type
        if self.model_type == CrowdModelType.SFM:
            self.velocity = self._compute_sfm_velocity(binary_img, nearby_agents)
        elif self.model_type == CrowdModelType.ORCA:
            self.velocity = self._compute_orca_velocity(binary_img, nearby_agents)
        
        # Limit speed
        self.velocity = self._limit_speed(self.velocity)
        
        # Update position
        self.position += self.velocity * dt
        
        # Keep agent in bounds
        self._keep_in_bounds()
        
        # Update last time
        self.last_update_time = current_time
        
        # Record position history
        self.path_history.append(self.position.copy())
        if len(self.path_history) > self.max_path_history:
            self.path_history = self.path_history[-self.max_path_history:]

    def _compute_sfm_velocity(self, binary_img: np.ndarray, nearby_agents: List["Agent"]) -> np.ndarray:
        """
        Compute velocity using the Social Force Model:
        1. Goal attraction force
        2. Agent-agent repulsion force
        3. Obstacle repulsion force
        """
        # Initialize forces
        force = np.zeros(2)
        
        # Goal attraction force
        if self.behavior_type != BehaviorType.STATIONARY or self.is_moving:
            goal_direction = self.goal - self.position
            distance_to_goal = np.linalg.norm(goal_direction)
            
            if distance_to_goal > 0.001:  # Avoid division by zero
                goal_direction /= distance_to_goal
                desired_velocity = goal_direction * self.preferred_speed
                goal_force = (desired_velocity - self.velocity) * self.behavior_params["goal_weight"]
                force += goal_force
        
        # Agent-agent repulsion forces
        for other in nearby_agents:
            if other.id != self.id:
                direction = self.position - other.position
                distance = np.linalg.norm(direction)
                
                # Minimum distance to maintain
                min_distance = self.radius + other.radius
                
                if distance < min_distance + 1.0:  # Only consider nearby agents
                    # Normalize direction
                    if distance > 0.001:  # Avoid division by zero
                        direction /= distance
                    else:
                        direction = np.array([1.0, 0.0])  # Default direction if too close
                    
                    # Calculate repulsion force (inversely proportional to distance)
                    repulsion_strength = self.behavior_params["agent_weight"] * (min_distance + 1.0 - distance) / (min_distance + 1.0)
                    repulsion_force = direction * repulsion_strength
                    
                    # Add to total force
                    force += repulsion_force
        
        # Obstacle repulsion forces (using binary image)
        # For simplicity, we'll use a simplified approach since calculating forces from binary image is complex
        obstacle_force = self._calculate_obstacle_forces(binary_img)
        force += obstacle_force * self.behavior_params["obstacle_weight"]
        
        # Apply acceleration limits
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > self.max_acceleration:
            force = (force / force_magnitude) * self.max_acceleration
        
        # Update velocity
        new_velocity = self.velocity + force
        
        return new_velocity
    
    def _compute_orca_velocity(self, binary_img: np.ndarray, nearby_agents: List["Agent"]) -> np.ndarray:
        """
        Compute velocity using ORCA (Optimal Reciprocal Collision Avoidance):
        1. Compute velocity obstacles for each nearby agent
        2. Find permissible velocities
        3. Select optimal velocity closest to preferred velocity
        
        This is a simplified implementation of ORCA.
        """
        # Desired velocity toward goal
        goal_direction = self.goal - self.position
        distance_to_goal = np.linalg.norm(goal_direction)
        
        if distance_to_goal < 0.001:  # Very close to goal
            preferred_velocity = np.zeros(2)
        else:
            preferred_velocity = (goal_direction / distance_to_goal) * self.preferred_speed
        
        # If stationary and not moving, prefer zero velocity
        if self.behavior_type == BehaviorType.STATIONARY and not self.is_moving:
            preferred_velocity = np.zeros(2)
            
        # Start with preferred velocity
        new_velocity = preferred_velocity.copy()
        
        # Time horizon for collision avoidance
        time_horizon = 2.0
        
        # Adjust velocity for each nearby agent
        for other in nearby_agents:
            if other.id != self.id:
                # Vector from this agent to other agent
                relative_position = other.position - self.position
                distance = np.linalg.norm(relative_position)
                
                # Combined radius of the two agents
                combined_radius = self.radius + other.radius
                
                # Skip if too far
                if distance > combined_radius + 3.0:
                    continue
                
                # Relative velocity
                relative_velocity = self.velocity - other.velocity
                
                # Check if collision will occur
                if distance < combined_radius:
                    # Already colliding, move away
                    collision_normal = relative_position / distance if distance > 0.001 else np.array([1.0, 0.0])
                    new_velocity = collision_normal * self.preferred_speed
                    continue
                
                # Time to collision
                ttc = self._compute_time_to_collision(relative_position, relative_velocity, combined_radius)
                
                # Skip if no collision within time horizon
                if ttc <= 0 or ttc > time_horizon:
                    continue
                
                # Compute collision avoidance velocity
                collision_point = self.position + self.velocity * ttc
                collision_normal = (collision_point - (other.position + other.velocity * ttc))
                if np.linalg.norm(collision_normal) > 0.001:
                    collision_normal = collision_normal / np.linalg.norm(collision_normal)
                else:
                    collision_normal = np.array([1.0, 0.0])
                
                # Adjust velocity to avoid collision
                adjustment = collision_normal * self.max_acceleration
                new_velocity += adjustment
        
        # Avoid obstacles
        obstacle_force = self._calculate_obstacle_forces(binary_img)
        new_velocity += obstacle_force * self.behavior_params["obstacle_weight"]
        
        return new_velocity
    
    def _compute_time_to_collision(self, relative_position: np.ndarray, relative_velocity: np.ndarray, 
                                  combined_radius: float) -> float:
        """Compute time to collision between two agents"""
        # Quadratic equation coefficients
        a = np.dot(relative_velocity, relative_velocity)
        if abs(a) < 1e-10:
            return float('inf')  # Agents moving in parallel
        
        b = 2 * np.dot(relative_position, relative_velocity)
        c = np.dot(relative_position, relative_position) - combined_radius * combined_radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return float('inf')  # No collision
        
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        
        if t1 < 0 and t2 < 0:
            return float('inf')  # Collision in the past
        
        return min(t for t in [t1, t2] if t > 0)
    
    def _calculate_obstacle_forces(self, binary_img: np.ndarray) -> np.ndarray:
        """
        Calculate repulsion forces from obstacles.
        This is a simplified approach that assumes obstacles are detected within a certain radius.
        """
        force = np.zeros(2)
        
        # Detect obstacles in 8 directions
        num_directions = 8
        max_detection_dist = 1.0
        
        for i in range(num_directions):
            angle = 2 * np.pi * i / num_directions
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Check for collision along this direction
            for dist in np.linspace(0.1, max_detection_dist, 5):
                check_point = self.position + direction * dist
                
                # Convert to image coordinates
                img_height, img_width = binary_img.shape
                world_width = self.world_limits[0][1] - self.world_limits[0][0]
                world_height = self.world_limits[1][1] - self.world_limits[1][0]
                
                x_img = int((check_point[0] - self.world_limits[0][0]) * (img_width / world_width))
                y_img = int((self.world_limits[1][1] - check_point[1]) * (img_height / world_height))
                
                # Check if coordinates are within the image bounds
                if 0 <= x_img < img_width and 0 <= y_img < img_height:
                    # Check if point is in obstacle (binary_img == 0 means obstacle)
                    if binary_img[y_img, x_img] == 0:
                        # Add repulsion force away from obstacle
                        repulsion = -direction * (max_detection_dist - dist) / max_detection_dist
                        force += repulsion
                        break
        
        return force

class CrowdSimulation:
    """Class to manage multiple agents and their interactions"""
    def __init__(self, 
                 binary_img_path: str,
                 world_limits: np.ndarray = np.array([[-10, 10], [-8, 8]]),
                 dt: float = 0.1):
        self.world_limits = world_limits
        self.dt = dt
        self.agents = []
        self.next_id = 0
        self.binary_img = self._load_binary_image(binary_img_path)
        self.fig = None
        self.ax = None
        
    def _load_binary_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess the binary image"""
        img = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
        if img is None:
            # Create a simple binary image if the file doesn't exist
            img = np.ones((200, 200), dtype=np.uint8) * 255
            # Add some obstacles
            img[80:120, 80:120] = 0  # Center obstacle
        
        # Threshold to create binary map (0=obstacle, 1=free space)
        binary_img = (img > 128).astype(np.uint8)
        binary_img = cv2.resize(binary_img, (0,0), fx=0.25, fy=0.25)
        return binary_img
    
    def add_agents(self, 
                  count: int, 
                  agent_type: AgentType = AgentType.NORMAL, 
                  behavior_type: BehaviorType = BehaviorType.GOAL_DIRECTED,
                  model_type: CrowdModelType = CrowdModelType.SFM,
                  positions: Optional[List[np.ndarray]] = None,
                  goals: Optional[List[np.ndarray]] = None) -> List[int]:
        """Add multiple agents with specified parameters"""
        agent_ids = []
        
        for i in range(count):
            position = positions[i] if positions and i < len(positions) else self._generate_valid_position()
            goal = goals[i] if goals and i < len(goals) else None
            
            agent = Agent(
                id=self.next_id,
                position=position,
                goal=goal,
                agent_type=agent_type,
                behavior_type=behavior_type,
                model_type=model_type,
                world_limits=self.world_limits
            )
            
            self.agents.append(agent)
            agent_ids.append(self.next_id)
            self.next_id += 1
            
        return agent_ids
    
    def add_group(self, 
                 count: int, 
                 agent_type: AgentType = AgentType.NORMAL,
                 model_type: CrowdModelType = CrowdModelType.SFM,
                 center_position: Optional[np.ndarray] = None) -> List[int]:
        """Add a group of agents that move together"""
        if center_position is None:
            center_position = self._generate_valid_position()
            
        group_id = f"group_{self.next_id}"
        agent_ids = []
        
        # Create agents in a circle around the center
        for i in range(count):
            angle = 2 * np.pi * i / count
            offset = np.array([np.cos(angle), np.sin(angle)]) * 0.5  # Half meter apart
            position = center_position + offset
            
            # Ensure position is valid
            position = self._adjust_to_valid_position(position)
            
            agent = Agent(
                id=self.next_id,
                position=position,
                goal=None,  # Will be set based on group movement
                agent_type=agent_type,
                behavior_type=BehaviorType.GROUPED,
                model_type=model_type,
                world_limits=self.world_limits
            )
            
            agent.group_id = group_id
            self.agents.append(agent)
            agent_ids.append(self.next_id)
            self.next_id += 1
        
        # Set group members for each agent
        for agent_id in agent_ids:
            agent = self.get_agent_by_id(agent_id)
            agent.group_members = [a_id for a_id in agent_ids if a_id != agent_id]
            
        # Set a common goal for the group
        group_goal = self._generate_valid_position()
        for agent_id in agent_ids:
            agent = self.get_agent_by_id(agent_id)
            agent.goal = group_goal
            
        return agent_ids
    
    def get_agent_by_id(self, agent_id: int) -> Optional[Agent]:
        """Get agent by ID"""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None
    
    def _generate_valid_position(self) -> np.ndarray:
        """Generate a random valid position (not in an obstacle)"""
        max_attempts = 50
        
        for _ in range(max_attempts):
            # Generate random position within world limits
            x = np.random.uniform(self.world_limits[0][0] + 0.5, self.world_limits[0][1] - 0.5)
            y = np.random.uniform(self.world_limits[1][0] + 0.5, self.world_limits[1][1] - 0.5)
            position = np.array([x, y])
            
            # Check if position is valid (not in obstacle)
            if self._is_valid_position(position):
                return position
                
        # If we couldn't find a valid position, return a default one
        return np.array([0.0, 0.0])
    
    def _adjust_to_valid_position(self, position: np.ndarray) -> np.ndarray:
        """Adjust a position to be valid (not in an obstacle)"""
        if self._is_valid_position(position):
            return position
            
        # Try to find a nearby valid position
        for radius in np.linspace(0.5, 3.0, 10):
            for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                test_pos = position + np.array([np.cos(angle), np.sin(angle)]) * radius
                if self._is_valid_position(test_pos):
                    return test_pos
                    
        # If we couldn't find a valid position, generate a random one
        return self._generate_valid_position()
    
    def _is_valid_position(self, position: np.ndarray) -> bool:
        """Check if a position is valid (not in an obstacle)"""
        # Convert position to image coordinates
        img_height, img_width = self.binary_img.shape
        world_width = self.world_limits[0][1] - self.world_limits[0][0]
        world_height = self.world_limits[1][1] - self.world_limits[1][0]
        
        x_img = int((position[0] - self.world_limits[0][0]) * (img_width / world_width))
        y_img = int((self.world_limits[1][1] - position[1]) * (img_height / world_height))
        
        # Check if within image bounds
        if 0 <= x_img < img_width and 0
