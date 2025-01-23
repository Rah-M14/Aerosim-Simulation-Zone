import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import deque
import time
from typing import Dict, List

class RewardMonitor:
    def __init__(self, max_steps=1000, history_length=100):
        self.max_steps = max_steps
        self.history_length = history_length
        
        # Initialize reward component histories
        self.histories = {
            'total_reward': deque(maxlen=history_length),
            'goal_potential': deque(maxlen=history_length),
            'path_potential': deque(maxlen=history_length),
            'progress': deque(maxlen=history_length),
            'path_following': deque(maxlen=history_length),
            'heading': deque(maxlen=history_length),
            'oscillation_penalty': deque(maxlen=history_length)
        }
        
        # Setup interactive plotting
        plt.ion()
        self.fig = plt.figure(figsize=(15, 10))
        self.gs = GridSpec(3, 3, self.fig)
        
        # Create subplots
        self.axes = {
            'total_reward': self.fig.add_subplot(self.gs[0, :]),
            'potentials': self.fig.add_subplot(self.gs[1, 0]),
            'progress': self.fig.add_subplot(self.gs[1, 1]),
            'path_following': self.fig.add_subplot(self.gs[1, 2]),
            'heading': self.fig.add_subplot(self.gs[2, 0]),
            'penalties': self.fig.add_subplot(self.gs[2, 1]),
            'reward_dist': self.fig.add_subplot(self.gs[2, 2])
        }
        
        # Initialize lines
        self.lines = {}
        self._setup_plots()
        
        # Statistics
        self.episode_rewards = []
        self.step_counter = 0
        
    def _setup_plots(self):
        """Initialize all plots with proper labels and styles"""
        # Total Reward
        self.axes['total_reward'].set_title('Total Reward Over Time')
        self.axes['total_reward'].set_xlabel('Step')
        self.axes['total_reward'].set_ylabel('Reward')
        self.lines['total_reward'] = self.axes['total_reward'].plot([], [], 'b-', label='Total Reward')[0]
        
        # Potentials
        self.axes['potentials'].set_title('Potential Components')
        self.axes['potentials'].set_xlabel('Step')
        self.axes['potentials'].set_ylabel('Value')
        self.lines['goal_potential'] = self.axes['potentials'].plot([], [], 'r-', label='Goal Potential')[0]
        self.lines['path_potential'] = self.axes['potentials'].plot([], [], 'g-', label='Path Potential')[0]
        self.axes['potentials'].legend()
        
        # Progress
        self.axes['progress'].set_title('Progress Reward')
        self.axes['progress'].set_xlabel('Step')
        self.axes['progress'].set_ylabel('Value')
        self.lines['progress'] = self.axes['progress'].plot([], [], 'g-')[0]
        
        # Path Following
        self.axes['path_following'].set_title('Path Following Reward')
        self.axes['path_following'].set_xlabel('Step')
        self.axes['path_following'].set_ylabel('Value')
        self.lines['path_following'] = self.axes['path_following'].plot([], [], 'b-')[0]
        
        # Heading
        self.axes['heading'].set_title('Heading Alignment')
        self.axes['heading'].set_xlabel('Step')
        self.axes['heading'].set_ylabel('Value')
        self.lines['heading'] = self.axes['heading'].plot([], [], 'y-')[0]
        
        # Penalties
        self.axes['penalties'].set_title('Penalties')
        self.axes['penalties'].set_xlabel('Step')
        self.axes['penalties'].set_ylabel('Value')
        self.lines['oscillation_penalty'] = self.axes['penalties'].plot([], [], 'orange', label='Oscillation')[0]
        self.axes['penalties'].legend()
        
        # Reward Distribution
        self.axes['reward_dist'].set_title('Reward Distribution')
        self.axes['reward_dist'].set_xlabel('Reward')
        self.axes['reward_dist'].set_ylabel('Frequency')
        
        plt.tight_layout()
        
    def update(self, reward_components: Dict[str, float]):
        """Update the visualization with new reward components"""
        self.step_counter += 1
        
        # Update histories
        for key, value in reward_components.items():
            if key in self.histories:
                self.histories[key].append(value)
        
        # Update line data
        x = list(range(max(0, self.step_counter - self.history_length), self.step_counter))
        
        # Update total reward plot
        self.lines['total_reward'].set_data(x, list(self.histories['total_reward']))
        self.axes['total_reward'].relim()
        self.axes['total_reward'].autoscale_view()
        
        # Update potential plots
        self.lines['goal_potential'].set_data(x, list(self.histories['goal_potential']))
        self.lines['path_potential'].set_data(x, list(self.histories['path_potential']))
        self.axes['potentials'].relim()
        self.axes['potentials'].autoscale_view()
        
        # Update other components
        for key in ['progress', 'path_following', 'heading']:
            self.lines[key].set_data(x, list(self.histories[key]))
            self.axes[key].relim()
            self.axes[key].autoscale_view()
        
        # Update penalties
        self.lines['oscillation_penalty'].set_data(x, list(self.histories['oscillation_penalty']))
        self.axes['penalties'].relim()
        self.axes['penalties'].autoscale_view()
        
        # Update reward distribution
        self.axes['reward_dist'].clear()
        self.axes['reward_dist'].set_title('Reward Distribution')
        self.axes['reward_dist'].hist(list(self.histories['total_reward']), bins=20)
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def reset(self):
        """Reset histories for new episode"""
        for history in self.histories.values():
            history.clear()
        self.step_counter = 0

    def get_data(self):
        """Returns the total reward history data for visualization"""
        if len(self.histories['total_reward']) > 0:
            return list(self.histories['total_reward'])
        return None