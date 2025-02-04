import cv2
import numpy as np

class EnvironmentRenderer:
    def __init__(self, env_world_limits, name):
        self.env_world_limits = env_world_limits
        self.name = name
        self.img_width = 800
        self.img_height = 600

    def world_to_img(self, x, y):
        # Transform from world coordinates to image coordinates using env limits
        x_range = self.env_world_limits[0][1] - self.env_world_limits[0][0]
        y_range = self.env_world_limits[1][1] - self.env_world_limits[1][0]
        
        img_x = int((x - self.env_world_limits[0][0]) * (self.img_width / x_range))
        img_y = int((self.env_world_limits[1][1] - y) * (self.img_height / y_range))
        return img_x, img_y

    def render_normal(self, env_state, path_manager, reward_manager):
        display_img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        
        # Draw grid
        x_min, x_max = self.env_world_limits[0]
        y_min, y_max = self.env_world_limits[1]
        
        # Draw vertical grid lines
        for x in range(int(x_min), int(x_max) + 1):
            x_img = self.world_to_img(x, 0)[0]
            cv2.line(display_img, (x_img, 0), (x_img, self.img_height), (50, 50, 50), 1)
        
        # Draw horizontal grid lines
        for y in range(int(y_min), int(y_max) + 1):
            y_img = self.world_to_img(0, y)[1]
            cv2.line(display_img, (0, y_img), (self.img_width, y_img), (50, 50, 50), 1)

        # Draw path
        if path_manager.get_full_path() is not None:
            path = path_manager.get_full_path()
            if len(path) > 1:
                for i in range(len(path) - 1):
                    start = self.world_to_img(path[i][0], path[i][1])
                    end = self.world_to_img(path[i+1][0], path[i+1][1])
                    cv2.line(display_img, start, end, (0, 255, 255), 2)

        # Draw agent
        agent_pos = self.world_to_img(env_state['current_pos'][0], env_state['current_pos'][1])
        triangle_size = 15
        angle = env_state['agent_theta']
        
        # Calculate triangle vertices
        tip = (
            int(agent_pos[0] + np.cos(angle) * triangle_size),
            int(agent_pos[1] - np.sin(angle) * triangle_size)
        )
        base_l = (
            int(agent_pos[0] - np.cos(angle + np.pi/6) * triangle_size),
            int(agent_pos[1] + np.sin(angle + np.pi/6) * triangle_size)
        )
        base_r = (
            int(agent_pos[0] - np.cos(angle - np.pi/6) * triangle_size),
            int(agent_pos[1] + np.sin(angle - np.pi/6) * triangle_size)
        )
        
        # Draw yellow triangle body
        triangle_pts = np.array([tip, base_l, base_r], np.int32)
        cv2.fillPoly(display_img, [triangle_pts], (0, 255, 255))
        
        # Draw red tip
        tip_size = 5
        red_tip = (
            int(agent_pos[0] + np.cos(angle) * triangle_size),
            int(agent_pos[1] - np.sin(angle) * triangle_size)
        )
        cv2.circle(display_img, red_tip, tip_size, (0, 0, 255), -1)
        
        # Draw goal
        goal_pos = self.world_to_img(env_state['goal_pos'][0], env_state['goal_pos'][1])
        cv2.circle(display_img, goal_pos, 10, (0, 255, 0), -1)

        # Draw episode info (top left)
        episode_info = [
            f"Episode: {env_state['episode_num']}",
            f"Step: {env_state['current_step']}",
            f"Reward: {env_state['episode_reward']:.2f}",
            f"Distance: {np.linalg.norm(env_state['current_pos'] - env_state['goal_pos']):.2f}",
            f"Angle: {np.degrees(env_state['agent_theta']):.1f}°"
        ]
        
        for i, text in enumerate(episode_info):
            cv2.putText(display_img, text, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw reward components and monitor
        if hasattr(reward_manager, 'monitor') and reward_manager.monitor is not None:
            self._draw_reward_info(display_img, reward_manager)

        # Show the image
        cv2.imshow(f'Path Following Environment - {self.name}', display_img)
        cv2.waitKey(1)

    def _draw_reward_info(self, display_img, reward_manager):
        # Draw reward components (top right)
        reward_text = [
            "Reward Components:",
            f"Goal Potential: {reward_manager.monitor.histories['goal_potential'][-1] if len(reward_manager.monitor.histories['goal_potential']) > 0 else 0:.2f}",
            f"Path Potential: {reward_manager.monitor.histories['path_potential'][-1] if len(reward_manager.monitor.histories['path_potential']) > 0 else 0:.2f}",
            f"Progress: {reward_manager.monitor.histories['progress'][-1] if len(reward_manager.monitor.histories['progress']) > 0 else 0:.2f}",
            f"Path Following: {reward_manager.monitor.histories['path_following'][-1] if len(reward_manager.monitor.histories['path_following']) > 0 else 0:.2f}",
            f"Heading: {reward_manager.monitor.histories['heading'][-1] if len(reward_manager.monitor.histories['heading']) > 0 else 0:.2f}",
            f"Oscillation: {reward_manager.monitor.histories['oscillation_penalty'][-1] if len(reward_manager.monitor.histories['oscillation_penalty']) > 0 else 0:.2f}",
            f"Total: {reward_manager.monitor.histories['total_reward'][-1] if len(reward_manager.monitor.histories['total_reward']) > 0 else 0:.2f}"
        ]
        
        for i, text in enumerate(reward_text):
            cv2.putText(display_img, text,
                    (self.img_width - 300, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw reward monitor plot (bottom right)
        monitor_data = reward_manager.monitor.get_data()
        if monitor_data is not None and len(monitor_data) > 1:
            self._draw_reward_plot(display_img, monitor_data)

    def _draw_reward_plot(self, display_img, monitor_data):
        plot_width = 250
        plot_height = 150
        plot_margin = 20
        plot_x = self.img_width - plot_width - plot_margin
        plot_y = self.img_height - plot_height - plot_margin
        
        # Draw plot background and border
        cv2.rectangle(display_img, 
                    (plot_x, plot_y),
                    (plot_x + plot_width, plot_y + plot_height),
                    (30, 30, 30), -1)
        cv2.rectangle(display_img,
                    (plot_x, plot_y),
                    (plot_x + plot_width, plot_y + plot_height),
                    (100, 100, 100), 1)
        
        # Scale and plot reward data
        rewards = np.array(monitor_data)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        reward_range = max_reward - min_reward if max_reward != min_reward else 1
        
        points = []
        for i in range(len(rewards)):
            x = plot_x + int((i / len(rewards)) * plot_width)
            normalized_reward = (rewards[i] - min_reward) / reward_range
            y = plot_y + plot_height - int(normalized_reward * plot_height)
            points.append((x, y))
        
        # Draw curve
        for i in range(len(points) - 1):
            cv2.line(display_img, points[i], points[i + 1], (0, 0, 255), 1)
        
        # Add min/max labels
        cv2.putText(display_img, f"max: {max_reward:.1f}",
                (plot_x + 5, plot_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display_img, f"min: {min_reward:.1f}",
                (plot_x + 5, plot_y + plot_height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1) 