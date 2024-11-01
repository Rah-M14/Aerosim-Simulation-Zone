import numpy as np

class Action_Bot():
    def __init__(self, bot, controller, n_steps=5):
        self.bot = bot
        self.controller = controller
        self.n_steps = n_steps
    
    def move_bot(self, vals: np.array):
        for _ in range(self.n_steps):
            self.bot.apply_wheel_actions(self.controller.forward(vals))
        return None