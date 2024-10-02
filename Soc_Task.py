import math

import numpy as np
import omni.kit
import torch
from gymnasium import spaces
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.nucleus import get_assets_root_path


class Soc_Task(BaseTask):
    def __init__(self, env, name, offset=None) -> None:
        self.env = env

    