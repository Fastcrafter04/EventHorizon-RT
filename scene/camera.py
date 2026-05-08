import taichi as ti
import numpy as np


@ti.data_oriented
class Camera:
    def __init__(self, res_x, res_y):
        self.res = (res_x, res_y)
        self.aspect_ratio = res_x / res_y

    @ti.func
    def get_ray_dir(self, u, v):
        # Convert pixel (0 to 1) to screen space (-1 to 1)
        # This is basic Ray Initialization logic
        x = (2.0 * u - 1.0) * self.aspect_ratio
        y = 2.0 * v - 1.0
        return ti.Vector([x, y, -1.0]).normalized()