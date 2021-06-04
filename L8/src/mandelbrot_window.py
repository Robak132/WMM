import moderngl
import numpy as np

from base_window import BaseWindowConfig


class MandelbrotWindowConfig(BaseWindowConfig):
    def __init__(self, **kwargs):
        super(MandelbrotWindowConfig, self).__init__(**kwargs)

    def init_shaders_variables(self):
        vertices = np.array([-1.0, -1.0,  # Bottom left
                             -1.0, 1.0,   # Bottom right
                             1.0, -1.0,   # Top left
                             1.0, 1.0],   # Top right
                            dtype='float32')

        self.vao = self.ctx.simple_vertex_array(self.program, self.ctx.buffer(vertices.data), 'in_position')
        self.center = self.program['center']
        self.scale = self.program['scale']
        self.iter = self.program['iterations']
        self.ratio = self.program['aspect_ratio']

    def render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 1.0, 1.0, 0.0)

        self.center.value = (0.0, 0.0)
        self.iter.value = 100
        self.scale.value = 1.0
        self.ratio.value = self.aspect_ratio

        self.vao.render(moderngl.TRIANGLE_STRIP)
