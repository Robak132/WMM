import moderngl
from pyrr import Matrix44

from base_window import BaseWindowConfig


class PhongWindow(BaseWindowConfig):
    VIEW_POSITION = (3.0, 1.0, -5.0)
    LIGHT_COLOR = (1.0, 1.0, 1.0)
    OBJECT_COLOR = (0.34, .354, .6)
    LIGHT_POSITION = (2.0, 1.0, 1.0)
    SHININESS = 8

    def __init__(self, **kwargs):
        super(PhongWindow, self).__init__(**kwargs)

    def init_shaders_variables(self):
        self.mvp = self.program['mvp']

    def render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 1.0, 1.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            (3.0, 1.0, -5.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        )
        self.mvp.write((proj * lookat).astype('float32'))
        self.vao.render()
