import moderngl

from base_window import BaseWindowConfig


class MandelbrotWindowConfig(BaseWindowConfig):
    def __init__(self, **kwargs):
        super(MandelbrotWindowConfig, self).__init__(**kwargs)

    def init_shaders_variables(self):
        pass
        # TODO: Init shader variables

    def render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 1.0, 1.0, 0.0)

        # TODO: Write variables to shader

        self.vao.render(moderngl.TRIANGLE_STRIP)
