import sys

import moderngl_window

from base_window import BaseWindowConfig
from mandelbrot_window import MandelbrotWindowConfig
from robot_window import RobotWindow
from phong_window import PhongWindow

if __name__ == '__main__':
    if sys.argv[4] == "fractals":
        moderngl_window.run_window_config(MandelbrotWindowConfig)
    elif sys.argv[4] == "robot":
        moderngl_window.run_window_config(RobotWindow)
    elif sys.argv[4] == "shading":
        moderngl_window.run_window_config(PhongWindow)
    else:
        moderngl_window.run_window_config(BaseWindowConfig)
