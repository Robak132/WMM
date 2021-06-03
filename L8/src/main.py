import moderngl_window

from base_window import BaseWindowConfig
from mandelbrot_window import MandelbrotWindowConfig

if __name__ == '__main__':
    moderngl_window.run_window_config(MandelbrotWindowConfig)
