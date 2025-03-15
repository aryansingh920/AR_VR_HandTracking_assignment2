import moderngl
import moderngl_window as mglw
import numpy as np


class SimpleTest(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Simple Test"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_position;
                uniform float time;
                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                out vec4 f_color;
                uniform float time;
                void main() {
                    f_color = vec4(sin(time) * 0.5 + 0.5, 
                                  cos(time) * 0.5 + 0.5, 
                                  0.5, 1.0);
                }
            """
        )

        self.prog['time'] = 0.0

        vertices = np.array([
            -0.6, -0.6,
            0.6, -0.6,
            0.0, 0.6,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, '2f', 'in_position')]
        )

    def on_render(self, time, frame_time):
        self.ctx.clear(0.2, 0.2, 0.2, 1.0)
        self.prog['time'] = time
        self.vao.render(moderngl.TRIANGLES)


if __name__ == '__main__':
    mglw.run_window_config(SimpleTest)
