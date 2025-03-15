import moderngl
import moderngl_window
import cv2
import numpy as np


class TestCamera(moderngl_window.WindowConfig):
    gl_version = (3, 3)
    title = "TestCamera"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ---------------------------------------
        # 1) Capture from default camera (ID=0)
        # ---------------------------------------
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        if not ret:
            print("[ERROR] Couldn't read from camera.")
            exit(1)

        # Dimensions of the first frame
        self.cam_h, self.cam_w = frame.shape[:2]

        # ---------------------------------------
        # 2) Create a simple pass-through shader
        # ---------------------------------------
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec2 in_uv;
                out vec2 uv;

                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                    uv = in_uv;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D cam_tex;
                in vec2 uv;
                out vec4 color;

                void main() {
                    // We'll flip vertically so OpenCV frames appear upright
                    vec2 flipped = vec2(uv.x, 1.0 - uv.y);
                    color = texture(cam_tex, flipped);
                }
            """
        )
        # Bind the uniform sampler to texture unit 0
        self.prog["cam_tex"].value = 0

        # ---------------------------------------
        # 3) Fullscreen quad data
        # ---------------------------------------
        quad_data = np.array([
            # x,    y,     u,    v
            -1.0, -1.0,   0.0,  0.0,
            1.0, -1.0,   1.0,  0.0,
            1.0,  1.0,   1.0,  1.0,

            -1.0, -1.0,   0.0,  0.0,
            1.0,  1.0,   1.0,  1.0,
            -1.0,  1.0,   0.0,  1.0,
        ], dtype=np.float32)

        self.vbo = self.ctx.buffer(quad_data.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "2f 2f", "in_pos", "in_uv")]
        )

        # ---------------------------------------
        # 4) Create an RGBA texture for the camera
        # ---------------------------------------
        self.cam_tex = self.ctx.texture(
            (self.cam_w, self.cam_h),
            4,  # RGBA channels
            dtype="u1"
        )
        self.cam_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    def on_render(self, time, frame_time):
        ret, frame = self.cap.read()
        if not ret:
            print("[WARNING] No frame read from camera")
            return

        # Flip horizontally so it looks like a mirror
        frame = cv2.flip(frame, 1)
        # Convert BGR (OpenCV) -> RGBA
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # Upload camera data to the texture
        self.cam_tex.write(frame_rgba.tobytes())

        # Clear the screen and draw the quad
        self.ctx.clear(0, 0, 0, 1)
        self.cam_tex.use(location=0)
        self.vao.render()

    def close(self):
        self.cap.release()
        super().close()


if __name__ == "__main__":
    moderngl_window.run_window_config(TestCamera)
