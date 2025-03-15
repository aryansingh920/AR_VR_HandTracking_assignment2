"""
Created on 15/03/2025

@author: Aryan

Filename: gp.py
"""

import os
import cv2
import numpy as np
from pyrr import Matrix44
from PIL import Image

import moderngl
import moderngl_window as mglw
import pywavefront

# If your code depends on custom hand tracking or pose, keep or remove as needed.
import prediction


class CameraAR(mglw.WindowConfig):
    """
    ModernGL application for rendering the camera feed plus a textured 3D crate.
    """
    gl_version = (3, 3)
    title = "CameraAR"
    resource_dir = "data"  # Where crate.obj / crate.png are located

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set unpack alignment to 1
        self.ctx.unpack_alignment = 1
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        # ----------------------------------------------------------------
        # 1) Fullscreen quad for displaying camera feed
        # ----------------------------------------------------------------
        self.prog_rect = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_position;
                in vec2 in_texcoord;

                out vec2 v_texcoord;

                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                    v_texcoord = in_texcoord;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D cam_texture;

                in vec2 v_texcoord;
                out vec4 f_color;

                void main() {
                    // Flip the texture vertically here:
                    vec2 flipped = vec2(v_texcoord.x, 1.0 - v_texcoord.y);
                    f_color = texture(cam_texture, flipped);
                }
            """
        )

        # Make sure sampler is bound to texture unit 0
        self.u_cam_texture = self.prog_rect["cam_texture"]
        self.u_cam_texture.value = 0  # "cam_texture" is texture unit 0

        # A simple quad that fills the screen
        quad_data = np.array([
            # x,    y,    u,    v
            -1.0,  1.0,  0.0,  0.0,
            -1.0, -1.0,  0.0,  1.0,
            1.0,  1.0,  1.0,  0.0,
            1.0, -1.0,  1.0,  1.0,
        ], dtype="f4")

        self.vbo_rect = self.ctx.buffer(quad_data.tobytes())
        self.vao_rect = self.ctx.vertex_array(
            self.prog_rect,
            [(self.vbo_rect, "2f 2f", "in_position", "in_texcoord")],
        )

        # ----------------------------------------------------------------
        # 2) Program for rendering the textured crate
        # ----------------------------------------------------------------
        self.prog_cube = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 Mvp;

                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord;

                out vec3 v_normal;
                out vec2 v_texcoord;

                void main() {
                    gl_Position = Mvp * vec4(in_position, 1.0);
                    v_normal = in_normal;
                    v_texcoord = in_texcoord;
                }
            """,
            fragment_shader="""
                #version 330

                uniform sampler2D Texture;
                in vec3 v_normal;
                in vec2 v_texcoord;
                out vec4 f_color;

                void main() {
                    vec3 base = texture(Texture, v_texcoord).rgb;
                    float light = max(dot(normalize(v_normal), vec3(0.0, 0.0, 1.0)), 0.0);
                    f_color = vec4(base * light, 1.0);
                }
            """
        )
        self.u_mvp = self.prog_cube["Mvp"]

        # ----------------------------------------------------------------
        # 3) Load crate model from crate.obj
        # ----------------------------------------------------------------
        crate_path = os.path.join(self.resource_dir, "crate.obj")
        crate_scene = pywavefront.Wavefront(crate_path, collect_faces=True)

        # crate.obj is assumed to have 3 pos + 3 normal + 2 texcoord = 8 floats per vertex
        vertices = np.array(crate_scene.vertices, dtype="f4")
        indices = np.array(sum(crate_scene.mesh_list[0].faces, []), dtype="i4")

        self.vbo_cube = self.ctx.buffer(vertices.tobytes())
        self.ibo_cube = self.ctx.buffer(indices.tobytes())

        self.vao_cube = self.ctx.vertex_array(
            self.prog_cube,
            [
                (self.vbo_cube, "3f 3f 2f", "in_position", "in_normal", "in_texcoord")
            ],
            self.ibo_cube
        )

        # ----------------------------------------------------------------
        # 4) Load crate texture from crate.png (RGBA)
        # ----------------------------------------------------------------
        crate_tex_path = os.path.join(self.resource_dir, "crate.png")
        img = Image.open(crate_tex_path).convert("RGBA")
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        self.crate_tex = self.ctx.texture(
            (img.width, img.height), 4, data=img.tobytes()
        )
        self.crate_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # ----------------------------------------------------------------
        # 5) Start OpenCV camera
        # ----------------------------------------------------------------
        self.capture = cv2.VideoCapture(0)
        ret, frame = self.capture.read()
        if not ret:
            print("[ERROR] Could not read from camera.")
            exit(1)

        self.cam_h, self.cam_w = frame.shape[:2]

        # ***IMPORTANT***: Use 4 components (RGBA), because Metal drivers
        # are often finicky about 3-channel textures.
        # We'll fill the texture in on_render().
        self.cam_texture = self.ctx.texture(
            (self.cam_w, self.cam_h),
            4,  # RGBA
            dtype="u1"
        )
        self.cam_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # AR logic placeholders
        self.cube_pos = np.array([0, 0, -30], dtype="f4")
        self.is_grabbed = False

    def on_render(self, time, frame_time):
        # Read the current camera frame
        ret, frame = self.capture.read()
        if not ret:
            return

        # Flip horizontally (mirror)
        frame = cv2.flip(frame, 1)

        # Convert BGR -> RGBA
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        # We do not flip vertically here because we do that in the shader
        # If you prefer flipping in Python, uncomment below and remove the flip in the shader:
        # frame_rgba = cv2.flip(frame_rgba, 0)

        # Update the camera texture with the new frame
        self.cam_texture.write(frame_rgba.tobytes())

        # Clear screen
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        # Draw the camera feed (full-screen quad)
        self.cam_texture.use(location=0)  # Bind to texture unit 0
        # self.vao_rect.render(moderngl.TRIANGLE_STRIP)
        self.vao_rect.render(mode=moderngl.TRIANGLE_STRIP)

        # -----------------------------------------------------------
        # Optional: Hand detection & AR logic from your `prediction`
        # -----------------------------------------------------------
        detection_result = prediction.predict(frame_rgba)
        camera_matrix = prediction.get_camera_matrix(self.cam_w, self.cam_h)
        world_landmarks_list = prediction.solvepnp(
            detection_result.model_landmarks_list,
            detection_result.hand_landmarks,
            camera_matrix,
            self.cam_w,
            self.cam_h
        )

        # Very simple pinch detection
        is_pinching = False
        for i, hand2d in enumerate(detection_result.hand_landmarks):
            if prediction.check_pinch_gesture(hand2d):
                is_pinching = True
                if i < len(world_landmarks_list):
                    finger_3d = world_landmarks_list[i][8]
                    dist = np.linalg.norm(finger_3d - self.cube_pos)
                    if dist < 10.0:
                        self.is_grabbed = True
                        break
        if not is_pinching:
            self.is_grabbed = False

        # -----------------------------------------------------------
        # Render the crate with a simple perspective
        # -----------------------------------------------------------
        fov_y = prediction.get_fov_y(camera_matrix, self.cam_h)
        aspect = self.cam_w / self.cam_h

        proj = Matrix44.perspective_projection(fov_y, aspect, 1.0, 1000.0)

        # Rotate the crate around Y axis and place it at self.cube_pos
        model = Matrix44.from_translation(self.cube_pos)
        model *= Matrix44.from_y_rotation(time)

        mvp = proj * model
        self.u_mvp.write(mvp.astype("f4"))

        # Bind and render the crate
        self.crate_tex.use(location=0)
        self.vao_cube.render()

    def close(self):
        self.capture.release()
        super().close()


if __name__ == "__main__":
    """
    Usage:
        python gp.py --mode moderngl
    or just
        python gp.py
    depending on how you have moderngl_window set up.
    """
    mglw.run_window_config(CameraAR)
