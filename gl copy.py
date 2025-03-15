"""
Created on 15/03/2025

@author: Aryan

Filename: gl.py

Relative Path: gl.py
"""

import os
import moderngl
import moderngl_window as mglw
from pyrr import Matrix44
import cv2
import numpy as np
import pywavefront
from PIL import Image
import prediction


class CameraAR(mglw.WindowConfig):
    """
    ModernGL application for rendering the camera feed plus a textured 3D crate.
    Uses crate.obj and crate.png from the data folder.
    """
    gl_version = (3, 3)
    title = "CameraAR"
    resource_dir = "data"  # crate.obj and crate.png live in the data folder

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set unpack alignment to 1 (can help with texture uploads on Metal)
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
                    f_color = texture(cam_texture, v_texcoord);
                }
            """
        )
        self.u_cam_texture = self.prog_rect["cam_texture"]

        quad_data = np.array([
            # x,    y,    u,   v
            -1.0,  1.0,  0.0, 0.0,
            -1.0, -1.0,  0.0, 1.0,
            1.0,  1.0,  1.0, 0.0,
            1.0, -1.0,  1.0, 1.0
        ], dtype="f4")
        self.vbo_rect = self.ctx.buffer(quad_data.tobytes())
        self.vao_rect = self.ctx.vertex_array(
            self.prog_rect,
            [(self.vbo_rect, "2f 2f", "in_position", "in_texcoord")]
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
                    float light = max(dot(normalize(v_normal), vec3(0,0,1)), 0.0);
                    f_color = vec4(base * light, 1.0);
                }
            """
        )
        self.u_mvp = self.prog_cube["Mvp"]

        # ----------------------------------------------------------------
        # 3) Load crate model from `crate.obj`
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
            [(self.vbo_cube, "3f 3f 2f", "in_position", "in_normal", "in_texcoord")],
            self.ibo_cube
        )

        # ----------------------------------------------------------------
        # 4) Load crate texture from `crate.png`
        # ----------------------------------------------------------------
        crate_tex_path = os.path.join(self.resource_dir, "crate.png")
        # Convert image to RGBA (4 channels) so Metal accepts it
        img = Image.open(crate_tex_path).convert("RGBA")
        # Flip vertically for OpenGL
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # Create texture with 4 channels and do not build mipmaps (to avoid issues)
        self.crate_tex = self.ctx.texture(
            (img.width, img.height), components=4, data=img.tobytes())
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
        # Use a 3-channel texture for the camera feed
        self.cam_texture = self.ctx.texture(
            (self.cam_w, self.cam_h), components=3, dtype="u1")
        self.cam_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.size = (self.cam_w, self.cam_h)

        self.cube_pos = np.array([0, 0, -30], dtype="f4")
        self.is_grabbed = False

    def on_render(self, time, frame_time):
        ret, frame = self.capture.read()
        if not ret:
            return

        # print("[DEBUG] ret:", ret, "frame shape:", frame.shape)
        frame = cv2.flip(frame, 1)
        # Convert BGR -> RGB (3 channels)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.cam_texture.write(frame_rgb.tobytes())

        # Clear and draw camera feed as background (green clear to see if our quad draws)
        self.ctx.clear(0.0, 1.0, 0.0, 1.0)
        self.cam_texture.use(location=0)
        self.vao_rect.render(moderngl.TRIANGLE_STRIP)

        # -- Hand detection and solvePnP --
        detection_result = prediction.predict(frame_rgb)
        camera_matrix = prediction.get_camera_matrix(self.cam_w, self.cam_h)
        world_landmarks_list = prediction.solvepnp(
            detection_result.model_landmarks_list,
            detection_result.hand_landmarks,
            camera_matrix,
            self.cam_w,
            self.cam_h
        )

        # Simple pinch detection (not affecting rendering)
        is_pinching = False
        for i, h2d in enumerate(detection_result.hand_landmarks):
            if prediction.check_pinch_gesture(h2d):
                is_pinching = True
                if i < len(world_landmarks_list):
                    finger_3d = world_landmarks_list[i][8]
                    dist = np.linalg.norm(finger_3d - self.cube_pos)
                    if dist < 10.0:
                        self.is_grabbed = True
                        break
        if not is_pinching:
            self.is_grabbed = False

        # -- Render the crate with camera perspective --
        fov_y = prediction.get_fov_y(camera_matrix, self.cam_h)
        aspect = self.cam_w / self.cam_h
        proj = Matrix44.perspective_projection(fov_y, aspect, 1.0, 1000.0)

        model = Matrix44.from_translation(self.cube_pos)
        model *= Matrix44.from_y_rotation(time)
        mvp = proj * model
        self.u_mvp.write(mvp.astype("f4"))

        self.crate_tex.use(location=0)
        self.vao_cube.render()

    def close(self):
        self.capture.release()
        super().close()
