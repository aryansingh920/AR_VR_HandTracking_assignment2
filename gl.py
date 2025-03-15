"""
Created on 15/03/2025

@author: Aryan

Filename: gl.py

Relative Path: gl.py
"""


import moderngl
import moderngl_window as mglw
from pyrr import Matrix44
import cv2
import numpy as np
import prediction


class CameraAR(mglw.WindowConfig):
    """
    ModernGL application for rendering the camera feed, a 3D cube, and hand tracking.
    """
    gl_version = (3, 3)
    title = "CameraAR"
    resource_dir = "."  # Update if your resources are stored elsewhere

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        # Setup shader and quad for camera feed
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

        quad_data = np.float32([
            # X, Y, U, V
            -1.0,  1.0,  0.0, 0.0,
            -1.0, -1.0,  0.0, 1.0,
            1.0,  1.0,  1.0, 0.0,
            1.0, -1.0,  1.0, 1.0
        ])
        self.vbo_rect = self.ctx.buffer(quad_data.tobytes())
        self.vao_rect = self.ctx.vertex_array(
            self.prog_rect,
            [(self.vbo_rect, "2f 2f", "in_position", "in_texcoord")]
        )

        # Setup 3D cube shader
        self.prog3d = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 Mvp;
                in vec3 in_position;
                in vec3 in_normal;
                out vec3 v_normal;
                void main() {
                    gl_Position = Mvp * vec4(in_position, 1.0);
                    v_normal = in_normal;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_normal;
                out vec4 f_color;
                void main() {
                    float light = max(dot(normalize(v_normal), vec3(0,0,1)), 0.0);
                    f_color = vec4(0.8, 0.8, 0.0, 1.0) * light;
                }
            """
        )
        self.u_mvp = self.prog3d["Mvp"]

        # Define a simple cube (vertex positions and normals)
        cube_verts = np.float32([
            # x, y, z, nx, ny, nz
            # front
            -5, -5, -5, 0, 0, -1,
            5, -5, -5, 0, 0, -1,
            5,  5, -5, 0, 0, -1,
            -5,  5, -5, 0, 0, -1,
            # back
            -5, -5,  5, 0, 0, 1,
            5, -5,  5, 0, 0, 1,
            5,  5,  5, 0, 0, 1,
            -5,  5,  5, 0, 0, 1,
            # left
            -5, -5, -5, -1, 0, 0,
            -5,  5, -5, -1, 0, 0,
            -5,  5,  5, -1, 0, 0,
            -5, -5,  5, -1, 0, 0,
            # right
            5, -5, -5, 1, 0, 0,
            5,  5, -5, 1, 0, 0,
            5,  5,  5, 1, 0, 0,
            5, -5,  5, 1, 0, 0,
            # top
            -5,  5, -5, 0, 1, 0,
            5,  5, -5, 0, 1, 0,
            5,  5,  5, 0, 1, 0,
            -5,  5,  5, 0, 1, 0,
            # bottom
            -5, -5, -5, 0, -1, 0,
            5, -5, -5, 0, -1, 0,
            5, -5,  5, 0, -1, 0,
            -5, -5,  5, 0, -1, 0,
        ])
        cube_indices = np.int32([
            0,  1,  2,  2,  3,  0,      # front
            4,  5,  6,  6,  7,  4,      # back
            8,  9, 10, 10, 11,  8,      # left
            12, 13, 14, 14, 15, 12,      # right
            16, 17, 18, 18, 19, 16,      # top
            20, 21, 22, 22, 23, 20       # bottom
        ])
        self.vbo_cube = self.ctx.buffer(cube_verts.tobytes())
        self.ibo_cube = self.ctx.buffer(cube_indices.tobytes())
        self.vao_cube = self.ctx.vertex_array(
            self.prog3d,
            [(self.vbo_cube, "3f 3f", "in_position", "in_normal")],
            self.ibo_cube
        )

        # Start OpenCV camera for texture
        self.capture = cv2.VideoCapture(0)
        ret, frame = self.capture.read()
        if not ret:
            print("[ERROR] Could not read from camera.")
            exit(1)
        self.cam_h, self.cam_w = frame.shape[:2]
        self.cam_texture = self.ctx.texture(
            (self.cam_w, self.cam_h), 3, dtype="u1")
        self.cam_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Cube position in camera space (in centimeters)
        self.cube_pos = np.array([0, 0, -30], dtype=np.float32)
        self.is_grabbed = False

    def render(self, time, frame_time):
        ret, frame = self.capture.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.cam_texture.write(frame_rgb.tobytes())

        self.ctx.clear(1.0, 1.0, 1.0)
        self.cam_texture.use(location=0)
        self.vao_rect.render(moderngl.TRIANGLE_STRIP)

        # Hand detection & solvePnP
        detection_result = prediction.predict(frame_rgb)
        camera_matrix = prediction.get_camera_matrix(self.cam_w, self.cam_h)
        world_landmarks_list = prediction.solvepnp(
            detection_result.model_landmarks_list,
            detection_result.hand_landmarks,
            camera_matrix,
            self.cam_w,
            self.cam_h
        )

        # Basic pinch detection for grabbing the cube
        is_pinching = False
        for i, h2d in enumerate(detection_result.hand_landmarks):
            if prediction.check_pinch_gesture(h2d):
                is_pinching = True
                if i < len(world_landmarks_list):
                    finger_3d = world_landmarks_list[i][8]  # index fingertip
                    dist = np.linalg.norm(finger_3d - self.cube_pos)
                    if dist < 10.0:  # threshold in centimeters
                        self.is_grabbed = True
                        break
        if not is_pinching:
            self.is_grabbed = False

        # Render the 3D cube
        fov_y = prediction.get_fov_y(camera_matrix, self.cam_h)
        aspect = self.cam_w / self.cam_h
        proj = Matrix44.perspective_projection(fov_y, aspect, 1.0, 1000.0)
        model = Matrix44.from_translation(
            self.cube_pos) * Matrix44.from_y_rotation(time)
        mvp = proj * model
        self.u_mvp.write(mvp.astype("f4"))
        self.vao_cube.render()

    def close(self):
        self.capture.release()
        super().close()
