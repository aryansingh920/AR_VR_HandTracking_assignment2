"""
Created on 16/03/2025

@author: Aryan

Filename: gl.py

Relative Path: assignment2/gl.py
"""

import moderngl
import moderngl_window as mglw
from pyrr import Matrix44
import cv2
import numpy as np
import os
from array import array

from prediction import predict, get_camera_matrix, get_fov_y, solvepnp


class CameraAR(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "CameraAR"
    resource_dir = os.path.normpath(os.path.join( './data'))
    print("resource_dir: ", resource_dir)
    previousTime = 0
    currentTime = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ------------------ Setup shader for background video ------------------
        self.prog_bg = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                in vec2 in_tex;
                out vec2 v_text;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_text = in_tex;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D Texture;
                in vec2 v_text;
                out vec4 f_color;
                void main() {
                    f_color = texture(Texture, v_text);
                }
            ''',
        )

        # ------------------ Setup shader for 3D objects (cube and markers) ------------------
        self.prog3d = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord_0;
                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text;
                void main() {
                    gl_Position = Mvp * vec4(in_position, 1.0);
                    v_vert = in_position;
                    v_norm = in_normal;
                    v_text = in_texcoord_0;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec3 Color;
                uniform vec3 Light;
                uniform sampler2D Texture;
                uniform bool withTexture;
                in vec3 v_vert;
                in vec3 v_norm;
                in vec2 v_text;
                out vec4 f_color;
                void main() {
                    float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
                    if (withTexture) {
                        f_color = vec4(Color * texture(Texture, v_text).rgb * lum, 1.0);
                    } else {
                        f_color = vec4(Color * lum, 1.0);
                    }
                }
            ''',
        )
        self.mvp = self.prog3d['Mvp']
        self.light = self.prog3d['Light']
        self.color = self.prog3d['Color']
        self.withTexture = self.prog3d['withTexture']

        # ------------------ Load 3D objects ------------------
        self.scene_cube = self.load_scene('crate.obj')
        self.scene_marker = self.load_scene('marker.obj')
        self.vao_cube = self.scene_cube.root_nodes[0].mesh.vao.instance(
            self.prog3d)
        self.vao_marker = self.scene_marker.root_nodes[0].mesh.vao.instance(
            self.prog3d)
        self.texture = self.load_texture_2d('crate.png')

        # Initial position of the virtual cube (30 cm in front of camera)
        self.object_pos = np.array([0.0, 0.0, -30.0])

        # ------------------ Start OpenCV camera ------------------
        self.capture = cv2.VideoCapture(0)
        ret, frame = self.capture.read()
        self.aspect_ratio = float(frame.shape[1]) / frame.shape[0]
        self.window_size = (int(720.0 * self.aspect_ratio), 720)

        # ------------------ Setup video texture and background quad ------------------
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.video_texture = self.ctx.texture(
            (frame.shape[1], frame.shape[0]), 3, data=frame_rgb.tobytes())
        self.video_texture.build_mipmaps()
        self.video_texture.repeat_x = False
        self.video_texture.repeat_y = False

        # Background quad vertices: positions and texture coordinates
        vertices = np.array([
             -1.0,  1.0,  0.0,  0.0,  # top-left - CORRECTED V
             -1.0, -1.0,  0.0,  1.0,  # bottom-left - CORRECTED V
            1.0,  1.0,  1.0,  0.0,  # top-right - CORRECTED V
            1.0, -1.0,  1.0,  1.0,  # bottom-right - CORRECTED V
        ], dtype='f4')
        self.vbo_bg = self.ctx.buffer(vertices.tobytes())
        self.vao_bg = self.ctx.simple_vertex_array(
            self.prog_bg, self.vbo_bg, 'in_vert', 'in_tex')


    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0)

        # Render background video first (without depth test)
        self.ctx.disable(moderngl.DEPTH_TEST)

        ret, frame = self.capture.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)  # Mirror for intuitive interaction
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update video texture
        self.video_texture.write(frame_rgb.tobytes())
        self.video_texture.use(location=0)
        self.prog_bg['Texture'].value = 0
        self.vao_bg.render(moderngl.TRIANGLE_STRIP)

        # Enable depth test for 3D objects
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        # Hand detection
        detection_result = predict(frame_rgb)
        world_landmarks_list = []
        if detection_result is not None and detection_result.hand_landmarks:
            print(f"Detected {len(detection_result.hand_landmarks)} hands")
            image_landmarks_list = detection_result.hand_landmarks
            model_landmarks_list = detection_result.hand_world_landmarks
            cam_matrix = get_camera_matrix(
                self.window_size[0], self.window_size[1], scale=0.8)
            world_landmarks_list = solvepnp(
                model_landmarks_list, image_landmarks_list, cam_matrix,
                self.window_size[0], self.window_size[1])

        # Convert landmarks to OpenGL coordinates
        converted_landmarks = []
        for landmarks in world_landmarks_list:
            landmarks_gl = landmarks.copy()
            landmarks_gl[:, 1] = -landmarks_gl[:, 1]  # Flip Y
            landmarks_gl[:, 2] = -landmarks_gl[:, 2]  # Flip Z
            converted_landmarks.append(landmarks_gl)

        # Setup projection matrix
        fov_y = 45.0  # Default if no camera matrix
        if 'cam_matrix' in locals():
            fov_y = get_fov_y(cam_matrix, self.window_size[1])

        proj = Matrix44.perspective_projection(
            fov_y, self.aspect_ratio, 0.1, 1000)

        # Gesture recognition and cube interaction
        grabbed = False
        pinch_threshold = 5.0  # Easier pinching
        hit_threshold = 7.0    # Easier object selection

        for landmarks in converted_landmarks:
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]

            # Debug distances
            dist_thumb_index = np.linalg.norm(thumb_tip - index_tip)
            dist_index_cube = np.linalg.norm(index_tip - self.object_pos)
            print(
                f"Thumb-Index distance: {dist_thumb_index:.2f}, Index-Cube distance: {dist_index_cube:.2f}")

            if dist_thumb_index < pinch_threshold:
                print("Pinch detected!")
                if dist_index_cube < hit_threshold:
                    print("Object grabbed!")
                    grabbed = True
                    self.object_pos = index_tip

        # Render cube
        translate = Matrix44.from_translation(self.object_pos)
        rotate = Matrix44.from_y_rotation(np.sin(time) * 0.5 + 0.2)
        scale = Matrix44.from_scale((3, 3, 3))
        mvp = proj * translate * rotate * scale
        self.mvp.write(mvp.astype('f4'))
        self.color.value = (1.0, 0.0, 0.0) if grabbed else (1.0, 1.0, 1.0)
        self.light.value = (10, 10, 10)
        self.withTexture.value = True
        self.texture.use(location=0)
        self.vao_cube.render()

        # Render hand landmarks
        for landmarks in converted_landmarks:
            for point in landmarks:
                marker_translate = Matrix44.from_translation(point)
                marker_scale = Matrix44.from_scale((0.5, 0.5, 0.5))
                mvp_marker = proj * marker_translate * marker_scale
                self.mvp.write(mvp_marker.astype('f4'))
                self.color.value = (0.0, 1.0, 0.0)  # Green markers
                self.withTexture.value = False
                self.vao_marker.render()


if __name__ == '__main__':
    CameraAR.run()
