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
    resource_dir = os.path.normpath(os.path.join('./data'))
    # print("resource_dir: ", resource_dir)
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
                uniform vec3 ColorBalance;
                uniform float Saturation;
                in vec2 v_text;
                out vec4 f_color;
                
                // Convert RGB to HSV
                vec3 rgb2hsv(vec3 c) {
                    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                    
                    float d = q.x - min(q.w, q.y);
                    float e = 1.0e-10;
                    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
                }
                
                // Convert HSV to RGB
                vec3 hsv2rgb(vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }
                
                void main() {
                    vec4 color = texture(Texture, v_text);
                    
                    // Apply color balance (RGB scaling)
                    color.rgb *= ColorBalance;
                    
                    // Apply saturation adjustment using HSV
                    vec3 hsv = rgb2hsv(color.rgb);
                    hsv.y *= Saturation; // Adjust saturation
                    color.rgb = hsv2rgb(hsv);
                    
                    // Ensure values are in valid range
                    color.rgb = clamp(color.rgb, 0.0, 1.0);
                    
                    f_color = color;
                }
            ''',
        )
        # Add color correction uniforms
        self.color_balance = self.prog_bg['ColorBalance']
        self.saturation = self.prog_bg['Saturation']
        # Default values - correcting for red tint
        # Reduce red, boost blue/green
        # self.color_balance.value = (0.8, 1.1, 1.2)
        self.color_balance.value = (1.0, 1.0, 1.0)
        self.saturation.value = 0.8  # Reduce overall saturation

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

        # Reset camera to default settings
        self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Auto exposure on
        # Auto white balance on
        self.capture.set(cv2.CAP_PROP_AUTO_WB, 1)

        ret, frame = self.capture.read()
        self.aspect_ratio = float(frame.shape[1]) / frame.shape[0]
        self.window_size = (int(720.0 * self.aspect_ratio), 720)

        # ------------------ Setup video texture and background quad ------------------
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create texture with correct format
        self.video_texture = self.ctx.texture(
            (frame.shape[1], frame.shape[0]), 3, data=frame_rgb.tobytes())
        self.video_texture.build_mipmaps()
        self.video_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.video_texture.repeat_x = False
        self.video_texture.repeat_y = False

        # Background quad vertices: positions and texture coordinates
        vertices = np.array([
            -1.0,  1.0,  0.0,  0.0,  # top-left
            -1.0, -1.0,  0.0,  1.0,  # bottom-left
            1.0,  1.0,  1.0,  0.0,  # top-right
            1.0, -1.0,  1.0,  1.0,  # bottom-right
        ], dtype='f4')
        self.vbo_bg = self.ctx.buffer(vertices.tobytes())
        self.vao_bg = self.ctx.simple_vertex_array(
            self.prog_bg, self.vbo_bg, 'in_vert', 'in_tex')

    def on_key_press(self, key, action):
        # Add keyboard controls to adjust color balance and saturation
        if action == self.keys.ACTION_PRESS:
            if key == self.keys.R:
                # Adjust red balance
                r, g, b = self.color_balance.value
                self.color_balance.value = (max(0.1, r - 0.05), g, b)
                # print(f"Color balance: {self.color_balance.value}")
            elif key == self.keys.T:
                r, g, b = self.color_balance.value
                self.color_balance.value = (min(2.0, r + 0.05), g, b)
                # print(f"Color balance: {self.color_balance.value}")
            elif key == self.keys.G:
                # Adjust green balance
                r, g, b = self.color_balance.value
                self.color_balance.value = (r, max(0.1, g - 0.05), b)
                print(f"Color balance: {self.color_balance.value}")
            elif key == self.keys.H:
                r, g, b = self.color_balance.value
                self.color_balance.value = (r, min(2.0, g + 0.05), b)
                # print(f"Color balance: {self.color_balance.value}")
            elif key == self.keys.B:
                # Adjust blue balance
                r, g, b = self.color_balance.value
                self.color_balance.value = (r, g, max(0.1, b - 0.05))
                # print(f"Color balance: {self.color_balance.value}")
            elif key == self.keys.N:
                r, g, b = self.color_balance.value
                self.color_balance.value = (r, g, min(2.0, b + 0.05))
                # print(f"Color balance: {self.color_balance.value}")
            elif key == self.keys.S:
                # Decrease saturation
                self.saturation.value = max(0.0, self.saturation.value - 0.05)


    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0)

        # Calculate FPS
        current_time = time
        fps = 1.0 / frame_time  # Use frame_time directly for FPS calculation

        # Render background video first (without depth test)
        self.ctx.disable(moderngl.DEPTH_TEST)

        ret, frame = self.capture.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)  # Mirror for intuitive interaction

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Hand detection
        detection_result = predict(frame_rgb)

        # Initialize status variables
        is_pinching = False
        is_grabbed = False
        thumb_index_distance = 0.0
        index_cube_distance = 0.0

        # Draw hand landmarks and connections on the frame
        if detection_result is not None and detection_result.hand_landmarks:
            from mediapipe import solutions
            from mediapipe.framework.formats import landmark_pb2

            # Draw landmarks and connections for each detected hand
            for hand_landmarks in detection_result.hand_landmarks:
                # Convert normalized landmarks to landmark proto for drawing
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in hand_landmarks
                ])

                # Draw the connections between landmarks (colored lines)
                solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                    solutions.drawing_styles.get_default_hand_connections_style()
                )

        # Process hand landmarks for 3D positioning
        # Process hand landmarks for 3D positioning
        world_landmarks_list = []
        if detection_result is not None and detection_result.hand_landmarks and detection_result.hand_world_landmarks:
            # Only proceed if we have both 2D and 3D landmarks
            image_landmarks_list = detection_result.hand_landmarks
            model_landmarks_list = detection_result.hand_world_landmarks

            # Make sure the lists have the same length
            if len(image_landmarks_list) == len(model_landmarks_list) and len(image_landmarks_list) > 0:
                cam_matrix = get_camera_matrix(
                    self.window_size[0], self.window_size[1], scale=0.8)
                world_landmarks_list = solvepnp(
                    model_landmarks_list, image_landmarks_list, cam_matrix,
                    self.window_size[0], self.window_size[1])

        # Only render landmarks if we have valid data
        converted_landmarks = []
        if world_landmarks_list:
            for landmarks in world_landmarks_list:
                # Check if landmarks array is not empty and contains valid points
                if landmarks is not None and len(landmarks) > 0:
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
        is_grabbed = False
        pinch_threshold = 5.0  # Easier pinching
        hit_threshold = 7.0    # Easier object selection

        for landmarks in converted_landmarks:
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]

            # Calculate distances
            thumb_index_distance = np.linalg.norm(thumb_tip - index_tip)
            index_cube_distance = np.linalg.norm(index_tip - self.object_pos)
            # print(
            #     f"Thumb-Index distance: {thumb_index_distance:.2f}, Index-Cube distance: {index_cube_distance:.2f}")

            is_pinching = thumb_index_distance < pinch_threshold
            if is_pinching:
                # print("Pinch detected!")
                if index_cube_distance < hit_threshold:
                    # print("Object grabbed!")
                    is_grabbed = True
                    self.object_pos = index_tip

        # Draw status information on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Background for text - semi-transparent black rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # FPS display (green)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                    font, font_scale, (0, 255, 0), thickness)

        # Pinch status (yellow if pinching, white if not)
        pinch_color = (0, 255, 255) if is_pinching else (255, 255, 255)
        pinch_text = "Pinch: YES" if is_pinching else "Pinch: NO"
        cv2.putText(frame, pinch_text, (20, 70), font,
                    font_scale, pinch_color, thickness)

        # Grab status (red if grabbed, white if not)
        grab_color = (0, 0, 255) if is_grabbed else (255, 255, 255)
        grab_text = "Grabbed: YES" if is_grabbed else "Grabbed: NO"
        cv2.putText(frame, grab_text, (20, 100), font,
                    font_scale, grab_color, thickness)

        # Distance information
        cv2.putText(frame, f"Thumb-Index: {thumb_index_distance:.1f}", (20, 130), font,
                    font_scale, (255, 255, 255), thickness)

        # More distance info if available
        if index_cube_distance > 0:
            cv2.putText(frame, f"Index-Cube: {index_cube_distance:.1f}", (20, 160), font,
                        font_scale, (255, 255, 255), thickness)

        # Update video texture with the processed frame including text overlays
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.video_texture.write(frame_rgb.tobytes())
        self.video_texture.use(location=0)
        self.prog_bg['Texture'].value = 0
        self.vao_bg.render(moderngl.TRIANGLE_STRIP)

        # Enable depth test for 3D objects
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        # Render cube
        translate = Matrix44.from_translation(self.object_pos)
        rotate = Matrix44.from_y_rotation(np.sin(time) * 0.5 + 0.2)
        scale = Matrix44.from_scale((3, 3, 3))
        mvp = proj * translate * rotate * scale
        self.mvp.write(mvp.astype('f4'))
        self.color.value = (1.0, 0.0, 0.0) if is_grabbed else (1.0, 1.0, 1.0)
        self.light.value = (10, 10, 10)
        self.withTexture.value = True
        self.texture.use(location=0)
        self.vao_cube.render()


        # Render hand landmarks
        if converted_landmarks:  # Only render if we have valid landmarks
            for landmarks in converted_landmarks:
                for point_idx, point in enumerate(landmarks):
                    # Skip rendering if any coordinates are NaN
                    if np.isnan(point).any():
                        continue

                    # Only render specific landmarks (you can modify which ones)
                    # For example, render only fingertips (indices 4, 8, 12, 16, 20) and some knuckles
                    important_landmarks = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
                    if point_idx not in important_landmarks:
                        continue

                    marker_translate = Matrix44.from_translation(point)
                    marker_scale = Matrix44.from_scale(
                        (0.5, 0.5, 0.5))  # Slightly larger markers
                    mvp_marker = proj * marker_translate * marker_scale
                    self.mvp.write(mvp_marker.astype('f4'))

                    # Color fingertips differently
                    if point_idx in [4, 8, 12, 16, 20]:  # Fingertips
                        # Cyan for fingertips
                        self.color.value = (0.0, 1.0, 1.0)
                    else:
                        # Green for other landmarks
                        self.color.value = (0.0, 1.0, 0.0)

                    self.withTexture.value = False
                    self.vao_marker.render()
            for point in landmarks:
                marker_translate = Matrix44.from_translation(point)
                marker_scale = Matrix44.from_scale((0.25, 0.25, 0.25))
                mvp_marker = proj * marker_translate * marker_scale
                self.mvp.write(mvp_marker.astype('f4'))
                self.color.value = (0.0, 1.0, 0.0)  # Green markers
                self.withTexture.value = False
                self.vao_marker.render()
