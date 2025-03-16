# """
# Created on 15/03/2025

# @author: Aryan

# Filename: gp.py
# """

# import os
# import cv2
# import numpy as np
# from pyrr import Matrix44
# from PIL import Image

# import moderngl
# import moderngl_window as mglw
# import pywavefront

# # If your code depends on custom hand tracking or pose, keep or remove as needed.
# import prediction


# class CameraAR(mglw.WindowConfig):
#     """
#     Simplified version of CameraAR that focuses just on rendering the camera feed.
#     """
#     gl_version = (3, 3)
#     title = "CameraAR Debug"

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.ctx.unpack_alignment = 1

#         print("[DEBUG] Creating simpler renderer")

#         # Very simple fragment shader that renders a solid color first to verify rendering works
#         self.prog_simple = self.ctx.program(
#             vertex_shader="""
#                 #version 330
#                 in vec2 in_position;
#                 void main() {
#                     gl_Position = vec4(in_position, 0.0, 1.0);
#                 }
#             """,
#             fragment_shader="""
#                 #version 330
#                 out vec4 f_color;
#                 void main() {
#                     f_color = vec4(0.0, 0.5, 1.0, 1.0); // Solid blue color
#                 }
#             """
#         )

#         # Create a quad that fills the screen
#         quad_data = np.array([
#             # x,    y
#             -1.0,  1.0,
#             -1.0, -1.0,
#             1.0,  1.0,
#             1.0, -1.0,
#         ], dtype="f4")

#         self.vbo_simple = self.ctx.buffer(quad_data.tobytes())
#         self.vao_simple = self.ctx.vertex_array(
#             self.prog_simple,
#             [(self.vbo_simple, "2f", "in_position")],
#         )

#         # Start camera - explicit error handling
#         print("[DEBUG] Starting camera")
#         self.capture = cv2.VideoCapture(0)
#         if not self.capture.isOpened():
#             print("[ERROR] Could not open camera.")
#             exit(1)

#         ret, frame = self.capture.read()
#         if not ret:
#             print("[ERROR] Could not read from camera.")
#             exit(1)

#         self.cam_h, self.cam_w = frame.shape[:2]
#         print(f"[DEBUG] Camera initialized: {self.cam_w}x{self.cam_h}")

#         # Now create textures and actual rendering program
#         self.setup_texture_renderer()

#     def setup_texture_renderer(self):
#         """Set up the actual camera texture rendering after basics work"""
#         print("[DEBUG] Setting up texture renderer")

#         # Create basic camera texture program with fragment shader
#         self.prog_camera = self.ctx.program(
#             vertex_shader="""
#                 #version 330
#                 in vec2 in_position;
#                 in vec2 in_texcoord;
#                 out vec2 v_texcoord;
#                 void main() {
#                     gl_Position = vec4(in_position, 0.0, 1.0);
#                     v_texcoord = in_texcoord;
#                 }
#             """,
#             fragment_shader="""
#                 #version 330
#                 uniform sampler2D tex_image;
#                 in vec2 v_texcoord;
#                 out vec4 f_color;
#                 void main() {
#                     f_color = texture(tex_image, v_texcoord);
#                 }
#             """
#         )

#         # Set texture uniform
#         self.prog_camera["tex_image"].value = 0

#         # Create quad with INVERTED texture coordinates to correct the orientation
#         vertices = np.array([
#             # x,    y,    u,    v
#             -1.0,  1.0,  0.0,  0.0,  # top-left - CORRECTED V
#             -1.0, -1.0,  0.0,  1.0,  # bottom-left - CORRECTED V
#             1.0,  1.0,  1.0,  0.0,  # top-right - CORRECTED V
#             1.0, -1.0,  1.0,  1.0,  # bottom-right - CORRECTED V
#         ], dtype="f4")

#         self.vbo_camera = self.ctx.buffer(vertices.tobytes())
#         self.vao_camera = self.ctx.vertex_array(
#             self.prog_camera,
#             [(self.vbo_camera, "2f 2f", "in_position", "in_texcoord")],
#         )

#         # Create the texture with format matching our data
#         print(f"[DEBUG] Creating texture {self.cam_w}x{self.cam_h}")
#         self.texture = self.ctx.texture(
#             (self.cam_w, self.cam_h), 4, dtype="u1")  # u1 = unsigned byte format
#         self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

#         # For debugging, create a test pattern texture using 8-bit values
#         pattern = np.zeros((self.cam_h, self.cam_w, 4), dtype=np.uint8)
#         # Create a checkboard pattern
#         for y in range(self.cam_h):
#             for x in range(self.cam_w):
#                 cell_size = 64
#                 checker = ((x // cell_size) % 2) == ((y // cell_size) % 2)
#                 pattern[y, x] = [255, 255, 255, 255] if checker else [
#                     128, 128, 128, 255]

#         # Update the texture with test pattern
#         print("[DEBUG] Updating texture with test pattern")
#         self.texture.write(pattern.tobytes())

#         self.render_mode = 0  # 0=blue quad, 1=test pattern, 2=camera
#         print("[DEBUG] Setup complete")

#     def on_render(self, time, frame_time):
#         self.ctx.clear(0.0, 0.0, 0.0, 1.0)  # Black background

#         # Start with simple blue quad rendering to verify setup
#         if self.render_mode == 0:
#             self.vao_simple.render(moderngl.TRIANGLE_STRIP)
#             # Switch to test pattern on next frame
#             self.render_mode = 1
#             return

#         # Test pattern rendering to verify texture setup
#         if self.render_mode == 1:
#             self.texture.use(location=0)
#             self.vao_camera.render(moderngl.TRIANGLE_STRIP)
#             # Switch to camera on next frame
#             self.render_mode = 2
#             return

#         # Camera rendering - now we try to use the actual camera
#         try:
#             ret, frame = self.capture.read()
#             if not ret:
#                 print("[ERROR] Failed to read camera frame")
#                 return

#             # Only flip horizontally for mirror effect
#             frame = cv2.flip(frame, 1)

#             # Convert to RGBA explicitly - keep as uint8
#             frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

#             # Important: Make sure memory is contiguous
#             frame_rgba = np.ascontiguousarray(frame_rgba)

#             # Update texture with the new frame - directly use uint8 data
#             self.texture.write(frame_rgba.tobytes())

#             # Render the camera feed
#             self.texture.use(location=0)
#             self.vao_camera.render(moderngl.TRIANGLE_STRIP)

#         except Exception as e:
#             print(f"[ERROR] Camera rendering failed: {e}")
#             import traceback
#             traceback.print_exc()

#     def close(self):
#         if hasattr(self, 'capture'):
#             self.capture.release()
#         super().close()


# if __name__ == "__main__":
#     """
#     Usage:
#         python gp.py --mode moderngl
#     or just
#         python gp.py
#     depending on how you have moderngl_window set up.
#     """
#     mglw.run_window_config(CameraAR)
