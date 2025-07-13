import tkinter as tk

import mouse

from bereshit import Vector3
from bereshit import rotate_vector_old
import tkinter as tk
import pyautogui
screen_width, screen_height =0,0

def run_renderer(root_object):
    global screen_width, screen_height
    camera_obj = root_object.search_by_component("camera")
    if not camera_obj:
        print("No camera found.")
        return

    cam = camera_obj.camera
    screen_width, screen_height = pyautogui.size()
    center_x = screen_width // 2
    center_y = screen_height // 2
    mouse.move(center_x, center_y)
    WIDTH, HEIGHT = cam.width, cam.hight
    FOV = cam.FOV
    VIEWER_DISTANCE = cam.VIEWER_DISTANCE

    window = tk.Tk()
    canvas = tk.Canvas(window, width=screen_width, height=screen_height, bg="black")
    canvas.pack()

    def project(world_point):
        # Camera transform
        relative = world_point - camera_obj.position
        rotated = rotate_vector_old(relative, Vector3(0,0,0), camera_obj.rotation)


        if (VIEWER_DISTANCE + rotated.z) == 0:
            factor = 0
        else:
            factor = FOV / (VIEWER_DISTANCE + rotated.z)
        x = rotated.x * factor + WIDTH / 2
        y = -rotated.y * factor + HEIGHT / 2
        return x, y

    def draw_scene():
        canvas.delete("all")

        for obj in [root_object] + root_object.get_all_children_bereshit():
            # Compute position relative to camera
            center_relative = obj.position - camera_obj.position
            center_rotated = rotate_vector_old(center_relative, Vector3(0, 0, 0), camera_obj.rotation)

            if center_rotated.z <= 0:
                # Behind camera—skip
                continue

            # Compute corners
            size = obj.size * 0.5
            corners = [
                Vector3(x, y, z)
                for x in (-size.x, size.x)
                for y in (-size.y, size.y)
                for z in (-size.z, size.z)
            ]
            world_corners = [rotate_vector_old(c, Vector3(0, 0, 0), obj.rotation) + obj.position for c in corners]

            # Project all corners
            projected = []
            for c in world_corners:
                rel = c - camera_obj.position
                rot = rotate_vector_old(rel, Vector3(0, 0, 0), camera_obj.rotation)
                if rot.z <= 0:
                    # For edges crossing behind camera, just clamp to avoid crash.
                    # You can improve this with proper near-plane clipping.
                    rot.z = 0.001
                factor = FOV / (VIEWER_DISTANCE + rot.z)
                x = rot.x * factor + WIDTH / 2
                y = -rot.y * factor + HEIGHT / 2
                projected.append((x, y))

            edges = [
                (0, 1), (1, 3), (3, 2), (2, 0),
                (4, 5), (5, 7), (7, 6), (6, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]
            for i, j in edges:
                x1, y1 = projected[i]
                x2, y2 = projected[j]
                canvas.create_line(x1, y1, x2, y2, fill=obj.material.color)

        canvas.after(33, draw_scene)

    draw_scene()
    window.mainloop()

