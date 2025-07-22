import time


def run_renderer(root_object):
    import tkinter as tk
    import math

    from bereshit import Vector3 as Vector3D
    from bereshit import Object,Servo
    # === Constants ===
    WIDTH, HEIGHT = 1920 , 1080
    FOV = 120
    VIEWER_DISTANCE = 5

    CUBE_VERTICES = [
        (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5),
        (0.5,  0.5, -0.5), (-0.5, 0.5, -0.5),
        (-0.5, -0.5,  0.5), (0.5, -0.5,  0.5),
        (0.5,  0.5,  0.5), (-0.5, 0.5,  0.5),
    ]

    CUBE_EDGES = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7),
    ]

    # === Camera Object ===
    camera = Object(
        position=(0, 10, -20),
        rotation=(0, 0, 0),
        size=(0, 0, 0),
        children=[]
    )

    camera_root = Object(
        position=(0, 10, -20),
        rotation=(0, 0, 0),
        size=(0, 0, 0),
        children=[camera]
    )

    camera_root.set_local_rotation()

    # === Math Utilities ===
    def rotate_point(v, rotation):
        x, y, z = v
        rx, ry, rz = map(math.radians, rotation)

        cos_x, sin_x = math.cos(rx), math.sin(rx)
        y, z = y * cos_x - z * sin_x, y * sin_x + z * cos_x

        cos_y, sin_y = math.cos(ry), math.sin(ry)
        x, z = x * cos_y + z * sin_y, -x * sin_y + z * cos_y

        cos_z, sin_z = math.cos(rz), math.sin(rz)
        x, y = x * cos_z - y * sin_z, x * sin_z + y * cos_z

        return Vector3D(x, y, z)

    def project(world_point, camera_obj, camera_root_obj):
        cam_world_pos = camera_root_obj.local_position + rotate_point(camera_obj.local_position, camera_root_obj.rotation)
        cam_world_rot = Vector3D(
            camera_root_obj.rotation.x + camera_obj.rotation.x,
            camera_root_obj.rotation.y + camera_obj.rotation.y,
            camera_root_obj.rotation.z + camera_obj.rotation.z
        )

        relative = world_point - cam_world_pos
        rx, ry, rz = map(math.radians, (-cam_world_rot.x, -cam_world_rot.y, -cam_world_rot.z))
        x, y, z = relative

        # Rotate around Y
        cos_y, sin_y = math.cos(ry), math.sin(ry)
        x, z = x * cos_y + z * sin_y, -x * sin_y + z * cos_y

        # Rotate around X
        cos_x, sin_x = math.cos(rx), math.sin(rx)
        y, z = y * cos_x - z * sin_x, -y * sin_x + z * cos_x

        # Rotate around Z
        cos_z, sin_z = math.cos(rz), math.sin(rz)
        x, y = x * cos_z + y * sin_z, -x * sin_z + y * cos_z

        # ⛔️ Skip projection if behind camera
        if z <= 0:
            return None  # Not visible


        factor = FOV / (z + VIEWER_DISTANCE)
        x_proj = int(x * factor + WIDTH / 2)
        y_proj = int(-y * factor + HEIGHT / 2)
        return x_proj, y_proj

    def render_object(canvas, obj, global_pos, global_rot, global_scale):
        vertices = []
        for corner in CUBE_VERTICES:
            local = Vector3D(*corner)
            scaled = local * global_scale
            rotated = rotate_point(scaled, global_rot)
            world = global_pos + rotated
            screen = project(world, camera, camera_root)
            vertices.append(screen)

        for i, j in CUBE_EDGES:
            if vertices[i] is not None and vertices[j] is not None:
                 canvas.create_line(*vertices[i], *vertices[j], fill="black")

    def render_hierarchy(canvas, servo, parent_pos, parent_rot):
        if isinstance(servo, Servo):
            obj = servo.obj
        else:
            obj = servo
        local_pos = rotate_point(obj.local_position, parent_rot)
        global_pos = parent_pos + local_pos
        global_rot = tuple(p + r for p, r in zip(parent_rot, obj.local_rotation))
        global_scale = obj.size

        render_object(canvas, obj, global_pos, global_rot, global_scale)

        for child in obj.children:
            render_hierarchy(canvas, child, global_pos, global_rot)

    # === Controls ===
    pressed_keys = set()

    def key_down(event):
        key = event.keysym.lower()
        pressed_keys.add(key)

    def key_up(event):
        key = event.keysym.lower()
        pressed_keys.discard(key)

    def update_camera():
        # root_object.get_children_objects()[0].gravity(collidable_objects=[root_object.get_children_objects()[1]])
        # root_object.get_children_objects()[1].gravity()

        speed = 0.8
        rot_speed = 2
        root_pos = camera_root.local_position
        cam_rot = camera.rotation

        yaw = math.radians(cam_rot.y)
        forward = Vector3D(math.sin(yaw), 0, math.cos(yaw))
        right = Vector3D(math.cos(yaw), 0, -math.sin(yaw))
        up = Vector3D(0, 1, 0)

        # Move camera root
        move_vec = Vector3D(0, 0, 0)
        if 'w' in pressed_keys: move_vec += forward
        if 's' in pressed_keys: move_vec -= forward
        if 'a' in pressed_keys: move_vec -= right
        if 'd' in pressed_keys: move_vec += right
        if 'space' in pressed_keys: move_vec += up
        if 'shift_l' in pressed_keys: move_vec -= up
        camera_root.set_position(root_pos + move_vec * speed)

        # Rotate camera
        if 'left' in pressed_keys: cam_rot.y -= rot_speed
        if 'right' in pressed_keys: cam_rot.y += rot_speed
        if 'up' in pressed_keys: cam_rot.x -= rot_speed
        if 'down' in pressed_keys: cam_rot.x += rot_speed
        camera.set_rotation(cam_rot)

    # === Mouse Drag Rotation ===
    is_rotating = False
    last_mouse_pos = None

    def mouse_down(event):
        nonlocal is_rotating, last_mouse_pos
        if event.num == 2:
            is_rotating = True
            last_mouse_pos = (event.x, event.y)

    def mouse_up(event):
        nonlocal is_rotating
        if event.num == 2:
            is_rotating = False

    def mouse_motion(event):
        nonlocal last_mouse_pos
        if is_rotating and last_mouse_pos:
            dx = event.x - last_mouse_pos[0]
            dy = event.y - last_mouse_pos[1]
            sensitivity = 0.3
            rot = camera.rotation
            new_rot = Vector3D(rot.x + dy * sensitivity, rot.y + dx * sensitivity, rot.z)
            camera.set_rotation(new_rot)
            last_mouse_pos = (event.x, event.y)

    # === GUI Setup ===
    win = tk.Tk()
    win.title("3D Object Renderer")
    canvas = tk.Canvas(win, width=WIDTH, height=HEIGHT, bg="white")
    canvas.pack()

    canvas.focus_set()  # Ensures canvas gets keyboard focus

    canvas.bind("<KeyPress>", key_down)
    canvas.bind("<KeyRelease>", key_up)
    canvas.bind("<ButtonPress-2>", mouse_down)
    canvas.bind("<ButtonRelease-2>", mouse_up)
    canvas.bind("<B2-Motion>", mouse_motion)

    def render_loop():
        canvas.delete("all")
        update_camera()
        render_hierarchy(canvas, root_object, Vector3D(0, 0, 0), (0, 0, 0))
        win.after(100, render_loop)

    render_loop()
    win.mainloop()