import moderngl
import moderngl_window
from moderngl_window import geometry
from pyrr import Matrix44, Quaternion, Vector3 as PyrrVector3
import numpy as np
from bereshit import Vector3, rotate_vector_old

class BereshitRenderer(moderngl_window.WindowConfig):
    gl_version = (3, 3)
    title = "Bereshit moderngl Renderer"
    window_size = (1280, 720)
    aspect_ratio = None
    resizable = True
    resource_dir = '.'
    root_object = None  # 👈 class variable to inject data

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root_object = BereshitRenderer.root_object  # 👈 assign it here
        self.camera_obj = self.root_object.search_by_component("camera")
        if not self.camera_obj:
            raise Exception("No camera object found")

        cam = self.camera_obj.camera
        self.fov = cam.FOV
        self.viewer_distance = cam.VIEWER_DISTANCE

        self.prog = self.ctx.program(
            vertex_shader='''
            #version 330
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            in vec3 in_position;
            void main() {
                gl_Position = projection * view * model * vec4(in_position, 1.0);
            }
            ''',
            fragment_shader='''
            #version 330
            out vec4 f_color;
            uniform vec3 color;
            void main() {
                f_color = vec4(color, 1.0);
            }
            '''
        )

        self.view = Matrix44.identity()
        self.projection = Matrix44.perspective_projection(self.fov, self.wnd.aspect_ratio, 0.1, 1000.0)

        self.meshes = []
        self.prepare_meshes()

    def prepare_meshes(self):
        for obj in [self.root_object] + self.root_object.get_all_children_bereshit():
            if obj.mesh is None:
                continue

            # Convert vertices to numpy
            verts = [(v * obj.size * 0.5).to_np() for v in
                     obj.mesh.vertices]  # Ensure this returns list or np.array of floats

            lines = []
            for i, j in obj.mesh.edges:
                lines.extend(verts[i])  # 👈 flatten the vector into x, y, z
                lines.extend(verts[j])

            vbo = np.array(lines, dtype='f4')
            vao = self.ctx.buffer(vbo.tobytes())
            self.meshes.append({
                'obj': obj,
                'vbo': vao,
                'vao': self.ctx.vertex_array(
                    self.prog,
                    [(vao, '3f', 'in_position')],
                ),
                'len': len(lines),
            })

    def on_render(self, time: float, frametime: float):
        self.ctx.clear(0.0, 0.0, 0.0)

        # --- Camera ---
        cam_pos = self.camera_obj.position.to_np()
        cam_rot = self.camera_obj.quaternion  # Assume this is your Quaternion class

        # Convert to pyrr.Quaternion: pyrr expects [w, x, y, z]
        q_cam = Quaternion([cam_rot.w, cam_rot.x, cam_rot.y, cam_rot.z])
        forward = q_cam * PyrrVector3([0, 0, 1])  # Rotate forward vector using quaternion
        target = cam_pos + forward

        self.view = Matrix44.look_at(
            PyrrVector3(cam_pos),
            PyrrVector3(target),
            PyrrVector3([0, 1, 0])
        )

        # --- Render each mesh ---
        for item in self.meshes:
            obj = item['obj']
            pos = obj.position.to_np()
            rot = obj.quaternion  # Your Quaternion
            size = obj.size.to_np()

            trans = Matrix44.from_translation(pos)
            scale = Matrix44.from_scale(size * 0.5)

            # Convert your Quaternion to pyrr
            q_obj = Quaternion([rot.w, rot.x, rot.y, rot.z])
            quaternion = q_obj.matrix44

            model = trans @ quaternion @ scale

            self.prog['model'].write(model.astype('f4').tobytes())
            self.prog['view'].write(self.view.astype('f4').tobytes())
            self.prog['projection'].write(self.projection.astype('f4').tobytes())

            color = obj.material.color if hasattr(obj.material, 'color') else (1.0, 1.0, 1.0)
            self.prog['color'].value = color

            item['vao'].render(mode=moderngl.LINES, vertices=item['len'])

def run_renderer(root_object):
    BereshitRenderer.root_object = root_object  # 👈 inject your object here
    moderngl_window.run_window_config(BereshitRenderer, args=['--window', 'glfw'])

