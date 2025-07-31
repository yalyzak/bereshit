import moderngl
import moderngl_window
from moderngl_window import geometry
from pyrr import Vector4, Vector3 as PyrrVector3, Quaternion as PyrrQuat, Matrix44
import numpy as np
from bereshit import Vector3, rotate_vector_old



class BereshitRenderer(moderngl_window.WindowConfig):
    gl_version = (3, 3)
    title = "Bereshit moderngl Renderer"
    window_size = (1920, 1080)
    aspect_ratio = None
    resizable = True
    resource_dir = '.'
    root_object = None  # 👈 class variable to inject data

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root_object = BereshitRenderer.root_object  # 👈 assign it here
        self.camera_obj = self.root_object.search_by_component('Camera')
        if not self.camera_obj:
            raise Exception("No camera object found")

        self.cam = self.camera_obj.Camera
        self.fov = self.cam.FOV
        self.viewer_distance = self.cam.VIEWER_DISTANCE

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
        shading = self.cam.shading
        if shading == "wire":
            for obj in [self.root_object] + self.root_object.get_all_children_bereshit():
                if obj.Mesh is None or obj.Mesh.vertices == []:
                    continue

                # Convert vertices to numpy
                verts = [(v * obj.size * 0.5).to_np() for v in
                         obj.Mesh.vertices]  # Ensure this returns list or np.array of floats

                lines = []
                for i, j in obj.Mesh.edges:
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
        if shading == "solid":
            for obj in [self.root_object] + self.root_object.get_all_children_bereshit():
                if obj.Mesh is None:
                    continue

                # Convert vertices to numpy (scaled and centered)
                verts = [(v * obj.size * 0.5).to_np() for v in obj.Mesh.vertices]

                # Build triangle vertex list
                triangles = []
                for tri in obj.Mesh.triangles:  # tri = (i, j, k)
                    for index in tri:
                        triangles.extend(verts[index])  # flatten x, y, z into list

                vbo = np.array(triangles, dtype='f4')
                vao_buffer = self.ctx.buffer(vbo.tobytes())

                self.meshes.append({
                    'obj': obj,
                    'vbo': vao_buffer,
                    'vao': self.ctx.vertex_array(
                        self.prog,
                        [(vao_buffer, '3f', 'in_position')],
                    ),
                    'len': len(triangles),
                })
    def on_render(self, time: float, frametime: float):
        shading = self.cam.shading

        self.ctx.clear(0.0, 0.0, 0.0)

        cam_pos = self.camera_obj.position.to_np()
        cam_rot = self.camera_obj.quaternion
        # Rotate forward vector (0, 0, 1) using the rotation matrix
        pyrr_q = PyrrQuat([cam_rot.x, cam_rot.y, cam_rot.z, cam_rot.w])
        rot_matrix = Matrix44.from_quaternion(pyrr_q)

        forward_vec4 = np.array([0.0, 0.0, 1.0, 0.0])  # direction vector (w=0)
        rotated_forward = rot_matrix @ forward_vec4
        forward = rotated_forward[:3]

        target = cam_pos + forward

        up_vec4 = np.array([0.0, 1.0, 0.0, 0.0])
        rotated_up = rot_matrix @ up_vec4
        up = rotated_up[:3]

        self.view = Matrix44.look_at(
            PyrrVector3(cam_pos.tolist()),
            PyrrVector3(target.tolist()),
            PyrrVector3(up.tolist())  # or [0,1,0] if not rotated
        )

        for item in self.meshes:
            obj = item['obj']
            pos = obj.position.to_np()
            size = obj.size.to_np()
            rot = obj.quaternion

            # Use object's rotation
            pyrr_obj_q = PyrrQuat([rot.x, rot.y, rot.z, rot.w])
            obj_rot_matrix = Matrix44.from_quaternion(pyrr_obj_q)

            model = (

                    Matrix44.from_quaternion(PyrrQuat([rot.x, rot.y, rot.z, rot.w]))
                    @ Matrix44.from_translation(pos)
            )
            # @ Matrix44.from_scale(size)
            self.prog['model'].write(model.astype('f4').tobytes())
            self.prog['view'].write(self.view.astype('f4').tobytes())
            self.prog['projection'].write(self.projection.astype('f4').tobytes())

            color = obj.material.color if hasattr(obj.material, 'color') else (1.0, 1.0, 1.0)
            self.prog['color'].value = color
            if shading == "wire":
                item['vao'].render(mode=moderngl.LINES, vertices=item['len'])
            elif shading == "solid":
                item['vao'].render(mode=moderngl.TRIANGLES, vertices=item['len'] // 3)


def run_renderer(root_object):
    BereshitRenderer.root_object = root_object  # 👈 inject your object here
    moderngl_window.run_window_config(BereshitRenderer, args=['--window', 'glfw'])

