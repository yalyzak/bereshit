import moderngl
import moderngl_window as mglw
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET


import json
import moderngl
import moderngl_window as mglw
import numpy as np
from PIL import Image


import json
import moderngl
import moderngl_window as mglw
import numpy as np
from PIL import Image


class SDFTextRenderer(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "MSDF Text Renderer"
    resource_dir = "."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.window_size = kwargs.get("window_size", (800, 600))

        # ---- Load MSDF JSON ----
        self.glyphs, self.metrics, self.atlas = self.load_msdf_json("bereshit/shaders/fonts/atlas.json")

        # ---- Load MSDF Atlas Texture ----
        img = Image.open("bereshit/shaders/fonts/atlas.png").convert("RGBA")
        self.font_tex = self.ctx.texture(img.size, 4, img.tobytes())
        self.font_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.font_tex.use(0)

        # ---- Shader ----
        self.prog = self.ctx.program(
            vertex_shader=self.vertex_shader(),
            fragment_shader=self.fragment_shader()
        )

        # GPU buffers
        self.vbo = self.ctx.buffer(reserve=2_000_000)
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_pos", "in_uv")

        # Ortho projection
        w, h = self.window_size
        self.prog["ortho"].write(self.ortho_matrix(w, h).astype("f4").tobytes())

    # -------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------
    def render_text(self, text, x=50, y=300, size=64):
        verts = []
        cursor_x = x
        cursor_y = y

        for ch in text:
            code = ord(ch)
            if code not in self.glyphs:
                cursor_x += size * 0.3
                continue

            g = self.glyphs[code]

            if "planeBounds" not in g:
                cursor_x += g["advance"] * size
                continue

            pb = g["planeBounds"]

            # ---- PLANE BOUNDS (correct Y flip) ----
            x0 = cursor_x + pb["left"] * size
            x1 = cursor_x + pb["right"] * size

            # Your JSON uses bottom-origin → flip into OpenGL
            y0 = cursor_y - pb["top"] * size
            y1 = cursor_y - pb["bottom"] * size

            # ---- ATLAS BOUNDS (PIXELS → UV, with Y flip) ----
            ab = g["atlasBounds"]
            w = self.atlas["width"]
            h = self.atlas["height"]

            # Correct order for msdf-atlas-gen
            u0 = ab["left"] / w
            u1 = ab["right"] / w

            # YOU MUST FLIP V LIKE THIS:
            v0 = 1.0 - (ab["top"] / h)
            v1 = 1.0 - (ab["bottom"] / h)

            # ---- Triangles ----
            verts.extend([
                x0, y0, u0, v0,
                x1, y0, u1, v0,
                x0, y1, u0, v1,

                x1, y0, u1, v0,
                x1, y1, u1, v1,
                x0, y1, u0, v1,
            ])

            cursor_x += g["advance"] * size

        verts = np.array(verts, dtype="f4")
        self.vbo.orphan(verts.nbytes)
        self.vbo.write(verts)

        self.font_tex.use(0)
        self.vao.render(moderngl.TRIANGLES)

    # -------------------------------------------------------------------
    # JSON Loader for MSDF Format
    # -------------------------------------------------------------------
    def load_msdf_json(self, path):
        with open(path, "r") as f:
            data = json.load(f)

        # Atlas info
        atlas = {
            "width": data["atlas"]["width"],
            "height": data["atlas"]["height"],
        }

        # Global font metrics
        metrics = data["metrics"]

        glyphs = {}
        for g in data["glyphs"]:
            code = g["unicode"]

            glyph = {
                "advance": g["advance"]
            }

            if "planeBounds" in g:
                glyph["planeBounds"] = {
                    "left": g["planeBounds"]["left"],
                    "right": g["planeBounds"]["right"],
                    "top": g["planeBounds"]["top"],
                    "bottom": g["planeBounds"]["bottom"],
                }
                glyph["atlasBounds"] = {
                    "left": g["atlasBounds"]["left"],
                    "right": g["atlasBounds"]["right"],
                    "top": g["atlasBounds"]["top"],
                    "bottom": g["atlasBounds"]["bottom"],
                }

            glyphs[code] = glyph

        return glyphs, metrics, atlas

    # -------------------------------------------------------------------
    # Ortho Matrix
    # -------------------------------------------------------------------
    def ortho_matrix(self, w, h):
        return np.array([
            [2/w, 0,   0, -1],
            [0, -2/h,  0,  1],
            [0,  0,  -1,  0],
            [0,  0,   0,  1],
        ], dtype="f4")

    # -------------------------------------------------------------------
    # Shaders
    # -------------------------------------------------------------------
    def vertex_shader(self):
        return """
        #version 330
        in vec2 in_pos;
        in vec2 in_uv;

        uniform mat4 ortho;
        out vec2 uv;

        void main() {
            gl_Position = ortho * vec4(in_pos, 0.0, 1.0);
            uv = in_uv;
        }
        """

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)
        self.render_text("Hello World!", x=50, y=300, size=64)

    def fragment_shader(self):
        return """
        #version 330

        uniform sampler2D font_tex;

        in vec2 uv;
        out vec4 fragColor;

        // Standard MSDF edge smoothing
        float median(float r, float g, float b) {
            return max(min(r, g), min(max(r, g), b));
        }

        void main() {
            vec3 sample = texture(font_tex, uv).rgb;
            float sd = median(sample.r, sample.g, sample.b);

            float w = fwidth(sd);
            float alpha = smoothstep(0.5 - w, 0.5 + w, sd);

            fragColor = vec4(1.0, 1.0, 1.0, alpha);
        }
        """




import moderngl_window
def run_renderer():
    moderngl_window.run_window_config(SDFTextRenderer, args=['--window', 'glfw'])


run_renderer()