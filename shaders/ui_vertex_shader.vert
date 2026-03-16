#version 330

in vec2 in_pos;
in vec2 in_uv;
in vec4 in_color;

uniform mat4 ortho;

out vec2 uv;
out vec4 color;

void main() {
    uv = in_uv;
    color = in_color;
    gl_Position = ortho * vec4(in_pos, 0.0, 1.0);
}