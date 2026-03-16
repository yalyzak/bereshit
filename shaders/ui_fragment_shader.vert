#version 330

uniform sampler2D tex;
uniform int use_texture;

in vec2 uv;
in vec4 color;

out vec4 fragColor;

void main() {

    if (use_texture == 1)
        fragColor = texture(tex, uv) * color;
    else
        fragColor = color;
}