#version 330

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

in vec3 in_position;
in vec2 in_texcoord;

out vec2 texcoord;

void main()
{
    texcoord = in_texcoord;
    gl_Position = projection * view * model * vec4(in_position, 1.0);
}