#version 330

in vec2 texcoord;

uniform sampler2D texture1;

out vec4 frag_color;

void main()
{
    frag_color = texture(texture1, texcoord);
}