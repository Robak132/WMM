#version 330

in vec2 in_position;
out vec2 frag_position;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    frag_position = in_position;
}
