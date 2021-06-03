#version 330

uniform mat4 mvp;
in vec3 in_position;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
}
