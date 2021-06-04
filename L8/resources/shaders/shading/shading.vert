#version 330

uniform mat4 mvp;

in vec3 in_position;
in vec3 in_normal;
out vec3 v_vert;
out vec3 v_normal;

void main() {
    v_vert = in_position;
    v_normal = in_normal;
    gl_Position = mvp * vec4(in_position, 1.0);
}
