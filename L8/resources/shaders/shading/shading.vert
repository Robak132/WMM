#version 330
uniform mat4 mvp;

in vec3 in_position;
in vec3 in_normal;

out vec3 position;
out vec3 normal;

void main() {
    position = in_position;
    normal = in_normal;
    gl_Position = mvp * vec4(in_position, 1.0);
}
