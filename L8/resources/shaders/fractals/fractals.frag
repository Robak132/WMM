#version 330

in vec2 frag_position;
out vec4 f_color;

uniform vec2 center;
uniform float scale;
uniform int iterations;
uniform float aspect_ratio;

/*
Fraktal Mandelbrota
z[0] = 0
z[1] = p
z[n+1] = (z[n])^2 + p
*/

void main() {
    vec2 start_point;
    start_point.x = aspect_ratio * frag_position.x * scale - center.x;
    start_point.y = frag_position.y * scale - center.y;
    vec2 n_point = start_point;

    int i=0;
    while (i < iterations) {
        float x = (n_point.x * n_point.x - n_point.y * n_point.y) + start_point.x;
        float y = (2.0 * n_point.y * n_point.x) + start_point.y;

        if ((x * x + y * y) > 4.0) break;

        n_point.x = x;
        n_point.y = y;
        i++;
    }

    if (i != iterations) {
        f_color = vec4(0.0f, float(i) / iterations, 0.0f, 1.0f);
    } else {
        f_color = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }
}