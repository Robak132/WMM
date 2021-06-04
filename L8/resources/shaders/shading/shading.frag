#version 330
out vec3 f_color;

in vec3 position;
in vec3 normal;

uniform vec3 viewer_position = vec3(1, 0.75, -0.8);

uniform vec3 light_color = vec3(1.0, 1.0, 1.0);
uniform vec3 light_position = vec3(0.5, 0.75, -0.8);

uniform int shininess = 16;
uniform vec3 object_color = vec3(1.0, 0.0, 0.0);

uniform float ambient_str = 0.2;
uniform float diffuse_str = 0.9;
uniform float specular_str = 0.3;

void main() {
    // Ambient
    vec3 ambient = ambient_str * light_color;

    // Diffuse
    vec3 norm = normalize(normal);
    // Wyliczanie kąta padania swiatla
    vec3 light_direction = normalize(light_position - position);
    // Wyliczanie efektu "diffuse" i ograniczenie go do przedziału 0-1
    float diffuse_modifier = min(max(dot(norm, light_direction), 0.0), 1.0);
    vec3 diffuse = diffuse_modifier * light_color * diffuse_str;

    // Specular
    // Wyliczanie kąta patrzenia na obiekt
    vec3 view_direction = normalize(viewer_position - position);
    // Wyliczanie kąta, pod którym odbija się światło
    vec3 reflect_direction = reflect(-light_direction, norm);
    // Wyliczanie efektu "specular" i ogarniczenie go do przedziału 0-1
    float specular_modifier = pow(min(max(dot(view_direction, reflect_direction), 0.0), 1.0), shininess);
    vec3 specular = specular_str * specular_modifier * light_color;

    // Połączenie efektów i powiązanie z kolorem obiektu
    f_color = vec3(min(max((ambient + diffuse) * object_color + specular, 0.0), 1.0));
}