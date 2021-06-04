#version 330
uniform vec3 look;

out vec4 f_color;
in vec3 v_vert;
in vec3 v_normal;

vec3 light_position = vec3(1.0, 0.5, 0.0);
vec3 light_color = vec3(1.0, 1.0, 1.0);
float light_ambient = 0.1;
float light_diffuse = 0.5;
float light_specular = 0.5;

vec3 object_diffuse = vec3(0.563, 0.059, 0.376);
float object_specular = 0.5;
int object_shininess = 8;

void main() {
    // Difussion
    vec3 light_position_n = normalize(light_position);
    vec3 color = object_diffuse * light_color * clamp(dot(v_normal,light_position_n), 0, 1);

    // Ambient
    color += light_ambient * light_color;

    // Specular
//    vec4 look_n = normalize(viewPos - v_vert);
//    vec3 reflect = reflect(-light_position_n, v_normal);
//    color += object_specular * light_color * pow(clamp(dot(look_n, reflect), 0, 1), object_shininess);

    // kalkulacja oświetlenia ambientowego
//    vec3 ambient = light_ambient * light_color;

    // kalkulacja dyfuzji
//    vec3 norm = normalize(v_norm);
//    vec3 lightDir = normalize(lightPos - v_vert);
//    float diff = max(dot(norm, lightDir), 0.0);
//    vec3 diffuse = diff * lightColor * diffuse_m;
//
//    vec3 viewDir = normalize(viewPos - v_vert);
//    vec3 reflectDir = reflect(-lightDir, norm);
//    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
//    vec3 specular = specular_m * spec * lightColor;

    f_color = vec4(color, 1.0);
}
