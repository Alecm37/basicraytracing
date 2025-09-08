const fragShaderSrc = /*glsl*/`#version 300 es

precision mediump float; //every float declared is 16 bit 

out vec4 fragColor;

uniform float uFrame; //frame index
uniform mat4 uView;  //view matrix 
uniform vec2 uResolution; 
uniform sampler2D uLastFrame;//last frame rendered 
uniform sampler2D uSky; 

#define FOCAL_LEN 2.0 //length between camera and pixels 
#define SPHERE_CENTER vec3(0.0, 0.0, 10.0)
#define SPHERE_RADIUS 1.0
#define PLANE_CENTER vec4(0.0, 1.0, 0.0, 1.0)
#define PLANE_OFFSET vec3(0.0, 0.0, 10.0)
#define PI 3.14159265

//ro = ray origin, rd = ray direction, ce = center pos circle, ra= radius 
vec2 sphIntersect( in vec3 ro, in vec3 rd, in vec3 ce, float ra ) //sphere intersection function, returns distance between first and second intersections, -1 if missed entirely (no intersection)
{
    vec3 oc = ro - ce; //distance between the camera and the sphere  
    float b = dot( oc, rd ); //dot product between the distance vector and the ray direction vector 
    float c = dot( oc, oc ) - ra*ra; //dot product squared of distance to center vector - squared radius = how far ray origin is from radius 
    float h = b*b - c;
    if( h<0.0 ) return vec2(-1.0); // no intersection
    h = sqrt( h );
    return vec2( -b-h, -b+h ); //intersection in 2D space 
}

float plaIntersect(in vec3 ro, in vec3 rd, in vec4 p)
{
    ro -= PLANE_OFFSET; //moving ray origin to just simply offset the ray intersection 
    float denom = dot(rd, p.xyz);
    if (abs(denom) < 1e-6) return -1.0; // parallel ray

    float t = -(dot(ro, p.xyz) + p.w) / denom;
    if (t < 0.0) return -1.0; // behind the camera

    vec3 hp = ro + t * rd;
    if (abs(hp.x) <= 5.0 && abs(hp.z) <= 5.0) {
        return t;
    } else {
        return -1.0;
    }
}

vec3 sampleHdri(vec3 d){ //input ray direction, and returns texture 
    float lon = atan(d.z, d.x); 
    float lat = asin(d.y);

    float u = 1. - (lon + PI) / (2. * PI);
    float v = (lat + PI*0.5) / PI;

    return texture(uSky, vec2(u, v)).rgb;
}

struct Surface{
    bool hit; 
    vec3 position; 
    vec3 normal; 
    vec3 albedo; 
    vec3 emission; 
};

Surface shootRay(vec3 ro, vec3 rd){
    Surface surface; 
    surface.hit = false; 

    float tMin = 1e10; 

    float sphere = sphIntersect(ro, rd, SPHERE_CENTER, SPHERE_RADIUS).x;
    if (sphere > 0.){
        tMin = sphere; 
        surface.hit = true; 
        surface.position = ro + rd * sphere; 
        surface.normal = normalize(surface.position - SPHERE_CENTER);
        surface.albedo = vec3(0.8, 0.3, 0.3);
        surface.emission = vec3(0.);
    }
    float plane = plaIntersect(ro, rd, PLANE_CENTER);
    if (plane > 0. && plane < tMin){
        tMin = plane; 
        surface.hit = true; 
        surface.position = ro+rd * plane;
        surface.normal = PLANE_CENTER.xyz; 
        surface.albedo = vec3(0.8, 0.8, 0.3);
        surface.emission = vec3(0.);
    }
    return surface; 
}

uint hash21( uvec2 p ){ 
    p *= uvec2(73333,7777);
    p ^= (uvec2(3333777777)>>(p>>28));
    uint n = p.x*p.y;
    return n^(n>>15);
}

float hash( uvec2 p ){
    uint h = hash21( p );
    return float(h)*(1.0/float(0xffffffffU));
}

uvec2 getSeed(){
    return uvec2(gl_FragCoord.xy + mod(uFrame * vec2(913.27, 719.92), 9382.239));
}

uvec2 seed; 

float pickRandom(){
    seed += uvec2(1, 1);
    return hash(seed);
}

vec3 sampleHemiSphere(vec3 n){
    float theta = pickRandom() * 2. * PI;
    float phi = pickRandom() * 2. * PI;
    vec3 p = vec3(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));
    p+=n;
    return normalize(p);
}

float fresnel(float costheta, float eta){
    float c = abs(costheta);
    float g = eta * eta - 1. + c * c;
    if(g > 0.){
        g = sqrt(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1.) / (c * (g - c) + 1.);
        return 0.5 * A * A * (1. + B * B);
    }else{
        return 1.;
    }
}


void main(){
    seed = getSeed();
    vec3 col; 
    vec2 uv = (gl_FragCoord.xy/uResolution)*2.0-1.0; //screen coordinates normalized between -1, 1
    uv *= vec2(uResolution.x/uResolution.y, 1.); //multiply x by aspect ratio so it's a circle  
    vec3 pixelPos = vec3(uv, FOCAL_LEN); //pixel position from uv and FOCAL_LEN (z)
    vec3 rayDirection = normalize(pixelPos); //return ray direction 

    rayDirection = mat3(uView) * rayDirection;
    vec3 rayOrigin = uView[3].xyz;

    vec3 radiance = vec3(0.);
    vec3 throughput = vec3(1.);

    for (int i = 0; i < 8; i++){
        Surface surface = shootRay(rayOrigin, rayDirection);
        if(!surface.hit){
            radiance+=throughput * sampleHdri(rayDirection); 
            break; 
        }
        vec3 wo; 
        float fresnel = fresnel(dot(rayDirection, surface.normal), 1.5);
        //radiance = vec3(fresnel); break;
        bool isReflection = false; 
        if(pickRandom() < fresnel){
            wo = reflect(rayDirection, surface.normal);
            isReflection = true; 
        } else {
            wo = sampleHemiSphere(surface.normal);
        }


        radiance += throughput * surface.emission; 
        throughput *= (isReflection)? vec3(1.) : surface.albedo;

        rayOrigin = surface.position + surface.normal * 0.0001; 
        //rayDirection = reflect(rayDirection, surface.normal);
        rayDirection = wo;
    }

    vec3 lastFrame = texture(uLastFrame, gl_FragCoord.xy/uResolution).rgb; //averages each frame 
    fragColor.rgb = (radiance + lastFrame*uFrame)/(uFrame+1.);
}
`

const postShaderSrc = /*glsl*/`#version 300 es
precision mediump float;
out vec4 fragColor;
uniform vec2 uResolution;
uniform sampler2D uTex;

vec3 aces(vec3 x){
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0); //maps from HDR to LDR 
}

//gl_FragCoord.xy: pixel coordinates  -> /uResolution, normalizes between 0 and 1

void main(){
    vec3 col = texture(uTex, gl_FragCoord.xy/uResolution).rgb;

    //col = aces(col);

    col = pow(col, vec3(1./2.2)); //gamma correction -> inverse curve to make the gamma mapping actually linear 
    fragColor = vec4(col, 1.);
}
`

//hash function: takes a number called 'seed', outputs random-looking number 
//used in diffuse lighting 
//