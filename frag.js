const fragShaderSrc = /*glsl*/`#version 300 es

precision mediump float; //every float declared is 16 bit 

out vec4 fragColor;

uniform float uFrame; //frame index
uniform mat4 uView;  //view matrix 
uniform vec2 uResolution; 
uniform sampler2D uLastFrame;//last frame rendered 
uniform sampler2D uSky; 
uniform sampler2D uBrick; 

#define FOCAL_LEN 2.0 //length between camera and pixels 
#define SPHERE_CENTER vec3(0.0, -2., 12.0)
#define SPHERE_RADIUS 0.75
#define PLANE_CENTER vec4(0.0, 1.0, 0.0, 1.0)
#define PLANE_OFFSET vec3(0.0, 0.0, 10.0)
#define PI 3.14159265
#define MAX_BOUNCES 8
#define EPS 1e-4

//ro = ray origin, rd = ray direction, ce = center pos circle, ra= radius 
vec2 sphIntersect( in vec3 ro, in vec3 rd, in vec3 ce, float ra, in vec3 sphOffset) //sphere intersection function, returns distance between first and second intersections, -1 if missed entirely (no intersection)
{
    vec3 oc = (ro-sphOffset) - ce; //distance between the camera and the sphere  
    float b = dot( oc, rd ); //dot product between the distance vector and the ray direction vector 
    float c = dot( oc, oc ) - ra*ra; //dot product squared of distance to center vector - squared radius = how far ray origin is from radius 
    float h = b*b - c;
    if( h<0.0 ) return vec2(-1.0); // no intersection
    h = sqrt( h );
    return vec2( -b-h, -b+h ); //intersection in 2D space 
}



float plaIntersect(in vec3 ro, in vec3 rd, in vec4 p, in vec3 plaOffset)
{
    ro -= (PLANE_OFFSET + plaOffset); //moving ray origin to just simply offset the ray intersection 
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

// axis aligned box centered at the origin, with size boxSize
vec2 boxIntersection( in vec3 ro, in vec3 rd, vec3 boxSize, out vec3 outNormal, in vec3 boxOffset) 
{
    ro -= boxOffset;
    vec3 m = 1.0/rd; // can precompute if traversing a set of aligned boxes
    vec3 n = m*ro;   // can precompute if traversing a set of aligned boxes
    vec3 k = abs(m)*boxSize;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    float tN = max( max( t1.x, t1.y ), t1.z );
    float tF = min( min( t2.x, t2.y ), t2.z );
    if( tN>tF || tF<0.0) return vec2(-1.0); // no intersection
    outNormal = (tN>0.0) ? step(vec3(tN),t1) : // ro ouside the box
                           step(t2,vec3(tF));  // ro inside the box
    outNormal *= -sign(rd);
    return vec2( tN, tF );
}




vec3 sampleHdri(vec3 d){ //input ray direction 
    float lon = atan(d.z, d.x); //returns angle of opposite over adjancent (x)
    float lat = asin(d.y); //returns angle of sine y (y)
    float u = 1. - (lon + PI) / (2. * PI); //transform into y pixel coords (where texture would go)
    float v = (lat + PI*0.5) / PI; //transform into x pixel coords 
    return texture(uSky, vec2(u, v)).rgb; //input sky uniform and our texture pixels to return a vec4 -> vec3 of rgb texture vals for a pixel 
}

struct Surface{
    bool hit; //did it hit? yes or no (intersection)
    bool exit; 
    vec3 position; //vector pointing to intersection
    vec3 normal; //normalize the position of point intersected 
    vec3 albedo; //base color of object 
    vec3 emission; //emissive color of object 
    float ior; //index of refraction 
    bool std; 
};

Surface shootRay(vec3 ro, vec3 rd){
    Surface surface; 
    surface.hit = false; 

    float tMin = 1e10; 
    vec3 sphereOffset = vec3(0., 2.0, 0.);
    float sphereX = sphIntersect(ro, rd, SPHERE_CENTER, SPHERE_RADIUS, sphereOffset).x;
    float sphereY = sphIntersect(ro, rd, SPHERE_CENTER, SPHERE_RADIUS, sphereOffset).y;

    if (sphereX > 0.){
        tMin = sphereX; 
        surface.hit = true; 
        surface.position = ro + rd * sphereX; 
        surface.normal = normalize(surface.position - SPHERE_CENTER - sphereOffset);
        surface.albedo = vec3(0.1, 0.4, 0.9); 
        surface.emission = vec3(0.2);
        surface.ior = 1.5;
        surface.std = false; 
    }
    else if (sphereX < 0. && sphereY > 0.){
        tMin = sphereY; 
        surface.hit = true; 
        surface.position = ro + rd * sphereY; 
        surface.normal = -normalize(surface.position - SPHERE_CENTER - sphereOffset);
        surface.albedo = vec3(0.1, 0.4, 0.9); 
        surface.emission = vec3(0.2);
        surface.ior = 1./1.5;
        surface.std = false; 
    }
    vec3 plaOffset = vec3(0.);
    float plane = plaIntersect(ro, rd, PLANE_CENTER, plaOffset);
    if (plane > 0. && plane < tMin){
        tMin = plane; 
        surface.hit = true; 
        surface.position = ro+rd * plane;
        surface.normal = PLANE_CENTER.xyz - plaOffset; 
        surface.albedo = vec3(0.4, 0.4, 0.4);
        surface.emission = vec3(0.);
        surface.ior = 1.;
        surface.std = false; 
    }
    vec3 boxSize = vec3(0.5);
    vec3 boxNormal = vec3(0.);
    vec3 boxOffset = vec3(0., 0., 8.);
    float boxX = boxIntersection(ro, rd, boxSize, boxNormal, boxOffset).x;
    float boxY = boxIntersection(ro, rd, boxSize, boxNormal, boxOffset).y;
    if (boxX > 0. && boxX < tMin){
        tMin = boxX; 
        surface.hit = true; 
        surface.position = ro+rd*boxX; 
        surface.normal = boxNormal;
        surface.albedo = vec3(0.4, 0.4, 0.4);
        surface.emission = vec3(0.);
        surface.ior = 1.5; 
        surface.std = false; 
    }
    else if (boxX < 0. && boxY < tMin && boxY > 0.){
        tMin = boxY; 
        surface.hit = true; 
        surface.position = ro + rd * boxY; 
        surface.normal = -boxNormal;
        surface.albedo = vec3(0.1, 0.4, 0.9); 
        surface.emission = vec3(0.2);
        surface.ior = 1./1.5;
        surface.std = false; 
    }
    vec3 lightBoxSize = vec3(0.5);
    vec3 lightBoxNormal = vec3(0.);
    vec3 lightBoxOffset = vec3(3., 1., 10.);
    float lightBox = boxIntersection(ro, rd, lightBoxSize, lightBoxNormal, lightBoxOffset).x;
    if (lightBox > 0. && lightBox < tMin){
        tMin = lightBox; 
        surface.hit = true; 
        surface.position = ro+rd*lightBox; 
        surface.normal = lightBoxNormal;
        surface.albedo = vec3(0.8, 0.8, 0.8);
        surface.emission = vec3(10.);
        surface.ior = 1.5; 
        surface.std = false; 
    }
    return surface;
}

uint hash21( uvec2 p ){  //pseudo random to get random num
    p *= uvec2(73333,7777);
    p ^= (uvec2(3333777777)>>(p>>28));
    uint n = p.x*p.y;
    return n^(n>>15);
}

float hash( uvec2 p ){ //pseudo random to get random num
    uint h = hash21( p );
    return float(h)*(1.0/float(0xffffffffU));
}

uvec2 getSeed(){
    return uvec2(gl_FragCoord.xy + mod(uFrame * vec2(913.27, 719.92), 9382.239)); // do some  bs to get a "random" number 
}

uvec2 seed; 

float pickRandom(){
    seed += uvec2(1, 1); //change seed 
    return hash(seed);
}

vec3 sampleHemiSphere(vec3 n){//input surface.normal, output vector representing diffuse lighting 
    float theta = pickRandom() * 2. * PI; //random number in radians (angle)
    float phi = pickRandom() * 2. * PI;
    vec3 p = vec3(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)); //vector that converts spherical coordinates to cartesian
    p+=n; //add this to our normal of intersection point vector with shape 
    return normalize(p);
}

float fresnel(float costheta, float eta){ //input costheta of incoming ray and normal, ratio between current ior and future ior 
    //fixing ior to scale correctly using (abriged) inverse sigmoid function 
    //n(scaled) = -ln((1/x)-1)+1
   
    float c = abs(costheta);
    float g = eta * eta * 2. - 1. + c * c; //snell's law, if TIR occurs 
    if(g > 0.){ //fresnel equations 
        g = sqrt(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1.) / (c * (g - c) + 1.);
        return 0.5 * A * A * (1. + B * B); //return average reflectance 
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

    rayDirection = mat3(uView) * rayDirection; //standard ray direction
    vec3 rayOrigin = uView[3].xyz;
    
    vec3 radiance = vec3(0.);
    vec3 throughput = vec3(1.);



   
   
    //reflections and refractions 
    for (int i = 0; i < MAX_BOUNCES; i++){
        Surface surface = shootRay(rayOrigin, rayDirection);
        float ior = surface.ior;
        float eta = (dot(rayDirection, surface.normal) < 0.0) ? 1./ior : ior;  
        vec3 N = (dot(rayDirection, surface.normal) < 0.0) ? surface.normal : -surface.normal;

        if(!surface.hit){
            radiance+=throughput * sampleHdri(rayDirection); 
            break; 
        }
        if(surface.std){
            radiance = surface.albedo; 
            break;
        }
        vec3 wo; 
        float fresnelRefl = fresnel(dot(rayDirection, N), ior);
        bool isReflection = false; 
        if(pickRandom() < fresnelRefl){
            if(pickRandom()>.5){
            wo = reflect(rayDirection, N);
            isReflection = true;
            }
            else{
            wo = sampleHemiSphere(N);
            }
        } 
        else {
            if(pickRandom()>.5){
            wo = refract(rayDirection, N, eta);
            isReflection = true;
            }
            else{
            wo = sampleHemiSphere(N);
            }
        }
        radiance += surface.emission * throughput;
        throughput *= ((isReflection)? vec3(1.) : surface.albedo);

        rayOrigin = surface.position + normalize(wo) * EPS; 
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