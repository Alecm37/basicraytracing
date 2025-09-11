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
#define SPHERE_CENTER vec3(0.0, -0.25, 10.0)
#define SPHERE_RADIUS 0.75
#define PLANE_CENTER vec4(0.0, 1.0, 0.0, 1.0)
#define PLANE_OFFSET vec3(0.0, 0.0, 10.0)
#define PI 3.14159265

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

float torIntersect( in vec3 ro, in vec3 rd, in vec2 tor, in vec3 torOffset )
{
    ro -= torOffset;
    float po = 1.0;
    float Ra2 = tor.x*tor.x;
    float ra2 = tor.y*tor.y;
    float m = dot(ro,ro);
    float n = dot(ro,rd);
    float k = (m + Ra2 - ra2)/2.0;
    float k3 = n;
    float k2 = n*n - Ra2*dot(rd.xy,rd.xy) + k;
    float k1 = n*k - Ra2*dot(rd.xy,ro.xy);
    float k0 = k*k - Ra2*dot(ro.xy,ro.xy);
    
    if( abs(k3*(k3*k3-k2)+k1) < 0.01 )
    {
        po = -1.0;
        float tmp=k1; k1=k3; k3=tmp;
        k0 = 1.0/k0;
        k1 = k1*k0;
        k2 = k2*k0;
        k3 = k3*k0;
    }
    
    float c2 = k2*2.0 - 3.0*k3*k3;
    float c1 = k3*(k3*k3-k2)+k1;
    float c0 = k3*(k3*(c2+2.0*k2)-8.0*k1)+4.0*k0;
    c2 /= 3.0;
    c1 *= 2.0;
    c0 /= 3.0;
    float Q = c2*c2 + c0;
    float R = c2*c2*c2 - 3.0*c2*c0 + c1*c1;
    float h = R*R - Q*Q*Q;
    
    if( h>=0.0 )  
    {
        h = sqrt(h);
        float v = sign(R+h)*pow(abs(R+h),1.0/3.0); // cube root
        float u = sign(R-h)*pow(abs(R-h),1.0/3.0); // cube root
        vec2 s = vec2( (v+u)+4.0*c2, (v-u)*sqrt(3.0));
        float y = sqrt(0.5*(length(s)+s.x));
        float x = 0.5*s.y/y;
        float r = 2.0*c1/(x*x+y*y);
        float t1 =  x - r - k3; t1 = (po<0.0)?2.0/t1:t1;
        float t2 = -x - r - k3; t2 = (po<0.0)?2.0/t2:t2;
        float t = 1e20;
        if( t1>0.0 ) t=t1;
        if( t2>0.0 ) t=min(t,t2);
        return t;
    }
    
    float sQ = sqrt(Q);
    float w = sQ*cos( acos(-R/(sQ*Q)) / 3.0 );
    float d2 = -(w+c2); if( d2<0.0 ) return -1.0;
    float d1 = sqrt(d2);
    float h1 = sqrt(w - 2.0*c2 + c1/d1);
    float h2 = sqrt(w - 2.0*c2 - c1/d1);
    float t1 = -d1 - h1 - k3; t1 = (po<0.0)?2.0/t1:t1;
    float t2 = -d1 + h1 - k3; t2 = (po<0.0)?2.0/t2:t2;
    float t3 =  d1 - h2 - k3; t3 = (po<0.0)?2.0/t3:t3;
    float t4 =  d1 + h2 - k3; t4 = (po<0.0)?2.0/t4:t4;
    float t = 1e20;
    if( t1>0.0 ) t=t1;
    if( t2>0.0 ) t=min(t,t2);
    if( t3>0.0 ) t=min(t,t3);
    if( t4>0.0 ) t=min(t,t4);
    return t;
}

vec3 torNormal( in vec3 pos, vec2 tor)
{
    return normalize( pos*(dot(pos,pos)-tor.y*tor.y - tor.x*tor.x*vec3(1.0,1.0,-1.0)));
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
    vec3 position; //vector pointing to intersection
    vec3 normal; //normalize the position of point intersected 
    vec3 albedo; //base color of object 
    vec3 emission; //emissive color of object 
    float ior; //index of refraction 
    float transp; //transparency (1 is fully transparent, 0 is fully opaque)
};

Surface shootRay(vec3 ro, vec3 rd){
    Surface surface; 
    surface.hit = false; 

    float tMin = 1e10; 
    vec3 sphereOffset = vec3(0., 2.0, 0.);
    float sphere = sphIntersect(ro, rd, SPHERE_CENTER, SPHERE_RADIUS, sphereOffset).x;
    if (sphere > 0.){
        tMin = sphere; 
        surface.hit = true; 
        surface.position = ro + rd * sphere; 
        surface.normal = normalize(surface.position - SPHERE_CENTER - sphereOffset);
        surface.albedo = vec3(0.5, 0.5, 0.5); 
        surface.emission = vec3(0.);
        surface.ior = 3.;
        surface.transp = 1.; 
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
        surface.ior = 1.45;
        surface.transp = 1.; 
    }
    vec2 torVec = vec2(0.5, 0.25);
    vec3 torOffset = vec3(0., 0., 7.0);
    float tor = torIntersect(ro, rd, torVec, torOffset);
    if (tor > 0. && tor < tMin){
        tMin = tor; 
        surface.hit = true; 
        surface.position = ro+rd*tor; 
        surface.normal = torNormal(surface.position - torOffset, torVec);
        surface.albedo = vec3(0.5, 0.5, 0.5);
        surface.emission = vec3(0.);
        surface.ior = 3.; 
        surface.transp = 1.;
    }
    vec3 boxSize = vec3(0.5);
    vec3 boxNormal = vec3(0.);
    //vec3 boxPos  = vec3(2.0, -0.5, 10.0);
    vec3 boxOffset = vec3(0., 0., 8.);
    float box = boxIntersection(ro, rd, boxSize, boxNormal, boxOffset).x;
    if (box > 0. && box < tMin){
        tMin = box; 
        surface.hit = true; 
        surface.position = ro+rd*box; 
        surface.normal = boxNormal;
        surface.albedo = vec3(0.5, 0.5, 0.5);
        surface.emission = vec3(0.);
        surface.ior = 3.; 
        surface.transp = 0.5;
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
    float c = abs(costheta);
    float g = eta * eta - 1. + c * c; //snell's law, if TIR occurs 
    if(g > 0.){ //fresnel equations 
        g = sqrt(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1.) / (c * (g - c) + 1.);
        return 0.5 * A * A * (1. + B * B); //return average reflectance 
    }else{
        return 1.;
    }
}

//glass 

//fog 

//maybe port to C++??


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
        float fresnel = fresnel(dot(rayDirection, surface.normal), surface.ior);
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