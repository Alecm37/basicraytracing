// this file has the camera controller

const keys = {up: 0, down: 0, left: 0, right: 0, w: 0, a: 0, s: 0, d: 0, e: 0, q: 0, slow: 0};
// let cam = {x: -5, y: 5, z: -10, dx: 0, dy: 0, dz: 0};
let cam = {x: 0, y: 0, z: 2, dx: 0, dy: 0, dz: 0};
// let cam = {
//     dx: 0, dy: -0.3059980000000002, dz: 0,
//     x:
//         -0.1875333085761132,
//     y:
//         0, z: 1.0972727223701113,
// }

document.addEventListener('keydown', function(event) {
    switch(event.key) {
        case 'ArrowUp':
            keys.up = 1;
            break;
        case 'ArrowDown':
            keys.down = 1;
            break;
        case 'ArrowLeft':
            keys.left = 1;
            break;
        case 'ArrowRight':
            keys.right = 1;
            break;
        case 'w':
        case 'W':
            keys.w = 1;
            break;
        case 'a':
        case 'A':
            keys.a = 1;
            break;
        case 's':
        case 'S':
            keys.s = 1;
            break;
        case 'd':
        case 'D':
            keys.d = 1;
            break;
        case 'e':
        case 'E':
        case ' ':
            keys.e = 1;
            break;
        case 'q':
        case 'Q':
        case 'Shift':
            keys.q = 1;
            break;
        case 'v':
        case 'V':
            keys.slow = 1;
            break;
    }
});

document.addEventListener('keyup', function(event) {
    switch(event.key) {
        case 'ArrowUp':
            keys.up = 0;
            break;
        case 'ArrowDown':
            keys.down = 0;
            break;
        case 'ArrowLeft':
            keys.left = 0;
            break;
        case 'ArrowRight':
            keys.right = 0;
            break;
        case 'w':
        case 'W':
            keys.w = 0;
            break;
        case 'a':
        case 'A':
            keys.a = 0;
            break;
        case 's':
        case 'S':
            keys.s = 0;
            break;
        case 'd':
        case 'D':
            keys.d = 0;
            break;
        case 'e':
        case 'E':
        case ' ':
            keys.e = 0;
            break;
        case 'q':
        case 'Q':
        case 'Shift':
            keys.q = 0;
            break;
        case 'v':
        case 'V':
            keys.slow = 0;
            break;
    }
});

canvas.addEventListener('mousemove', function(event) {
    if(event.buttons == 1){
        frame=0;
        cam.dy -= event.movementX * 0.000666;
        cam.dx += event.movementY * -0.000666;
    }
}
);

let view = new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
]);

function moveCam() {
    let dt = 0.01;
    if(keys.slow){
        dt *= 0.067;
    }

    // Clamp pitch
    cam.dx += (keys.up - keys.down) * 8.0 * 0.001;
    if (cam.dx < -Math.PI * 0.5) cam.dx = -Math.PI * 0.5;
    if (cam.dx >  Math.PI * 0.5) cam.dx =  Math.PI * 0.5;

    cam.dy += (keys.left - keys.right) * 8.0 * 0.001;

    let sinY = Math.sin(cam.dy), cosY = Math.cos(cam.dy);

    let fwbw = (keys.s - keys.w) * 3.0 * dt;
    let lfrt = (keys.a - keys.d) * 3.0 * dt;

    cam.x += fwbw * -sinY + lfrt * cosY;
    cam.z += fwbw * cosY - lfrt * -sinY;
    cam.y += (keys.e - keys.q) * 3.0 * dt;
    
    let sx = Math.sin(-cam.dx), cx = Math.cos(-cam.dx);
    let sy = Math.sin(-cam.dy), cy = Math.cos(-cam.dy);
    view = [
        cy,        0,         -sy,        0,
        sx * sy,   cx,        sx * cy,    0,
        cx * sy,   -sx,       cx * cy,    0,
        -cam.x,     cam.y,     -cam.z,      1
    ];

    if(Object.values(keys).some(v => v === 1)){
        frame=0;
    }
}
