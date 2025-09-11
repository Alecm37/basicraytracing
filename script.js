const canvas = document.querySelector('canvas');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const gl = canvas.getContext('webgl2', {antialias: false, depth:false});
gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
gl.getExtension('EXT_color_buffer_float');
gl.getExtension('OES_texture_float_linear');


const program = gl.createProgram();
const program2 = gl.createProgram();

const vertShader = gl.createShader(gl.VERTEX_SHADER);
gl.shaderSource(vertShader, vertShaderSrc);
gl.compileShader(vertShader);
gl.attachShader(program, vertShader);

const fragShader = gl.createShader(gl.FRAGMENT_SHADER);
gl.shaderSource(fragShader, fragShaderSrc);
gl.compileShader(fragShader);
gl.attachShader(program, fragShader);

gl.linkProgram(program);
if(!gl.getProgramParameter(program, gl.LINK_STATUS)){
    console.log('VERTEX_SHADER_ERROR:\n', gl.getShaderInfoLog(vertShader));
    console.log('FRAGMENT_SHADER_ERROR:\n', gl.getShaderInfoLog(fragShader));
}

gl.useProgram(program);
gl.uniform2f(gl.getUniformLocation(program, 'uResolution'), canvas.width, canvas.height);

gl.attachShader(program2, vertShader);

const postShader = gl.createShader(gl.FRAGMENT_SHADER);
gl.shaderSource(postShader, postShaderSrc);
gl.compileShader(postShader);
gl.attachShader(program2, postShader);

gl.linkProgram(program2);
if(!gl.getProgramParameter(program2, gl.LINK_STATUS)){
    console.log('POST_PROCESS_ERROR:\n', gl.getShaderInfoLog(postShader));
}
gl.useProgram(program2);

gl.uniform2f(gl.getUniformLocation(program2, 'uResolution'), canvas.width, canvas.height);

let framebuffers = [];
let textures = [];
for(let i = 0; i < 2; i++){
    let tex = createTex();
    textures.push(tex);

    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, gl.canvas.width, gl.canvas.height, 0, gl.RGBA, gl.FLOAT, null);

    let fb = gl.createFramebuffer();
    framebuffers.push(fb);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);

    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        console.error("Framebuffer is not complete");
    }
}

function createTex(){
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
 
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
 
    return texture;
}


let currentFb = 0;

let frame = 0;


const img = new Image();
const imgBrick = new Image();

img.onload = () => {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); //remove line between pixels (smoothing texture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); //if uvs are sampled outside of a range 0-1 (always do clamp to edge)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.useProgram(program);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(gl.getUniformLocation(program, "uSky"), 2);//pass sky into uniform 
    draw();
};
imgBrick.onload = () => {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imgBrick);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); //remove line between pixels (smoothing texture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); //if uvs are sampled outside of a range 0-1 (always do clamp to edge)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.useProgram(program);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(gl.getUniformLocation(program, "uBrick"), 2);//pass sky into uniform 
    draw();
}

img.src = "textures/sky.png";
imgBrick.src = "textures/brick.png";
function draw(){
    gl.useProgram(program);

    moveCam();

    // console.log(view);

    gl.uniform1f(gl.getUniformLocation(program, 'uFrame'), frame);
    gl.uniformMatrix4fv(gl.getUniformLocation(program, 'uView'), false, view);

    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers[currentFb]);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, textures[1-currentFb]);
    gl.uniform1i(gl.getUniformLocation(program, 'uLastFrame'), 1);
    
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);


    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    gl.useProgram(program2);

    
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, textures[currentFb]);
    gl.uniform1i(gl.getUniformLocation(program2, 'uTex'), 0);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);


    frame++;
    currentFb = 1-currentFb;
    requestAnimationFrame(draw);
}
