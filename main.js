// ═══════════════════════════════════════════════════════════════
// main.js — Three.js 渲染 + UI + 統一物理後端介面
// ═══════════════════════════════════════════════════════════════
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ────────────────── 1. 太陽系初始資料 ──────────────────
const TWO_PI = 2 * Math.PI;
const G_MAIN = 4 * Math.PI * Math.PI;

const initialBodiesData = [
    { id:'sun',     name:'太陽',   m:1.0,      x:0,     y:0, z:0, vx:0, vy:0, vz:0,                             color:0xffdd00, radius:0.1  },
    { id:'mercury', name:'水星',   m:1.65e-7,  x:0.387, y:0, z:0, vx:0, vy:0, vz:TWO_PI/Math.sqrt(0.387),       color:0xaaaaaa, radius:0.02 },
    { id:'venus',   name:'金星',   m:2.45e-6,  x:0.723, y:0, z:0, vx:0, vy:0, vz:TWO_PI/Math.sqrt(0.723),       color:0xffcc88, radius:0.04 },
    { id:'earth',   name:'地球',   m:3.0e-6,   x:1.0,   y:0, z:0, vx:0, vy:0, vz:TWO_PI,                        color:0x4488ff, radius:0.045},
    { id:'mars',    name:'火星',   m:3.2e-7,   x:1.524, y:0, z:0, vx:0, vy:0, vz:TWO_PI/Math.sqrt(1.524),       color:0xff5533, radius:0.03 },
    { id:'jupiter', name:'木星',   m:0.00095,  x:5.20,  y:0, z:0, vx:0, vy:0, vz:TWO_PI/Math.sqrt(5.20),        color:0xddaa77, radius:0.08 },
    { id:'saturn',  name:'土星',   m:0.00028,  x:9.58,  y:0, z:0, vx:0, vy:0, vz:TWO_PI/Math.sqrt(9.58),        color:0xeecc99, radius:0.07 },
    { id:'uranus',  name:'天王星', m:4.37e-5,  x:19.20, y:0, z:0, vx:0, vy:0, vz:TWO_PI/Math.sqrt(19.20),       color:0x66ccff, radius:0.06 },
    { id:'neptune', name:'海王星', m:0.00005,  x:30.05, y:0, z:0, vx:0, vy:0, vz:TWO_PI/Math.sqrt(30.05),       color:0x3366ff, radius:0.06 },
];

// 準備物理狀態 + 歸零質心速度
const physicsState = initialBodiesData.map(b => ({ m:b.m, x:b.x, y:b.y, z:b.z, vx:b.vx, vy:b.vy, vz:b.vz, ax:0, ay:0, az:0 }));
{
    let tM=0, px=0, py=0, pz=0;
    physicsState.forEach(b => { tM+=b.m; px+=b.m*b.vx; py+=b.m*b.vy; pz+=b.m*b.vz; });
    const cx=px/tM, cy=py/tM, cz=pz/tM;
    physicsState.forEach(b => { b.vx-=cx; b.vy-=cy; b.vz-=cz; });
}

let mainThreadBodies = initialBodiesData.map(b => ({ m:b.m, x:b.x, z:b.z }));

// ────────────────── 2. Three.js 場景 ──────────────────
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x000000, 0.015);
const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.01, 1000);
camera.position.set(0, 15, 20);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
container.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.05; controls.maxDistance = 200;
scene.add(new THREE.GridHelper(100, 100, 0x333333, 0x111111));
const planeGeo = new THREE.PlaneGeometry(500, 500); planeGeo.rotateX(-Math.PI/2);
const hitPlane = new THREE.Mesh(planeGeo, new THREE.MeshBasicMaterial({ visible: false }));
scene.add(hitPlane);

// ────────────────── 3. 天體視覺 ──────────────────
const meshes = [], trails = [], MAX_TRAIL = 300;

function createTextSprite(msg, col) {
    const c = document.createElement('canvas'); c.width=256; c.height=128;
    const ctx = c.getContext('2d');
    ctx.font='bold 32px sans-serif'; ctx.textAlign='center';
    ctx.fillStyle=col; ctx.shadowColor='black'; ctx.shadowBlur=4;
    ctx.fillText(msg, 128, 64);
    const tex = new THREE.CanvasTexture(c);
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map:tex, depthTest:false }));
    sp.scale.set(2,1,1); sp.renderOrder=999;
    return sp;
}

function createBodyVisual(d) {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(d.radius,32,32), new THREE.MeshBasicMaterial({color:d.color}));
    mesh.position.set(d.x, d.y||0, d.z);
    const sp = createTextSprite(d.name, '#'+d.color.toString(16).padStart(6,'0'));
    sp.position.y = d.radius + 0.2;
    mesh.add(sp);
    const tGeo = new THREE.BufferGeometry();
    const tPos = new Float32Array(MAX_TRAIL*3);
    tGeo.setAttribute('position', new THREE.BufferAttribute(tPos, 3));
    tGeo.setDrawRange(0, 0);
    const trail = new THREE.Line(tGeo, new THREE.LineBasicMaterial({color:d.color,transparent:true,opacity:0.4,blending:THREE.AdditiveBlending}));
    scene.add(mesh); scene.add(trail);
    meshes.push(mesh);
    trails.push({ line:trail, positions:tPos, count:0, skip:0 });
}
initialBodiesData.forEach(createBodyVisual);

// ────────────────── 4. 物理後端 (WebGPU 優先, Worker fallback) ──────────────────

// Worker fallback 封裝
class WorkerPhysics {
    constructor() { this.worker=null; this._resolve=null; }
    async init(bodies) {
        this.worker = new Worker('physics-worker.js');
        this.worker.onmessage = e => {
            if (e.data.type==='update' && this._resolve) {
                this._resolve({
                    positions: new Float32Array(e.data.positions),
                    masses: e.data.masses ? new Float32Array(e.data.masses) : null
                });
                this._resolve = null;
            }
        };
        this.worker.postMessage({ type:'init', bodies });
    }
    step(n) { return new Promise(r => { this._resolve=r; this.worker.postMessage({type:'step',steps:n}); }); }
    addBody(b) { this.worker.postMessage({type:'add',body:b}); }
    updateSettings(s) { this.worker.postMessage({type:'update_settings',...s}); }
}

let physics;
let backendName = 'CPU Worker';

async function initPhysics() {
    if (navigator.gpu) {
        try {
            const { WebGPUPhysics } = await import('./physics-gpu.js');
            physics = new WebGPUPhysics();
            await physics.init(physicsState);
            backendName = 'WebGPU';
            console.log('✅ WebGPU physics initialized');
            return;
        } catch(e) {
            console.warn('⚠️ WebGPU init failed, falling back to Worker:', e);
        }
    }
    physics = new WorkerPhysics();
    await physics.init(physicsState);
    console.log('✅ CPU Worker physics initialized (fallback)');
}

// ────────────────── 5. UI 互動 ──────────────────
const uiContent = document.getElementById('ui-content');
const toggleUiBtn = document.getElementById('toggle-ui-btn');
toggleUiBtn.addEventListener('click', () => {
    uiContent.classList.toggle('hidden');
    toggleUiBtn.innerText = uiContent.classList.contains('hidden') ? '🔼' : '🔽';
});

let interactionMode = 'view';
const modeViewBtn = document.getElementById('mode-view');
const modePlaceBtn = document.getElementById('mode-place');
const tipText = document.getElementById('tip-text');

function setMode(mode) {
    interactionMode = mode;
    if (mode==='view') {
        modeViewBtn.className='flex-1 py-1.5 px-2 bg-blue-600 rounded-lg text-sm font-bold border border-blue-400 transition-colors';
        modePlaceBtn.className='flex-1 py-1.5 px-2 bg-white/10 rounded-lg text-sm font-bold border border-white/20 hover:bg-white/20 transition-colors';
        tipText.innerText='💡 目前為「觀察模式」：可左鍵拖曳旋轉、右鍵平移、滾輪縮放視角';
    } else {
        modePlaceBtn.className='flex-1 py-1.5 px-2 bg-blue-600 rounded-lg text-sm font-bold border border-blue-400 transition-colors';
        modeViewBtn.className='flex-1 py-1.5 px-2 bg-white/10 rounded-lg text-sm font-bold border border-white/20 hover:bg-white/20 transition-colors';
        tipText.innerText='💡 目前為「放置模式」：點擊網格即可放置天體';
    }
}
modeViewBtn.addEventListener('click', () => setMode('view'));
modePlaceBtn.addEventListener('click', () => setMode('place'));
setMode('view');

let selectedExtremeType = 'wd';
const EXTREME = {
    'wd': { name:'白矮星', m:1.2,  color:0xffffff, radius:0.05 },
    'ns': { name:'中子星', m:2.0,  color:0x00ffff, radius:0.03 },
    'bh': { name:'黑洞',   m:50.0, color:0xaa00ff, radius:0.2  },
};
document.querySelectorAll('.celestial-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.celestial-btn').forEach(b => b.classList.remove('bg-white/30','border-white'));
        btn.classList.add('bg-white/30','border-white');
        selectedExtremeType = btn.dataset.type;
    });
});
document.getElementById('btn-wd').click();

const speedSlider = document.getElementById('speed-slider');
const grToggle = document.getElementById('gr-toggle');
const cScaleSlider = document.getElementById('c-scale-slider');
const cScaleDisplay = document.getElementById('c-scale-display');

function updateSettings() {
    const scale = Math.pow(10, parseFloat(cScaleSlider.value));
    cScaleDisplay.innerText = scale.toFixed(3) + 'x';
    cScaleSlider.disabled = !grToggle.checked;
    if (physics) physics.updateSettings({ enableGR: grToggle.checked, cScale: scale });
}
grToggle.addEventListener('change', updateSettings);
cScaleSlider.addEventListener('input', updateSettings);

// 放置天體
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
window.addEventListener('click', e => {
    if (interactionMode !== 'place') return;
    if (e.target.closest('#ui-layer > div')) return;
    mouse.x = (e.clientX/innerWidth)*2-1;
    mouse.y = -(e.clientY/innerHeight)*2+1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObject(hitPlane);
    if (hits.length > 0) {
        const hp = hits[0].point;
        const tmpl = EXTREME[selectedExtremeType];
        let tM=0, cx=0, cz=0;
        mainThreadBodies.forEach(b => { tM+=b.m; cx+=b.m*b.x; cz+=b.m*b.z; });
        cx/=tM; cz/=tM;
        const rdx=hp.x-cx, rdz=hp.z-cz, r=Math.sqrt(rdx*rdx+rdz*rdz);
        const vc = r>0.05 ? Math.sqrt(G_MAIN*tM/r) : 0;
        const vx=-vc*(rdz/r), vz=vc*(rdx/r);
        const body = { m:tmpl.m, x:hp.x, y:0, z:hp.z, vx, vy:0, vz, ax:0, ay:0, az:0 };
        mainThreadBodies.push({ m:tmpl.m, x:hp.x, z:hp.z });
        physics.addBody(body);
        createBodyVisual({ name:tmpl.name, m:tmpl.m, x:hp.x, y:0, z:hp.z, color:tmpl.color, radius:tmpl.radius });
        if (selectedExtremeType==='bh') {
            const ring = new THREE.Mesh(new THREE.TorusGeometry(0.5,0.05,16,100), new THREE.MeshBasicMaterial({color:0xffaa00,side:THREE.DoubleSide}));
            ring.rotation.x = Math.PI/2;
            meshes[meshes.length-1].add(ring);
        }
    }
});

window.addEventListener('resize', () => {
    camera.aspect = innerWidth/innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
});

// ────────────────── 6. 渲染迴圈 ──────────────────
function updateVisuals(positions, masses) {
    for (let i = 0; i < meshes.length; i++) {
        // 跳過已消亡的天體
        if (masses && masses[i] === 0) {
            meshes[i].visible = false;
            trails[i].line.visible = false;
            if (i < mainThreadBodies.length) mainThreadBodies[i].m = 0;
            continue;
        }
        const px=positions[i*3], py=positions[i*3+1], pz=positions[i*3+2];
        meshes[i].position.set(px, py, pz);
        const t = trails[i]; t.skip++;
        if (t.skip > 2) {
            t.skip = 0;
            for (let j = MAX_TRAIL-1; j > 0; j--) {
                t.positions[j*3]=t.positions[(j-1)*3];
                t.positions[j*3+1]=t.positions[(j-1)*3+1];
                t.positions[j*3+2]=t.positions[(j-1)*3+2];
            }
            t.positions[0]=px; t.positions[1]=py; t.positions[2]=pz;
            if (t.count < MAX_TRAIL) t.count++;
            t.line.geometry.setDrawRange(0, t.count);
            t.line.geometry.attributes.position.needsUpdate = true;
        }
    }
}

let pending = false;

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    
    // 移除 PHYSICS_EVERY_N_FRAMES 限制，讓 GPU 計算發揮全力，達到 60Hz 以上的滑順視覺
    if (!pending && physics) {
        pending = true;
        const steps = parseInt(speedSlider.value);
        physics.step(steps).then(result => {
            updateVisuals(result.positions, result.masses);
            pending = false;
        });
    }
    renderer.render(scene, camera);
}

// ────────────────── 啟動 ──────────────────
initPhysics().then(() => {
    // 在 UI 顯示後端資訊
    const desc = document.querySelector('#ui-content p');
    if (desc) desc.innerHTML = `物理後端：<b class="text-green-400">${backendName}</b>｜Velocity Verlet 積分器<br>單位：AU / M☉ / 年`;
    animate();
});
