// ═══════════════════════════════════════════════════════════════
// main.js — Three.js 渲染 + UI + 統一物理後端介面
// ═══════════════════════════════════════════════════════════════
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ────────────────── 1. 太陽系初始資料 ──────────────────
const TWO_PI = 2 * Math.PI;
const G_MAIN = 4 * Math.PI * Math.PI;
const DEG = Math.PI / 180;

// ── 軌道力學工具函式 ──
// 圓軌道 → 直角座標 (x-z = 黃道面, y = 北黃極)
function circOrbit(a, i_deg, Omega_deg, nu_deg, M_central) {
    const i = i_deg * DEG, O = Omega_deg * DEG, nu = nu_deg * DEG;
    const v = TWO_PI * Math.sqrt(M_central / a);
    const cO = Math.cos(O), sO = Math.sin(O);
    const ci = Math.cos(i), si = Math.sin(i);
    const cn = Math.cos(nu), sn = Math.sin(nu);
    return {
        x: (cO * cn - sO * ci * sn) * a,
        y: (si * sn) * a,
        z: (sO * cn + cO * ci * sn) * a,
        vx: (-cO * sn - sO * ci * cn) * v,
        vy: (si * cn) * v,
        vz: (-sO * sn + cO * ci * cn) * v,
    };
}
// 偏心軌道近日點 → 直角座標
function periOrbit(a, e, i_deg, Omega_deg, M_central) {
    const rp = a * (1 - e);
    const vp = TWO_PI * Math.sqrt(M_central * (1 + e) / (a * (1 - e)));
    const i = i_deg * DEG, O = Omega_deg * DEG;
    const cO = Math.cos(O), sO = Math.sin(O);
    const ci = Math.cos(i), si = Math.sin(i);
    return { x: cO * rp, y: 0, z: sO * rp, vx: -sO * ci * vp, vy: si * vp, vz: cO * ci * vp };
}
// 衛星座標 = 母體座標 + 相對軌道座標
function moonState(parent, a_moon, i_deg, Omega_deg, nu_deg, M_parent) {
    const rel = circOrbit(a_moon, i_deg, Omega_deg, nu_deg, M_parent);
    return {
        x: parent.x + rel.x, y: parent.y + rel.y, z: parent.z + rel.z,
        vx: parent.vx + rel.vx, vy: parent.vy + rel.vy, vz: parent.vz + rel.vz,
    };
}

// ── 行星資料 (含真實軌道傾角) ──
const M_SUN = 1.0;
const planetDefs = [
    // name, mass, a, i(°), Ω(°), ν(°), color, radius
    // 視覺半徑刻意縮小，確保衛星軌道在母體球體之外；遠距可見性由 updateVisuals 的動態縮放保障
    ['太陽', 1.0, 0, 0, 0, 0, 0xffdd00, 0.015],
    ['水星', 1.65e-7, 0.387, 7.005, 48.33, 0, 0xaaaaaa, 0.001],
    ['金星', 2.45e-6, 0.723, 3.395, 76.68, 60, 0xffcc88, 0.0015],
    ['地球', 3.00e-6, 1.000, 0.000, 0.00, 0, 0x4488ff, 0.0015],
    ['火星', 3.20e-7, 1.524, 1.850, 49.56, 120, 0xff5533, 0.001],
    ['木星', 9.50e-4, 5.200, 1.303, 100.46, 80, 0xddaa77, 0.003],
    ['土星', 2.80e-4, 9.580, 2.485, 113.67, 200, 0xeecc99, 0.0025],
    ['天王星', 4.37e-5, 19.200, 0.773, 74.01, 280, 0x66ccff, 0.002],
    ['海王星', 5.00e-5, 30.050, 1.770, 131.78, 330, 0x3366ff, 0.002],
];

// ── 衛星資料 ──
// [name, parentIdx, a(AU), mass(M☉), i(°), Ω(°), ν(°), color, radius]
const moonDefs = [
    // 地球系
    ['月球', 3, 2.570e-3, 3.69e-8, 5.145, 125.0, 0, 0xcccccc, 0.0008],
    // 木星系 (排除 Io 以維持效能)
    ['Europa', 5, 4.485e-3, 2.41e-8, 1.79, 0, 0, 0xccddff, 0.0008],
    ['Ganymede', 5, 7.155e-3, 7.45e-8, 2.21, 0, 120, 0xbbaa88, 0.001],
    ['Callisto', 5, 1.259e-2, 5.41e-8, 2.02, 0, 240, 0x887766, 0.0008],
    // 土星系
    ['Titan', 6, 8.168e-3, 6.76e-8, 27.0, 0, 0, 0xff8833, 0.001],
    ['Rhea', 6, 3.522e-3, 1.16e-9, 27.0, 0, 180, 0xddddcc, 0.0006],
    // 天王星系 (軌道近垂直黃道面, i≈97.8°)
    ['Titania', 7, 2.917e-3, 1.76e-9, 97.8, 0, 0, 0xaabbcc, 0.0006],
    ['Oberon', 7, 3.900e-3, 1.46e-9, 97.8, 0, 180, 0x998877, 0.0006],
    // 海王星系 (逆行軌道, i≈157°)
    ['Triton', 8, 2.371e-3, 1.08e-8, 157.0, 0, 0, 0xaaddff, 0.0008],
];

// ── 矮行星資料 ──
// [name, a(AU), e, mass(M☉), i(°), Ω(°), color, radius]
const dwarfDefs = [
    ['冥王星', 39.48, 0.250, 6.58e-9, 17.16, 110.30, 0xddbb88, 0.001],
    ['Ceres', 2.77, 0.076, 4.72e-10, 10.59, 80.33, 0x999999, 0.0008],
    ['Eris', 67.67, 0.440, 8.35e-9, 44.04, 35.87, 0xeeeeee, 0.001],
    ['Haumea', 43.22, 0.195, 2.01e-9, 28.21, 121.90, 0xddccbb, 0.0008],
    ['Makemake', 45.51, 0.161, 1.56e-9, 29.01, 79.42, 0xcc8866, 0.0008],
];

// ── 組裝初始資料 ──
const initialBodiesData = [];

// 1) 行星
const planetStates = []; // 暫存行星狀態供衛星參考
for (const [name, m, a, i, O, nu, color, radius] of planetDefs) {
    let state;
    if (a === 0) { // 太陽
        state = { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0 };
    } else {
        state = circOrbit(a, i, O, nu, M_SUN);
    }
    planetStates.push({ ...state, m });
    initialBodiesData.push({ name, m, ...state, color, radius });
}

// 2) 衛星
for (const [name, pi, a, m, i, O, nu, color, radius] of moonDefs) {
    const parent = planetStates[pi];
    const state = moonState(parent, a, i, O, nu, parent.m);
    initialBodiesData.push({ name, m, ...state, color, radius });
}

// 3) 矮行星
const plutoIdx = initialBodiesData.length; // 記住冥王星索引供 Charon 使用
for (const [name, a, e, m, i, O, color, radius] of dwarfDefs) {
    const state = periOrbit(a, e, i, O, M_SUN);
    initialBodiesData.push({ name, m, ...state, color, radius });
}

// 4) Charon (繞冥王星的衛星, 傾角 ≈119.6°)
{
    const pluto = initialBodiesData[plutoIdx];
    const parentState = { x: pluto.x, y: pluto.y, z: pluto.z, vx: pluto.vx, vy: pluto.vy, vz: pluto.vz };
    const charonState = moonState(parentState, 1.313e-4, 119.6, 0, 0, pluto.m);
    initialBodiesData.push({ name: 'Charon', m: 8.04e-10, ...charonState, color: 0xaaaaaa, radius: 0.0006 });
}

// 準備物理狀態 + 歸零質心速度
const physicsState = initialBodiesData.map(b => ({ m: b.m, x: b.x, y: b.y, z: b.z, vx: b.vx, vy: b.vy, vz: b.vz, ax: 0, ay: 0, az: 0 }));
{
    let tM = 0, px = 0, py = 0, pz = 0;
    physicsState.forEach(b => { tM += b.m; px += b.m * b.vx; py += b.m * b.vy; pz += b.m * b.vz; });
    const cx = px / tM, cy = py / tM, cz = pz / tM;
    physicsState.forEach(b => { b.vx -= cx; b.vy -= cy; b.vz -= cz; });
}

let mainThreadBodies = initialBodiesData.map(b => ({ name: b.name, m: b.m, x: b.x, y: b.y || 0, z: b.z }));

// ────────────────── 2. Three.js 場景 ──────────────────
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x000000, 0.008);
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 1000);
camera.position.set(0, 15, 20);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
container.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.05; controls.maxDistance = 400;
scene.add(new THREE.GridHelper(200, 200, 0x333333, 0x111111));
const planeGeo = new THREE.PlaneGeometry(500, 500); planeGeo.rotateX(-Math.PI / 2);
const hitPlane = new THREE.Mesh(planeGeo, new THREE.MeshBasicMaterial({ visible: false }));
scene.add(hitPlane);

// ────────────────── 3. 天體視覺 ──────────────────
const meshes = [], trails = [], MAX_TRAIL = 300;

function createTextSprite(msg, col) {
    const c = document.createElement('canvas'); c.width = 256; c.height = 128;
    const ctx = c.getContext('2d');
    ctx.font = 'bold 32px sans-serif'; ctx.textAlign = 'center';
    ctx.fillStyle = col; ctx.shadowColor = 'black'; ctx.shadowBlur = 4;
    ctx.fillText(msg, 128, 64);
    const tex = new THREE.CanvasTexture(c);
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
    sp.scale.set(2, 1, 1); sp.renderOrder = 999;
    return sp;
}

function createBodyVisual(d) {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 32), new THREE.MeshBasicMaterial({ color: d.color }));
    mesh.position.set(d.x, d.y || 0, d.z);
    mesh.scale.setScalar(d.radius); // 使用 scale 控制大小，便於動態縮放
    mesh.userData.baseRadius = d.radius;
    const sp = createTextSprite(d.name, '#' + d.color.toString(16).padStart(6, '0'));
    sp.position.y = 1.15; // 在單位球上方，隨 mesh.scale 自動縮放
    sp.userData.baseLabelScale = [2, 1, 1];
    mesh.add(sp);
    const tGeo = new THREE.BufferGeometry();
    const tPos = new Float32Array(MAX_TRAIL * 3);
    tGeo.setAttribute('position', new THREE.BufferAttribute(tPos, 3));
    tGeo.setDrawRange(0, 0);
    const trail = new THREE.Line(tGeo, new THREE.LineBasicMaterial({ color: d.color, transparent: true, opacity: 0.4, blending: THREE.AdditiveBlending }));
    scene.add(mesh); scene.add(trail);
    meshes.push(mesh);
    trails.push({ line: trail, positions: tPos, count: 0, skip: 0 });
}
initialBodiesData.forEach(createBodyVisual);

// ────────────────── 4. 物理後端 (WebGPU 優先, Worker fallback) ──────────────────

// Worker fallback 封裝
class WorkerPhysics {
    constructor() { this.worker = null; this._resolve = null; this._addResolve = null; }
    async init(bodies) {
        this.worker = new Worker('physics-worker.js');
        // 等待 Worker 完成初始化（含加速度計算）後再繼續
        await new Promise(resolve => {
            this.worker.onmessage = e => {
                if (e.data.type === 'ready') resolve();
            };
            this.worker.postMessage({ type: 'init', bodies });
        });
        // 切換到正常訊息處理器
        this.worker.onmessage = e => {
            if (e.data.type === 'update' && this._resolve) {
                this._resolve({
                    positions: new Float32Array(e.data.positions),
                    velocities: e.data.velocities ? new Float32Array(e.data.velocities) : null,
                    masses: e.data.masses ? new Float32Array(e.data.masses) : null
                });
                this._resolve = null;
            } else if (e.data.type === 'added' && this._addResolve) {
                this._addResolve();
                this._addResolve = null;
            }
        };
    }
    step(n) { return new Promise(r => { this._resolve = r; this.worker.postMessage({ type: 'step', steps: n }); }); }
    addBody(b) { return new Promise(r => { this._addResolve = r; this.worker.postMessage({ type: 'add', body: b }); }); }
    updateSettings(s) { this.worker.postMessage({ type: 'update_settings', ...s }); }
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
        } catch (e) {
            console.warn('⚠️ WebGPU init failed, falling back to Worker:', e);
        }
    }
    try {
        physics = new WorkerPhysics();
        await physics.init(physicsState);
        backendName = 'CPU Worker';
        console.log('✅ CPU Worker physics initialized (fallback)');
    } catch (e) {
        console.error('❌ All physics backends failed:', e);
        const desc = document.querySelector('#ui-content p');
        if (desc) desc.innerHTML = '❌ <span class="text-red-400">物理引擎初始化失敗</span>，請檢查瀏覽器相容性或重新整理頁面。';
        backendName = 'None';
        throw e;
    }
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
    if (mode === 'view') {
        modeViewBtn.className = 'flex-1 py-1.5 px-2 bg-blue-600 rounded-lg text-sm font-bold border border-blue-400 transition-colors';
        modePlaceBtn.className = 'flex-1 py-1.5 px-2 bg-white/10 rounded-lg text-sm font-bold border border-white/20 hover:bg-white/20 transition-colors';
        tipText.innerText = '💡 目前為「觀察模式」：可左鍵拖曳旋轉、右鍵平移、滾輪縮放視角';
    } else {
        modePlaceBtn.className = 'flex-1 py-1.5 px-2 bg-blue-600 rounded-lg text-sm font-bold border border-blue-400 transition-colors';
        modeViewBtn.className = 'flex-1 py-1.5 px-2 bg-white/10 rounded-lg text-sm font-bold border border-white/20 hover:bg-white/20 transition-colors';
        tipText.innerText = '💡 目前為「放置模式」：點擊網格即可放置天體';
    }
}
modeViewBtn.addEventListener('click', () => setMode('view'));
modePlaceBtn.addEventListener('click', () => setMode('place'));
setMode('view');

let selectedExtremeType = 'wd';
const EXTREME = {
    'wd': { name: '白矮星', m: 1.2, color: 0xffffff, radius: 0.05 },
    'ns': { name: '中子星', m: 2.0, color: 0x00ffff, radius: 0.03 },
    'bh': { name: '黑洞', m: 50.0, color: 0xaa00ff, radius: 0.2 },
};
document.querySelectorAll('.celestial-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.celestial-btn').forEach(b => b.classList.remove('bg-white/30', 'border-white'));
        btn.classList.add('bg-white/30', 'border-white');
        selectedExtremeType = btn.dataset.type;
    });
});
document.getElementById('btn-wd').click();

const speedSlider = document.getElementById('speed-slider');
const grToggle = document.getElementById('gr-toggle');
const cScaleSlider = document.getElementById('c-scale-slider');
const cScaleDisplay = document.getElementById('c-scale-display');

const cameraTargetSelect = document.getElementById('camera-target');

function updateCameraOptions() {
    if (!cameraTargetSelect) return;
    const currentVal = cameraTargetSelect.value;
    cameraTargetSelect.innerHTML = '<option value="none" class="bg-gray-800">自由視角 (Free)</option>';
    // 定義分組：[groupLabel, startIdx, endIdx]（依據 initialBodiesData 的順序）
    const nPlanets = planetDefs.length;
    const nMoons = moonDefs.length;
    const nDwarfs = dwarfDefs.length;
    const groups = [
        ['☀ 恆星與行星', 0, nPlanets],
        ['🌙 衛星', nPlanets, nPlanets + nMoons],
        ['🪨 矮行星', nPlanets + nMoons, nPlanets + nMoons + nDwarfs + 1], // +1 for Charon
    ];
    for (const [label, start, end] of groups) {
        const grp = document.createElement('optgroup');
        grp.label = label;
        grp.className = 'bg-gray-800';
        let hasItems = false;
        for (let i = start; i < Math.min(end, mainThreadBodies.length); i++) {
            if (mainThreadBodies[i].m > 0) {
                const opt = document.createElement('option');
                opt.value = i;
                opt.className = 'bg-gray-800';
                opt.innerText = mainThreadBodies[i].name || `Body ${i}`;
                grp.appendChild(opt);
                hasItems = true;
            }
        }
        // 動態新增的天體（超出預定義範圍）
        if (label === '🪨 矮行星') {
            for (let i = end; i < mainThreadBodies.length; i++) {
                if (mainThreadBodies[i].m > 0) {
                    const opt = document.createElement('option');
                    opt.value = i;
                    opt.className = 'bg-gray-800';
                    opt.innerText = mainThreadBodies[i].name || `Body ${i}`;
                    grp.appendChild(opt);
                    hasItems = true;
                }
            }
        }
        if (hasItems) cameraTargetSelect.appendChild(grp);
    }
    if (cameraTargetSelect.querySelector(`option[value="${currentVal}"]`)) {
        cameraTargetSelect.value = currentVal;
    } else {
        cameraTargetSelect.value = 'none';
    }
}
updateCameraOptions();

let initialEnergy = null;
function calcSystemEnergy(positions, velocities, masses) {
    if (!velocities) return;
    let K = 0, U = 0;
    const G = 4 * Math.PI * Math.PI;
    const C2 = 63239.7263 * 63239.7263 * Math.pow(10, parseFloat(cScaleSlider.value) * 2);
    const useGR = grToggle.checked;

    const count = Math.floor(positions.length / 3);
    for (let i = 0; i < count; i++) {
        if (masses[i] === 0) continue;
        const mi = masses[i];
        const v2 = velocities[i * 3] * velocities[i * 3] + velocities[i * 3 + 1] * velocities[i * 3 + 1] + velocities[i * 3 + 2] * velocities[i * 3 + 2];
        K += 0.5 * mi * v2;

        for (let j = i + 1; j < count; j++) {
            if (masses[j] === 0) continue;
            const mj = masses[j];
            const dx = positions[j * 3] - positions[i * 3];
            const dy = positions[j * 3 + 1] - positions[i * 3 + 1];
            const dz = positions[j * 3 + 2] - positions[i * 3 + 2];
            const d = Math.sqrt(dx * dx + dy * dy + dz * dz + 1e-10);

            if (useGR) {
                const rs = 2 * G * (mi + mj) / C2;
                const dr = Math.max(d - rs, rs * 0.05 + 1e-10);
                U -= G * mi * mj / dr;
            } else {
                U -= G * mi * mj / d;
            }
        }
    }

    const E = K + U;
    if (initialEnergy === null && E !== 0) initialEnergy = E;

    document.getElementById('energy-k').innerText = K.toExponential(4);
    document.getElementById('energy-u').innerText = U.toExponential(4);
    if (initialEnergy !== null && initialEnergy !== 0) {
        const drift = Math.abs((E - initialEnergy) / initialEnergy) * 100;
        const driftEl = document.getElementById('energy-drift');
        driftEl.innerText = drift.toFixed(6) + '%';
        if (drift > 1) driftEl.className = 'font-mono text-red-400';
        else if (drift > 0.01) driftEl.className = 'font-mono text-yellow-400';
        else driftEl.className = 'font-mono text-green-400';
    }
}

function spawnMergerVFX(pos) {
    const light = new THREE.PointLight(0xffaa00, 5, 10);
    light.position.copy(pos);
    scene.add(light);

    const geo = new THREE.SphereGeometry(0.2, 16, 16);
    const mat = new THREE.MeshBasicMaterial({ color: 0xffaa00, transparent: true, opacity: 1 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(pos);
    scene.add(mesh);

    const startTime = performance.now();
    function animateVFX() {
        const t = (performance.now() - startTime) / 500;
        if (t >= 1) {
            scene.remove(light);
            scene.remove(mesh);
            geo.dispose();
            mat.dispose();
            return;
        }
        mesh.scale.setScalar(1 + t * 5);
        mat.opacity = 1 - t;
        light.intensity = 5 * (1 - t);
        requestAnimationFrame(animateVFX);
    }
    animateVFX();
}

function updateSettings() {
    initialEnergy = null;
    const scale = Math.pow(10, parseFloat(cScaleSlider.value));
    cScaleDisplay.innerText = scale.toFixed(3) + 'x';
    cScaleSlider.disabled = !grToggle.checked;
    if (physics) physics.updateSettings({ enableGR: grToggle.checked, cScale: scale });
}
grToggle.addEventListener('change', updateSettings);
cScaleSlider.addEventListener('input', updateSettings);

// 放置天體 — 使用佇列避免與進行中的 step() 產生競態條件
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const pendingAdds = []; // 佇列：等待物理引擎空閒時再加入

window.addEventListener('click', e => {
    if (interactionMode !== 'place') return;
    if (e.target.closest('#ui-layer > div')) return;
    mouse.x = (e.clientX / innerWidth) * 2 - 1;
    mouse.y = -(e.clientY / innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObject(hitPlane);
    if (hits.length > 0) {
        const hp = hits[0].point;
        const tmpl = EXTREME[selectedExtremeType];
        let tM = 0, cx = 0, cz = 0;
        mainThreadBodies.forEach(b => { if (b.m > 0) { tM += b.m; cx += b.m * b.x; cz += b.m * b.z; } });
        if (tM > 0) { cx /= tM; cz /= tM; }
        const rdx = hp.x - cx, rdz = hp.z - cz, r = Math.sqrt(rdx * rdx + rdz * rdz);
        const vc = r > 0.05 ? Math.sqrt(G_MAIN * tM / r) : 0;
        const vx = -vc * (rdz / r), vz = vc * (rdx / r);
        const body = { m: tmpl.m, x: hp.x, y: 0, z: hp.z, vx, vy: 0, vz, ax: 0, ay: 0, az: 0 };
        const visual = { name: tmpl.name, m: tmpl.m, x: hp.x, y: 0, z: hp.z, color: tmpl.color, radius: tmpl.radius };
        const isBH = selectedExtremeType === 'bh';
        pendingAdds.push({ body, visual, isBH });
    }
});

// 在物理引擎空閒時處理待加入的天體
async function processPendingAdds() {
    while (pendingAdds.length > 0) {
        const { body, visual, isBH } = pendingAdds.shift();
        await physics.addBody(body);
        mainThreadBodies.push({ name: visual.name, m: body.m, x: body.x, y: body.y || 0, z: body.z });
        createBodyVisual(visual);
        updateCameraOptions();
        if (isBH) {
            const ring = new THREE.Mesh(new THREE.TorusGeometry(0.5, 0.05, 16, 100), new THREE.MeshBasicMaterial({ color: 0xffaa00, side: THREE.DoubleSide }));
            ring.rotation.x = Math.PI / 2;
            meshes[meshes.length - 1].add(ring);
        }
    }
    initialEnergy = null; // 重新計算基準能量，避免產生假漂移
}

window.addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
});

// ────────────────── 6. 渲染迴圈 ──────────────────
let mergerHappened = false;

function updateVisuals(positions, masses) {
    // 只更新 positions 陣列所涵蓋的天體數量，避免讀取超出範圍
    const posCount = Math.floor(positions.length / 3);
    const count = Math.min(meshes.length, posCount);
    for (let i = 0; i < count; i++) {
        // 跳過已消亡的天體（mass 確認為 0）
        if (masses && masses[i] === 0) {
            // 首次消亡時釋放 GPU 資源
            if (meshes[i].visible) {
                mergerHappened = true;
                const deathPos = new THREE.Vector3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
                if (deathPos.x < 1e11) {
                    spawnMergerVFX(deathPos);
                }
                meshes[i].traverse(child => {
                    if (child.geometry) child.geometry.dispose();
                    if (child.material) {
                        if (child.material.map) child.material.map.dispose();
                        child.material.dispose();
                    }
                });
                scene.remove(meshes[i]);
                trails[i].line.geometry.dispose();
                trails[i].line.material.dispose();
                scene.remove(trails[i].line);
            }
            meshes[i].visible = false;
            trails[i].line.visible = false;
            if (i < mainThreadBodies.length) {
                if (mainThreadBodies[i].m !== 0) {
                    mainThreadBodies[i].m = 0;
                    updateCameraOptions();
                }
            }
            continue;
        }
        const px = positions[i * 3], py = positions[i * 3 + 1], pz = positions[i * 3 + 2];
        // 防止 NaN / Infinity 座標導致視覺消失
        if (!isFinite(px) || !isFinite(py) || !isFinite(pz)) continue;
        // 同步位置與質量到 mainThreadBodies，確保放置新天體時使用當前位置
        if (i < mainThreadBodies.length) {
            mainThreadBodies[i].x = px;
            mainThreadBodies[i].y = py;
            mainThreadBodies[i].z = pz;
            if (masses) mainThreadBodies[i].m = masses[i];
        }
        meshes[i].position.set(px, py, pz);
        // 距離自適應縮放：保證天體在遠距時仍可見，近距時顯示真實比例
        const baseR = meshes[i].userData.baseRadius || 0.01;
        const camDist = camera.position.distanceTo(meshes[i].position);
        const minVisualR = camDist * 0.0025; // 保證最小螢幕尺寸
        const dynScale = Math.max(baseR, minVisualR);
        meshes[i].scale.setScalar(dynScale);
        // 標籤：維持恆定角大小 (constant angular size)，不受球體縮放影響
        const labelSp = meshes[i].children[0];
        if (labelSp) {
            const desiredSize = camDist * 0.06; // 螢幕上約 2% 寬度
            const s = desiredSize / dynScale;    // 抵消父層 scale
            labelSp.scale.set(2 * s, s, 1);
            labelSp.position.y = 1.0 + desiredSize * 0.3 / dynScale;
        }
        const t = trails[i]; t.skip++;
        if (t.skip > 2) {
            t.skip = 0;
            // 使用原生 copyWithin 取代手動迴圈，效能更佳
            t.positions.copyWithin(3, 0, (MAX_TRAIL - 1) * 3);
            t.positions[0] = px; t.positions[1] = py; t.positions[2] = pz;
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

    if (!pending && physics) {
        // 先處理待加入的天體（確保物理與視覺同步後再進行下一步模擬）
        if (pendingAdds.length > 0) {
            pending = true;
            processPendingAdds().then(() => { pending = false; });
        } else {
            pending = true;
            const steps = parseInt(speedSlider.value);
            physics.step(steps).then(result => {
                updateVisuals(result.positions, result.masses);
                if (mergerHappened) {
                    initialEnergy = null; // 合併屬非彈性碰撞，總能量會折損，必須重設基準點
                    mergerHappened = false;
                }
                if (result.velocities) calcSystemEnergy(result.positions, result.velocities, result.masses);
                pending = false;
            });
        }
    }

    if (cameraTargetSelect && cameraTargetSelect.value !== 'none') {
        const targetIdx = parseInt(cameraTargetSelect.value);
        if (mainThreadBodies[targetIdx] && mainThreadBodies[targetIdx].m > 0) {
            const tx = mainThreadBodies[targetIdx].x;
            const ty = mainThreadBodies[targetIdx].y;
            const tz = mainThreadBodies[targetIdx].z;
            controls.target.lerp(new THREE.Vector3(tx, ty, tz), 0.1);
        } else {
            cameraTargetSelect.value = 'none';
        }
    }

    renderer.render(scene, camera);
}

// ────────────────── 啟動 ──────────────────
initPhysics().then(() => {
    const desc = document.querySelector('#ui-content p');
    if (desc) desc.innerHTML = `物理後端：<b class="text-green-400">${backendName}</b>｜Velocity Verlet 積分器<br>單位：AU / M☉ / 年`;
    animate();
}).catch(() => {
    // 初始化失敗已在上方函數中顯示錯誤訊息
});
