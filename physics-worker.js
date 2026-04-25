// ═══════════════════════════════════════════════════════════════
// N-body 物理引擎 — Web Worker (CPU fallback)
// Velocity Verlet + 自適應步長 + 可選 1PN EIH GR 修正
// ═══════════════════════════════════════════════════════════════

const G = 4 * Math.pow(Math.PI, 2);       // AU³/(M☉·yr²)
const EPSILON_SQ = 1e-10;
const TRUE_C = 63239.7263;                 // AU/yr

let bodies = [];
const BASE_DT = 0.0001;
const MIN_DT  = 1e-8;
const ETA     = 0.03;
let nextSafeDt = BASE_DT;
let enableGR = false;
let cValue = TRUE_C;
let C2 = cValue * cValue;

self.onmessage = function(e) {
    const data = e.data;
    if (data.type === 'init') {
        bodies = data.bodies;
        computeAccelerations();
    } else if (data.type === 'update_settings') {
        enableGR = data.enableGR;
        cValue = TRUE_C * data.cScale;
        C2 = cValue * cValue;
    } else if (data.type === 'add') {
        bodies.push(data.body);
        computeAccelerations();
    } else if (data.type === 'step') {
        for (let s = 0; s < data.steps; s++) verletStep();
        const positions = new Float32Array(bodies.length * 3);
        const masses = new Float32Array(bodies.length);
        for (let i = 0; i < bodies.length; i++) {
            positions[i*3]   = bodies[i].x;
            positions[i*3+1] = bodies[i].y;
            positions[i*3+2] = bodies[i].z;
            masses[i]        = bodies[i].m;
        }
        self.postMessage({ type: 'update', positions, masses }, [positions.buffer, masses.buffer]);
    }
};

function computeAccelerations() {
    const N = bodies.length;
    for (let i = 0; i < N; i++) { bodies[i].ax = 0; bodies[i].ay = 0; bodies[i].az = 0; }
    let safeDt = BASE_DT;

    for (let i = 0; i < N; i++) {
        if (bodies[i].m === 0) continue;
        for (let j = i + 1; j < N; j++) {
            const bi = bodies[i], bj = bodies[j];
            if (bj.m === 0) continue;
            const dx = bj.x - bi.x, dy = bj.y - bi.y, dz = bj.z - bi.z;
            const distSq = dx*dx + dy*dy + dz*dz + EPSILON_SQ;
            const dist = Math.sqrt(distSq);

            // 黑洞吸收邏輯 (Black Hole Absorption) — 含質量與動量守恆
            const Rs = 2.0 * G * Math.max(bi.m, bj.m) / C2;
            const R_merge = Math.max(Rs * 1.5, 0.02);
            if (dist < R_merge) {
                let survivor, victim;
                if (bi.m <= bj.m) { survivor = bj; victim = bi; }
                else                { survivor = bi; victim = bj; }
                const totalM = survivor.m + victim.m;
                survivor.vx = (survivor.m*survivor.vx + victim.m*victim.vx) / totalM;
                survivor.vy = (survivor.m*survivor.vy + victim.m*victim.vy) / totalM;
                survivor.vz = (survivor.m*survivor.vz + victim.m*victim.vz) / totalM;
                survivor.m = totalM;
                victim.m = 0; victim.x = 1e12; victim.y = 1e12; victim.z = 1e12;
                victim.vx = 0; victim.vy = 0; victim.vz = 0;
                victim.ax = 0; victim.ay = 0; victim.az = 0;
                continue;
            }

            // 雙準則自適應步長
            const t_dyn = ETA * Math.sqrt(dist * distSq / (G * (bi.m + bj.m)));
            if (t_dyn < safeDt) safeDt = t_dyn;
            const dvx = bi.vx-bj.vx, dvy = bi.vy-bj.vy, dvz = bi.vz-bj.vz;
            const vRelSq = dvx*dvx + dvy*dvy + dvz*dvz;
            if (vRelSq > 1e-30) { const t_vel = ETA * dist / Math.sqrt(vRelSq); if (t_vel < safeDt) safeDt = t_vel; }

            const f = G / (distSq * dist);
            let axi = f*bj.m*dx, ayi = f*bj.m*dy, azi = f*bj.m*dz;
            let axj = -f*bi.m*dx, ayj = -f*bi.m*dy, azj = -f*bi.m*dz;

            // Paczyński-Wiita pseudo-Newtonian potential (strong-field GR)
            // Replaces Newton with F = Gm·r / [|r|·(|r| - rs)²], always attractive
            if (enableGR) {
                // Force on i due to j: rs based on mj
                const rs_j = 2 * G * bj.m / C2;
                const dr_j = Math.max(dist - rs_j, rs_j * 0.05 + 1e-10);
                const pw_j = G * bj.m / (dist * dr_j * dr_j);
                axi = pw_j * dx; ayi = pw_j * dy; azi = pw_j * dz;

                // Force on j due to i: rs based on mi
                const rs_i = 2 * G * bi.m / C2;
                const dr_i = Math.max(dist - rs_i, rs_i * 0.05 + 1e-10);
                const pw_i = G * bi.m / (dist * dr_i * dr_i);
                axj = -pw_i * dx; ayj = -pw_i * dy; azj = -pw_i * dz;
            }
            bi.ax+=axi; bi.ay+=ayi; bi.az+=azi;
            bj.ax+=axj; bj.ay+=ayj; bj.az+=azj;
        }
    }
    let targetDt = Math.max(safeDt, MIN_DT);
    nextSafeDt = Math.max(nextSafeDt * 0.5, Math.min(targetDt, nextSafeDt * 1.1));
}

function verletStep() {
    const N = bodies.length;
    const dt = nextSafeDt;
    for (let i = 0; i < N; i++) {
        const b = bodies[i];
        if (b.m === 0) continue;
        b.vx += 0.5*b.ax*dt; b.vy += 0.5*b.ay*dt; b.vz += 0.5*b.az*dt;
        b.x += b.vx*dt; b.y += b.vy*dt; b.z += b.vz*dt;
        if (!isFinite(b.x)||!isFinite(b.vx)) { b.m=0; b.x=1e12; b.y=1e12; b.z=1e12; b.vx=b.vy=b.vz=0; b.ax=b.ay=b.az=0; }
    }
    computeAccelerations();
    for (let i = 0; i < N; i++) {
        const b = bodies[i];
        if (b.m === 0) continue;
        b.vx += 0.5*b.ax*dt; b.vy += 0.5*b.ay*dt; b.vz += 0.5*b.az*dt;
    }
}
