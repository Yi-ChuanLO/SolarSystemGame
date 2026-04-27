// ═══════════════════════════════════════════════════════════════
// N-body 物理引擎 — Web Worker (CPU fallback)
// Velocity Verlet + 自適應步長 + 可選 Paczyński-Wiita GR 修正 (ISCO @ 3Rs)
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
        self.postMessage({ type: 'ready' });
    } else if (data.type === 'update_settings') {
        enableGR = data.enableGR;
        cValue = TRUE_C * data.cScale;
        C2 = cValue * cValue;
    } else if (data.type === 'add') {
        bodies.push(data.body);
        computeAccelerations();
        self.postMessage({ type: 'added' });
    } else if (data.type === 'step') {
        for (let s = 0; s < data.steps; s++) verletStep();
        const positions = new Float32Array(bodies.length * 3);
        const velocities = new Float32Array(bodies.length * 3);
        const masses = new Float32Array(bodies.length);
        for (let i = 0; i < bodies.length; i++) {
            positions[i*3]   = bodies[i].x;
            positions[i*3+1] = bodies[i].y;
            positions[i*3+2] = bodies[i].z;
            velocities[i*3]  = bodies[i].vx;
            velocities[i*3+1]= bodies[i].vy;
            velocities[i*3+2]= bodies[i].vz;
            masses[i]        = bodies[i].m;
        }
        self.postMessage({ type: 'update', positions, velocities, masses }, [positions.buffer, velocities.buffer, masses.buffer]);
    }
};

function computeAccelerations() {
    const N = bodies.length;
    for (let i = 0; i < N; i++) { bodies[i].ax = 0; bodies[i].ay = 0; bodies[i].az = 0; }
    let safeDt = BASE_DT;

    // ── Phase 1: 計算所有力 + 偵測合併 (基於一致的狀態快照) ──
    const mergeTargets = new Int32Array(N);
    mergeTargets.fill(-1);

    for (let i = 0; i < N; i++) {
        if (bodies[i].m === 0) continue;
        for (let j = i + 1; j < N; j++) {
            const bi = bodies[i], bj = bodies[j];
            if (bj.m === 0) continue;
            const dx = bj.x - bi.x, dy = bj.y - bi.y, dz = bj.z - bi.z;
            const distSq = dx*dx + dy*dy + dz*dz + EPSILON_SQ;
            const dist = Math.sqrt(distSq);

            // 合併偵測 — 標記但不立即執行，確保力計算一致性
            const Rs = 2.0 * G * (bi.m + bj.m) / C2;
            const R_merge = Math.max(3.0 * Rs, Math.sqrt(EPSILON_SQ));
            if (dist < R_merge) {
                if (bi.m < bj.m) { if (mergeTargets[i] < 0) mergeTargets[i] = j; }
                else             { if (mergeTargets[j] < 0) mergeTargets[j] = i; }
                continue; // 合併中的天體跳過力計算
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

            // 修正：對稱版 Paczyński-Wiita 勢能 (Symmetric PW potential)
            // 使用系統總質量 (mi+mj) 計算 Schwarzschild 半徑，確保符合牛頓第三運動定律 (動量守恆)
            if (enableGR) {
                const rs = 2 * G * (bi.m + bj.m) / C2;
                const dr = Math.max(dist - rs, rs * 0.05 + 1e-10);
                const pw_f = G / (dist * dr * dr);
                axi = pw_f * bj.m * dx; ayi = pw_f * bj.m * dy; azi = pw_f * bj.m * dz;
                axj = -pw_f * bi.m * dx; ayj = -pw_f * bi.m * dy; azj = -pw_f * bi.m * dz;
            }
            bi.ax+=axi; bi.ay+=ayi; bi.az+=azi;
            bj.ax+=axj; bj.ay+=ayj; bj.az+=azj;
        }
    }

    // ── Phase 2: 解析合併鏈 (A→B→C 展開至鏈末端存活者) ──
    for (let i = 0; i < N; i++) {
        if (mergeTargets[i] < 0) continue;
        let target = mergeTargets[i];
        let maxIter = N;
        while (mergeTargets[target] >= 0 && maxIter-- > 0) target = mergeTargets[target];
        mergeTargets[i] = target;
    }

    // ── Phase 3: 執行合併 (質量與動量守恆、質心守恆) ──
    for (let i = 0; i < N; i++) {
        if (mergeTargets[i] >= 0) continue; // 跳過被吞噬者
        if (bodies[i].m === 0) continue;
        let totalM = bodies[i].m;
        let tpx = bodies[i].m * bodies[i].vx;
        let tpy = bodies[i].m * bodies[i].vy;
        let tpz = bodies[i].m * bodies[i].vz;
        let tposx = bodies[i].m * bodies[i].x;
        let tposy = bodies[i].m * bodies[i].y;
        let tposz = bodies[i].m * bodies[i].z;
        for (let k = 0; k < N; k++) {
            if (mergeTargets[k] === i) {
                totalM += bodies[k].m;
                tpx += bodies[k].m * bodies[k].vx;
                tpy += bodies[k].m * bodies[k].vy;
                tpz += bodies[k].m * bodies[k].vz;
                tposx += bodies[k].m * bodies[k].x;
                tposy += bodies[k].m * bodies[k].y;
                tposz += bodies[k].m * bodies[k].z;
            }
        }
        if (totalM > bodies[i].m) {
            bodies[i].vx = tpx / totalM;
            bodies[i].vy = tpy / totalM;
            bodies[i].vz = tpz / totalM;
            bodies[i].x = tposx / totalM;
            bodies[i].y = tposy / totalM;
            bodies[i].z = tposz / totalM;
            bodies[i].m = totalM;
        }
    }

    // ── Phase 4: 清除被吞噬天體 ──
    for (let i = 0; i < N; i++) {
        if (mergeTargets[i] >= 0) {
            bodies[i].m = 0; // 保留 x,y,z 作為死亡座標供主線程讀取
            bodies[i].vx = 0; bodies[i].vy = 0; bodies[i].vz = 0;
            bodies[i].ax = 0; bodies[i].ay = 0; bodies[i].az = 0;
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
        if (!isFinite(b.x)||!isFinite(b.y)||!isFinite(b.z)||!isFinite(b.vx)||!isFinite(b.vy)||!isFinite(b.vz)) { b.m=0; b.x=1e12; b.y=1e12; b.z=1e12; b.vx=b.vy=b.vz=0; b.ax=b.ay=b.az=0; }
    }
    computeAccelerations();
    for (let i = 0; i < N; i++) {
        const b = bodies[i];
        if (b.m === 0) continue;
        b.vx += 0.5*b.ax*dt; b.vy += 0.5*b.ay*dt; b.vz += 0.5*b.az*dt;
    }
}
