// ═══════════════════════════════════════════════════════════════
// N-body 物理引擎 — Web Worker (CPU fallback)
// Velocity Verlet + 自適應步長 + 可選 Paczyński-Wiita GR 修正
// + KS 正則化 (Kustaanheimo-Stiefel Regularization)
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

// ────────────────── KS 正則化 (Kustaanheimo-Stiefel) ──────────────────
// 將密近雙星的 1/r² 奇異力轉換為 4D 諧振子，在虛擬時間 τ 中積分
// 參考: Stiefel & Scheifele (1971), Aarseth "Gravitational N-Body Simulations" (2003)
const KS_FACTOR = 20.0;   // 當 dist < KS_FACTOR * R_merge 時啟動正則化
const KS_MAX_SUB = 64;    // 虛擬時間子步上限
let ksActivePairs = [];    // 目前活躍的 KS 對 [{ia, ib}]

// 物理座標 → KS 座標: r=(rx,ry,rz) → u=[u1,u2,u3,u4]
// |r| = |u|² (奇異性消除的核心)
function physToKS(rx, ry, rz) {
    const r = Math.sqrt(rx*rx + ry*ry + rz*rz);
    if (r < 1e-30) return [0, 0, 0, 0];
    if (rx >= 0) {
        const p = Math.sqrt(2.0 * (r + rx));
        return [p / 2, ry / p, 0, rz / p];
    } else {
        const p = Math.sqrt(2.0 * (r - rx));
        return [ry / p, p / 2, rz / p, 0];
    }
}

// KS 座標 → 物理座標
function ksToPhy(u) {
    return [
        u[0]*u[0] - u[1]*u[1] - u[2]*u[2] + u[3]*u[3],  // rx
        2 * (u[0]*u[1] - u[2]*u[3]),                       // ry
        2 * (u[0]*u[2] + u[1]*u[3])                        // rz
    ];
}

// L(u)^T · v  (4×3 轉置矩陣 × 3維向量 → 4維向量)
function LtV(u, vx, vy, vz) {
    return [
         u[0]*vx + u[1]*vy + u[2]*vz,
        -u[1]*vx + u[0]*vy + u[3]*vz,
        -u[2]*vx - u[3]*vy + u[0]*vz,
         u[3]*vx - u[2]*vy + u[1]*vz
    ];
}

// L(u) · w  (3×4 矩陣 × 4維向量 → 3維向量)
function LuW(u, w) {
    return [
        u[0]*w[0] - u[1]*w[1] - u[2]*w[2] + u[3]*w[3],
        u[1]*w[0] + u[0]*w[1] - u[3]*w[2] - u[2]*w[3],
        u[2]*w[0] + u[3]*w[1] + u[0]*w[2] + u[1]*w[3]
    ];
}

// KS 子步積分：在虛擬時間 τ 中積分密近對 (ia, ib) 的相對運動
// 使用 Leapfrog 在 τ 空間中積分：u'' + (h/2)u = (r/2) L^T F_pert
function ksSubIntegrate(ia, ib, physDt) {
    const ba = bodies[ia], bb = bodies[ib];
    const mu = G * (ba.m + bb.m);
    const mRatio_a = bb.m / (ba.m + bb.m);  // 質量比：a 的位移比例
    const mRatio_b = ba.m / (ba.m + bb.m);

    // 質心
    const totalM = ba.m + bb.m;
    let cmx = (ba.m*ba.x + bb.m*bb.x) / totalM;
    let cmy = (ba.m*ba.y + bb.m*bb.y) / totalM;
    let cmz = (ba.m*ba.z + bb.m*bb.z) / totalM;
    let cmvx = (ba.m*ba.vx + bb.m*bb.vx) / totalM;
    let cmvy = (ba.m*ba.vy + bb.m*bb.vy) / totalM;
    let cmvz = (ba.m*ba.vz + bb.m*bb.vz) / totalM;

    // 相對座標
    let rx = bb.x - ba.x, ry = bb.y - ba.y, rz = bb.z - ba.z;
    let vrx = bb.vx - ba.vx, vry = bb.vy - ba.vy, vrz = bb.vz - ba.vz;

    // 物理 → KS
    let u = physToKS(rx, ry, rz);
    let r = u[0]*u[0] + u[1]*u[1] + u[2]*u[2] + u[3]*u[3]; // = |r|
    // u' = (1/2) L^T v_rel
    let up = LtV(u, vrx, vry, vrz);
    up[0] *= 0.5; up[1] *= 0.5; up[2] *= 0.5; up[3] *= 0.5;

    // 結合能 h = mu/r - v²/2  (h>0 = 束縛軌道)
    const v2 = vrx*vrx + vry*vry + vrz*vrz;
    let h = mu / Math.max(r, 1e-20) - 0.5 * v2;

    // 計算質心受到的外力加速度 + 相對運動擾動力
    let cmAx = 0, cmAy = 0, cmAz = 0;
    let fpx = 0, fpy = 0, fpz = 0;
    const N = bodies.length;
    for (let k = 0; k < N; k++) {
        if (k === ia || k === ib || bodies[k].m === 0) continue;
        const bk = bodies[k];
        // 對 a 的力
        let dxa = bk.x - ba.x, dya = bk.y - ba.y, dza = bk.z - ba.z;
        let dSqa = dxa*dxa + dya*dya + dza*dza + EPSILON_SQ, da = Math.sqrt(dSqa);
        let fa = G / (dSqa * da);
        // 對 b 的力
        let dxb = bk.x - bb.x, dyb = bk.y - bb.y, dzb = bk.z - bb.z;
        let dSqb = dxb*dxb + dyb*dyb + dzb*dzb + EPSILON_SQ, db = Math.sqrt(dSqb);
        let fb = G / (dSqb * db);
        if (enableGR) {
            const rsa = 2*G*(ba.m+bk.m)/C2;
            const dra = Math.max(da - rsa, rsa*0.05+1e-10);
            fa = G / (da * dra * dra);
            const rsb = 2*G*(bb.m+bk.m)/C2;
            const drb = Math.max(db - rsb, rsb*0.05+1e-10);
            fb = G / (db * drb * drb);
        }
        // 質心加速度
        cmAx += (ba.m * fa * bk.m * dxa + bb.m * fb * bk.m * dxb) / totalM;
        cmAy += (ba.m * fa * bk.m * dya + bb.m * fb * bk.m * dyb) / totalM;
        cmAz += (ba.m * fa * bk.m * dza + bb.m * fb * bk.m * dzb) / totalM;
        // 相對擾動: F_pert = a_b(ext) - a_a(ext)
        fpx += fb * bk.m * dxb - fa * bk.m * dxa;
        fpy += fb * bk.m * dyb - fa * bk.m * dya;
        fpz += fb * bk.m * dzb - fa * bk.m * dza;
    }

    // 虛擬時間步長估計: dτ ≈ dt / r
    const totalTau = physDt / Math.max(r, 1e-20);
    const nSub = Math.min(Math.max(Math.ceil(totalTau * Math.sqrt(Math.abs(h) + 1) * 4), 4), KS_MAX_SUB);
    const dTau = totalTau / nSub;

    // Leapfrog in τ-space: u'' + (h/2)u = (r/2) L^T F_pert
    let tPhys = 0;
    for (let s = 0; s < nSub; s++) {
        // KS 加速度
        const omega2 = h * 0.5;
        const ltf = LtV(u, fpx, fpy, fpz);
        let uddot = [0, 0, 0, 0];
        for (let d = 0; d < 4; d++) {
            uddot[d] = -omega2 * u[d] + 0.5 * r * ltf[d];
        }

        // Kick 1
        for (let d = 0; d < 4; d++) up[d] += 0.5 * dTau * uddot[d];
        // Drift
        for (let d = 0; d < 4; d++) u[d] += dTau * up[d];

        // 更新 r 和物理座標
        r = u[0]*u[0] + u[1]*u[1] + u[2]*u[2] + u[3]*u[3];
        tPhys += r * dTau;  // dt = r dτ

        // 更新 h (結合能隨擾動變化)
        const upDotLtf = up[0]*ltf[0] + up[1]*ltf[1] + up[2]*ltf[2] + up[3]*ltf[3];
        h -= 2.0 * dTau * upDotLtf;

        // 新的 KS 加速度 (用更新後的 u, r, h)
        const omega2_new = h * 0.5;
        const ltf_new = LtV(u, fpx, fpy, fpz);
        for (let d = 0; d < 4; d++) {
            uddot[d] = -omega2_new * u[d] + 0.5 * r * ltf_new[d];
        }
        // Kick 2
        for (let d = 0; d < 4; d++) up[d] += 0.5 * dTau * uddot[d];
    }

    // KS → 物理座標 (相對)
    const rNew = ksToPhy(u);
    // v_rel = 2 L(u) u' / r
    const vRel = LuW(u, up);
    const rMag = Math.max(r, 1e-20);
    const vrxN = 2 * vRel[0] / rMag;
    const vryN = 2 * vRel[1] / rMag;
    const vrzN = 2 * vRel[2] / rMag;

    // INT-2 修正：KS 物理時間校正
    // KS 子步積分的累計物理時間 tPhys 可能與目標 physDt 不完全吻合，
    // 因為 dt = r dτ 的映射未被精確強制。以一階修正 (v × Δt) 補償位置偏差。
    const timeError = tPhys - physDt;
    if (Math.abs(timeError) > 1e-15) {
        rNew[0] -= vrxN * timeError;
        rNew[1] -= vryN * timeError;
        rNew[2] -= vrzN * timeError;
    }

    // 用實際經過的物理時間更新質心
    cmx += cmvx * physDt + 0.5 * cmAx * physDt * physDt;
    cmy += cmvy * physDt + 0.5 * cmAy * physDt * physDt;
    cmz += cmvz * physDt + 0.5 * cmAz * physDt * physDt;
    cmvx += cmAx * physDt;
    cmvy += cmAy * physDt;
    cmvz += cmAz * physDt;

    // 從質心 + 相對座標還原個體位置
    ba.x = cmx - mRatio_a * rNew[0]; ba.y = cmy - mRatio_a * rNew[1]; ba.z = cmz - mRatio_a * rNew[2];
    bb.x = cmx + mRatio_b * rNew[0]; bb.y = cmy + mRatio_b * rNew[1]; bb.z = cmz + mRatio_b * rNew[2];
    ba.vx = cmvx - mRatio_a * vrxN; ba.vy = cmvy - mRatio_a * vryN; ba.vz = cmvz - mRatio_a * vrzN;
    bb.vx = cmvx + mRatio_b * vrxN; bb.vy = cmvy + mRatio_b * vryN; bb.vz = cmvz + mRatio_b * vrzN;
}

self.onmessage = function(e) {
    const data = e.data;
    if (data.type === 'init') {
        bodies = data.bodies;
        computeAccelerations();
        self.postMessage({ type: 'ready' });
    } else if (data.type === 'update_settings') {
        enableGR = data.enableGR;
        cValue = TRUE_C * (enableGR ? data.cScale : 1.0);
        C2 = cValue * cValue;
    } else if (data.type === 'add') {
        bodies.push(data.body);
        computeAccelerations();
        self.postMessage({ type: 'added', ok: true });
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
    ksActivePairs = [];  // 重置 KS 對列表
    const ksSet = new Set(); // 追蹤已經在 KS 對中的天體，防止重複加入

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
            const R_merge = Math.max((bi.radius || 0) + (bj.radius || 0), Math.max(3.0 * Rs, Math.sqrt(EPSILON_SQ)));
            if (dist < R_merge) {
                if (bi.m < bj.m) { if (mergeTargets[i] < 0) mergeTargets[i] = j; }
                else             { if (mergeTargets[j] < 0) mergeTargets[j] = i; }
                continue; // 合併中的天體跳過力計算
            }

            // KS 正則化區域偵測：dist 在合併半徑與 KS 閾值之間
            // 跳過直接力計算，改由 verletStep 中的 ksSubIntegrate 處理
            const R_ks = KS_FACTOR * R_merge;
            if (dist < R_ks) {
                if (!ksSet.has(i) && !ksSet.has(j)) {
                    ksActivePairs.push({ ia: i, ib: j });
                    ksSet.add(i);
                    ksSet.add(j);
                    continue; // KS 對跳過直接力計算，由子步積分處理
                }
                // 若已有天體在其他 KS 對中，則降級為直接力計算，仰賴自適應步長
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

    // ── Phase 5: 合併存活者加速度重算 ──
    // 避免 Kick 2 使用基於舊質量/位置的過時加速度
    for (let i = 0; i < N; i++) {
        if (mergeTargets[i] >= 0 || bodies[i].m === 0) continue;
        let isSurvivor = false;
        for (let k = 0; k < N; k++) { if (mergeTargets[k] === i) { isSurvivor = true; break; } }
        if (!isSurvivor) continue;
        bodies[i].ax = 0; bodies[i].ay = 0; bodies[i].az = 0;
        for (let j = 0; j < N; j++) {
            if (j === i || bodies[j].m === 0) continue;
            const dx = bodies[j].x - bodies[i].x, dy = bodies[j].y - bodies[i].y, dz = bodies[j].z - bodies[i].z;
            const dSq = dx*dx + dy*dy + dz*dz + EPSILON_SQ, d = Math.sqrt(dSq);
            if (enableGR) {
                const rs = 2*G*(bodies[i].m+bodies[j].m)/C2;
                const dr = Math.max(d - rs, rs*0.05 + 1e-10);
                const pf = G / (d * dr * dr);
                bodies[i].ax += pf*bodies[j].m*dx; bodies[i].ay += pf*bodies[j].m*dy; bodies[i].az += pf*bodies[j].m*dz;
            } else {
                const f = G / (dSq * d);
                bodies[i].ax += f*bodies[j].m*dx; bodies[i].ay += f*bodies[j].m*dy; bodies[i].az += f*bodies[j].m*dz;
            }
        }
    }

    let targetDt = Math.max(safeDt, MIN_DT);
    // 指數衰減平滑：縮小快 (α=0.7)、放大慢 (α=0.3)
    const prevDt = nextSafeDt;
    const ratio = targetDt / Math.max(prevDt, 1e-20);
    const alpha = ratio < 1.0 ? 0.7 : 0.3;
    nextSafeDt = Math.max(prevDt * Math.pow(ratio, alpha), MIN_DT);
    nextSafeDt = Math.min(nextSafeDt, prevDt * 2.0); // 成長上限 2x
}

function verletStep() {
    const N = bodies.length;
    const dt = nextSafeDt;

    // 標記 KS 對中的天體，跳過標準 Verlet
    const ksSet = new Set();
    for (const pair of ksActivePairs) { ksSet.add(pair.ia); ksSet.add(pair.ib); }

    // Kick 1 + Drift (非 KS 天體)
    for (let i = 0; i < N; i++) {
        const b = bodies[i];
        if (b.m === 0 || ksSet.has(i)) continue;
        b.vx += 0.5*b.ax*dt; b.vy += 0.5*b.ay*dt; b.vz += 0.5*b.az*dt;
        b.x += b.vx*dt; b.y += b.vy*dt; b.z += b.vz*dt;
        if (!isFinite(b.x)||!isFinite(b.y)||!isFinite(b.z)||!isFinite(b.vx)||!isFinite(b.vy)||!isFinite(b.vz)) { b.m=0; b.x=1e12; b.y=1e12; b.z=1e12; b.vx=b.vy=b.vz=0; b.ax=b.ay=b.az=0; }
    }

    // KS 對使用正則化子步積分（取代標準 Verlet）
    for (const pair of ksActivePairs) {
        if (bodies[pair.ia].m > 0 && bodies[pair.ib].m > 0) {
            ksSubIntegrate(pair.ia, pair.ib, dt);
        }
    }

    computeAccelerations();

    // Kick 2 (非 KS 天體；KS 天體的速度已由 ksSubIntegrate 完整更新)
    for (let i = 0; i < N; i++) {
        const b = bodies[i];
        if (b.m === 0 || ksSet.has(i)) continue;
        b.vx += 0.5*b.ax*dt; b.vy += 0.5*b.ay*dt; b.vz += 0.5*b.az*dt;
    }
}
