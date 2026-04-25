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

            // 黑洞吸收邏輯 (Black Hole Absorption)
            const Rs = 2.0 * G * Math.max(bi.m, bj.m) / C2;
            const R_merge = Math.max(Rs * 1.5, 0.02);
            if (dist < R_merge) {
                if (bi.m <= bj.m) {
                    bi.m = 0; bi.x = 1e12; bi.y = 1e12; bi.z = 1e12;
                    bi.vx = 0; bi.vy = 0; bi.vz = 0;
                    bi.ax = 0; bi.ay = 0; bi.az = 0;
                } else {
                    bj.m = 0; bj.x = 1e12; bj.y = 1e12; bj.z = 1e12;
                    bj.vx = 0; bj.vy = 0; bj.vz = 0;
                    bj.ax = 0; bj.ay = 0; bj.az = 0;
                }
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

            if (enableGR) {
                const vix=bi.vx, viy=bi.vy, viz=bi.vz, vjx=bj.vx, vjy=bj.vy, vjz=bj.vz;
                const vi2 = vix*vix+viy*viy+viz*viz, vj2 = vjx*vjx+vjy*vjy+vjz*vjz;
                const vidvj = vix*vjx+viy*vjy+viz*vjz;
                const rdvj = dx*vjx+dy*vjy+dz*vjz;

                // Body i correction (修正符號)
                const t1i = vi2 + 2*vj2 - 4*vidvj - 1.5*(rdvj/dist)*(rdvj/dist) - (G*bi.m+4*G*bj.m)/dist;
                const t2i = dx*(4*vix-3*vjx)+dy*(4*viy-3*vjy)+dz*(4*viz-3*vjz);
                const ci = (G*bj.m)/(C2*distSq*dist);
                let gxi=ci*(dx*t1i-(vix-vjx)*t2i), gyi=ci*(dy*t1i-(viy-vjy)*t2i), gzi=ci*(dz*t1i-(viz-vjz)*t2i);
                const nmi = Math.sqrt(axi*axi+ayi*ayi+azi*azi), gmi = Math.sqrt(gxi*gxi+gyi*gyi+gzi*gzi);
                if (gmi > 3*nmi && nmi > 0) { const c=nmi/gmi; gxi*=c; gyi*=c; gzi*=c; }
                axi+=gxi; ayi+=gyi; azi+=gzi;

                // Body j correction (修正符號)
                const rx=-dx, ry=-dy, rz=-dz;
                const rdvi = rx*vix+ry*viy+rz*viz;
                const t1j = vj2 + 2*vi2 - 4*vidvj - 1.5*(rdvi/dist)*(rdvi/dist) - (G*bj.m+4*G*bi.m)/dist;
                const t2j = rx*(4*vjx-3*vix)+ry*(4*vjy-3*viy)+rz*(4*vjz-3*viz);
                const cj = (G*bi.m)/(C2*distSq*dist);
                let gxj=cj*(rx*t1j-(vjx-vix)*t2j), gyj=cj*(ry*t1j-(vjy-viy)*t2j), gzj=cj*(rz*t1j-(vjz-viz)*t2j);
                const nmj = Math.sqrt(axj*axj+ayj*ayj+azj*azj), gmj = Math.sqrt(gxj*gxj+gyj*gyj+gzj*gzj);
                if (gmj > 3*nmj && nmj > 0) { const c=nmj/gmj; gxj*=c; gyj*=c; gzj*=c; }
                axj+=gxj; ayj+=gyj; azj+=gzj;
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
        if (!isFinite(b.x)||!isFinite(b.vx)) { b.x=b.y=b.z=0; b.vx=b.vy=b.vz=0; b.ax=b.ay=b.az=0; }
    }
    computeAccelerations();
    for (let i = 0; i < N; i++) {
        const b = bodies[i];
        if (b.m === 0) continue;
        b.vx += 0.5*b.ax*dt; b.vy += 0.5*b.ay*dt; b.vz += 0.5*b.az*dt;
    }
}
