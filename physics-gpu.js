// ═══════════════════════════════════════════════════════════════
// N-body 物理引擎 — WebGPU Compute Shader 後端
// 所有子步在 GPU 端批次執行，僅 readback 位置
// ═══════════════════════════════════════════════════════════════

const MAX_BODIES = 64; // 對齊 workgroup size
const BODY_STRIDE = 48; // 3 × vec4f = 12 floats = 48 bytes
const PARAMS_SIZE = 48; // 12 × f32/u32 (包含 subSteps 與 padding)

// ────────────────────────── WGSL 著色器 ──────────────────────────
const WGSL = /* wgsl */`
const WG: u32 = 64u;
const KS_FACTOR: f32 = 20.0;  // GPU 側近距交會偵測閾值 (對應 CPU 端 KS 正則化)

struct Body { pos: vec4f, vel: vec4f, acc: vec4f };
struct Params { 
    G: f32, EPS2: f32, BASE_DT: f32, MIN_DT: f32, 
    ETA: f32, C2: f32, N: u32, grOn: u32,
    subSteps: u32, pad1: u32, pad2: u32, pad3: u32
};
struct Dt { cur: u32, next: atomic<u32> };

@group(0) @binding(0) var<storage, read_write> B: array<Body>;
@group(0) @binding(1) var<uniform> P: Params;
@group(0) @binding(2) var<storage, read_write> dt: Dt;

// 宣告 Workgroup Shared Memory，用以加速 subSteps 迴圈
var<workgroup> sB: array<Body, WG>;
var<workgroup> sDtCur: f32;
var<workgroup> sDtNext: atomic<u32>;
var<workgroup> swallowedBy: array<i32, WG>;

@compute @workgroup_size(WG)
fn simulateSubsteps(@builtin(local_invocation_id) lid: vec3u) {
    let i = lid.x;
    let N = P.N;
    
    // 1. 將全域記憶體載入至 Shared Memory
    if (i < N) {
        sB[i] = B[i];
    }
    if (i == 0) {
        atomicStore(&sDtNext, atomicLoad(&dt.next));
    }
    workgroupBarrier();

    // 2. 在 Shared Memory 中進行 subSteps 次迭代 (極大降低 Global Memory 頻寬存取)
    for (var step = 0u; step < P.subSteps; step++) {
        // --- Swap DT ---
        if (i == 0) {
            sDtCur = bitcast<f32>(atomicLoad(&sDtNext));
            atomicStore(&sDtNext, bitcast<u32>(P.BASE_DT));
        }
        workgroupBarrier();

        // --- Kick 1 & Drift ---
        if (i < N) {
            let step_f = sDtCur;
            var v = sB[i].vel.xyz + 0.5 * sB[i].acc.xyz * step_f;
            var p = sB[i].pos.xyz + v * step_f;
            var m = sB[i].pos.w;
            if (any(abs(p) > vec3f(1e10)) || any(abs(v) > vec3f(1e10))) { 
                p = vec3f(1e12); 
                v = vec3f(0.0); 
                m = 0.0; // Zero out mass for out-of-bounds or swallowed objects
            }
            sB[i].pos = vec4f(p, m);
            sB[i].vel = vec4f(v, sB[i].vel.w);
        }
        workgroupBarrier();

        // --- 初始化合併追蹤 ---
        if (i < N) { swallowedBy[i] = -1; }
        workgroupBarrier();

        // --- Accel ---
        if (i < N) {
            let pi = sB[i].pos; let vi = sB[i].vel; let mi = pi.w;
            var acc = vec3f(0.0);
            var acc_comp = vec3f(0.0); // Kahan 補償求和：消除 f32 力累加截斷誤差
            var mdt = P.BASE_DT;

            if (mi > 0.0) {
                for (var j = 0u; j < N; j++) {
                    if (j == i) { continue; }
                    let pj = sB[j].pos; let vj = sB[j].vel; let mj = pj.w;
                    if (mj == 0.0) { continue; }

                    let r = pj.xyz - pi.xyz;
                    let d2 = dot(r,r) + P.EPS2;
                    let d = sqrt(d2);

                    let Rs = 2.0 * P.G * (mi + mj) / P.C2;
                    let R_merge = max(vi.w + vj.w, max(3.0 * Rs, sqrt(P.EPS2)));
                    if (d < R_merge) {
                        if (mi < mj || (mi == mj && i > j)) {
                            swallowedBy[i] = i32(j);
                            // 安全中斷：被吞噬者的加速度不再使用（Phase 4 清零），
                            // 多重合併目標由 pointer-jumping 鏈式解析（Phase 3）
                            break;
                        } else {
                            // 安全跳過：吞噬者暫時跳過此配對的力計算，
                            // Phase 5 會基於合併後的質量/位置重算完整加速度
                            continue;
                        }
                    }

                    let id3 = 1.0 / (d2 * d);

                    mdt = min(mdt, P.ETA * sqrt(d * d2 / (P.G * (mi+mj))));
                    let dv = vi.xyz - vj.xyz;
                    let vr2 = dot(dv,dv);
                    if (vr2 > 1e-30) { mdt = min(mdt, P.ETA * d / sqrt(vr2)); }

                    // INT-1 修正：GPU 側近距交會增強
                    // 在 KS 區域 (R_merge < d < KS_FACTOR*R_merge) 使用更激進的步長縮減，
                    // 等效於 CPU 端 KS 正則化在近心點的密集子步效果
                    let R_ks = KS_FACTOR * R_merge;
                    if (d < R_ks) {
                        mdt = min(mdt, P.ETA * 0.1 * sqrt(d * d2 / (P.G * (mi+mj))));
                    }

                    var g = P.G * id3 * mj * r;

                    // 修正：對稱版 Paczyński-Wiita 勢能 (Symmetric PW potential)
                    // 使用系統總質量 (mi+mj) 計算 Schwarzschild 半徑，確保符合牛頓第三運動定律 (動量守恆)
                    if (P.grOn != 0u) {
                        let rs = 2.0 * P.G * (mi + mj) / P.C2;
                        let dr = max(d - rs, rs * 0.05 + 1e-10);
                        g = P.G * mj * r / (d * dr * dr);
                    }
                    // Kahan compensated summation: 精度從 O(N·ε) 提升至 O(ε²)
                    let y_k = g - acc_comp;
                    let t_k = acc + y_k;
                    acc_comp = (t_k - acc) - y_k;
                    acc = t_k;
                }
            }
            
            if (swallowedBy[i] < 0 && mi > 0.0) {
                sB[i].acc = vec4f(acc, 0.0);
                atomicMin(&sDtNext, bitcast<u32>(max(mdt, P.MIN_DT)));
            }
        }
        workgroupBarrier();

        // --- 解析鏈式合併指標 (Chain Merger Resolution) ---
        // Pointer-jumping: 每次迭代將指標跳到 target 的 target
        // log2(64)=6 次迭代可解析最長 64 的鏈
        for (var pj_iter = 0u; pj_iter < 6u; pj_iter++) {
            var next_val = -1;
            if (i < N && swallowedBy[i] >= 0) {
                let mid = swallowedBy[i];
                next_val = swallowedBy[u32(mid)];
            }
            workgroupBarrier();
            if (i < N && next_val >= 0) {
                swallowedBy[i] = next_val;
            }
            workgroupBarrier();
        }

        // --- 質量、動量守恆與體積合併 ---
        if (i < N && swallowedBy[i] < 0 && sB[i].pos.w > 0.0) {
            var totalM = sB[i].pos.w;
            var totalP = sB[i].vel.xyz * totalM;
            var totalPos = sB[i].pos.xyz * totalM;
            var totalR3 = pow(sB[i].vel.w, 3.0);
            for (var k = 0u; k < N; k++) {
                if (swallowedBy[k] == i32(i)) {
                    let km = sB[k].pos.w;
                    totalP += sB[k].vel.xyz * km;
                    totalPos += sB[k].pos.xyz * km;
                    totalM += km;
                    totalR3 += pow(sB[k].vel.w, 3.0);
                }
            }
            if (totalM > sB[i].pos.w) {
                sB[i].pos = vec4f(totalPos / totalM, totalM);
                sB[i].vel = vec4f(totalP / totalM, pow(totalR3, 1.0 / 3.0));
            }
        }
        workgroupBarrier();

        // --- 清除被吞噬天體 ---
        if (i < N && swallowedBy[i] >= 0) {
            sB[i].pos.w = 0.0; // 僅將質量歸零，保留 xyz 作為死亡座標供主線程讀取
            sB[i].vel = vec4f(0.0);
            sB[i].acc = vec4f(0.0);
        }
        workgroupBarrier();

        // --- Phase 5: 合併存活者加速度重算 ---
        // 避免 Kick 2 使用基於舊質量/位置的過時加速度
        if (i < N && swallowedBy[i] < 0 && sB[i].pos.w > 0.0) {
            var didMerge = false;
            for (var km = 0u; km < N; km++) {
                if (swallowedBy[km] == i32(i)) { didMerge = true; break; }
            }
            if (didMerge) {
                let pi5 = sB[i].pos; let mi5 = pi5.w;
                var acc5 = vec3f(0.0);
                var acc5_comp = vec3f(0.0); // Kahan 補償
                for (var j5 = 0u; j5 < N; j5++) {
                    if (j5 == i) { continue; }
                    let pj5 = sB[j5].pos; let mj5 = pj5.w;
                    if (mj5 == 0.0) { continue; }
                    let r5 = pj5.xyz - pi5.xyz;
                    let d2_5 = dot(r5, r5) + P.EPS2;
                    let d_5 = sqrt(d2_5);
                    var g5 = P.G * mj5 * r5 / (d2_5 * d_5);
                    if (P.grOn != 0u) {
                        let rs5 = 2.0 * P.G * (mi5 + mj5) / P.C2;
                        let dr5 = max(d_5 - rs5, rs5 * 0.05 + 1e-10);
                        g5 = P.G * mj5 * r5 / (d_5 * dr5 * dr5);
                    }
                    let y5 = g5 - acc5_comp;
                    let t5 = acc5 + y5;
                    acc5_comp = (t5 - acc5) - y5;
                    acc5 = t5;
                }
                sB[i].acc = vec4f(acc5, 0.0);
            }
        }
        workgroupBarrier();

        // 步長指數衰減平滑：縮小快 (α=0.7)、放大慢 (α=0.3)，比 clamp 更平滑地保護辛結構
        if (i == 0) {
            let targetDt = bitcast<f32>(atomicLoad(&sDtNext));
            let ratio = targetDt / max(sDtCur, 1e-20);
            let alpha = select(0.3, 0.7, ratio < 1.0);
            var smoothedDt = max(sDtCur * pow(ratio, alpha), P.MIN_DT);
            smoothedDt = min(smoothedDt, sDtCur * 2.0); // 成長上限 2x，防止交會結束後步長暴漲破壞辛結構
            atomicStore(&sDtNext, bitcast<u32>(smoothedDt));
        }
        workgroupBarrier();

        // --- Kick 2 ---
        if (i < N) {
            let step_f = sDtCur;
            let v = sB[i].vel.xyz + 0.5 * sB[i].acc.xyz * step_f;
            sB[i].vel = vec4f(v, sB[i].vel.w);
        }
        workgroupBarrier();
    }

    // 3. 將結果寫回 Global Memory
    if (i < N) {
        B[i] = sB[i];
    }
    if (i == 0) {
        dt.cur = bitcast<u32>(sDtCur);
        atomicStore(&dt.next, atomicLoad(&sDtNext));
    }
}

// initAccel 用於加入新天體後的單次初始加速度計算
@compute @workgroup_size(WG)
fn initAccel(@builtin(local_invocation_id) lid: vec3u) {
    let i = lid.x;
    let N = P.N;
    if (i >= N) { return; }
    
    let pi = B[i].pos; let mi = pi.w;
    var acc = vec3f(0.0);
    var ia_comp = vec3f(0.0); // Kahan 補償

    if (mi == 0.0) {
        B[i].acc = vec4f(0.0);
        return;
    }

    for (var j = 0u; j < N; j++) {
        if (j == i) { continue; }
        let pj = B[j].pos; let vj = B[j].vel; let mj = pj.w;
        if (mj == 0.0) { continue; }
        let r = pj.xyz - pi.xyz;
        let d2 = dot(r,r) + P.EPS2;
        let d = sqrt(d2);

        let vi = B[i].vel;
        let Rs = 2.0 * P.G * (mi + mj) / P.C2;
        let R_merge = max(vi.w + vj.w, max(3.0 * Rs, sqrt(P.EPS2)));
        if (d < R_merge) { continue; }

        let id3 = 1.0 / (d2 * d);

        var g = P.G * id3 * mj * r;

        if (P.grOn != 0u) {
            let rs = 2.0 * P.G * (mi + mj) / P.C2;
            let dr = max(d - rs, rs * 0.05 + 1e-10);
            g = P.G * mj * r / (d * dr * dr);
        }
        let ia_y = g - ia_comp;
        let ia_t = acc + ia_y;
        ia_comp = (ia_t - acc) - ia_y;
        acc = ia_t;
    }
    B[i].acc = vec4f(acc, 0.0);
}
`;

// ────────────────────────── WebGPUPhysics 類 ──────────────────────────
export class WebGPUPhysics {
    constructor() {
        this.device = null;
        this.bodyBuf = null;
        this.paramsBuf = null;
        this.dtBuf = null;
        this.NUM_BUFFERS = 3;
        this.readBuffers = [];
        this.readIndex = 0;
        this.pendingReads = [];
        this.outPool = [];
        for (let i = 0; i < this.NUM_BUFFERS; i++) {
            this.outPool.push({
                positions: new Float32Array(MAX_BODIES * 3),
                velocities: new Float32Array(MAX_BODIES * 3),
                masses: new Float32Array(MAX_BODIES)
            });
        }
        this.pipelines = {};
        this.stateVersion = 0;
        this._enableGR = false;
        this._C2 = 63239.7263 * 63239.7263;
    }

    get version() { return this.stateVersion; }

    async init(bodies) {
        if (bodies.length > MAX_BODIES) {
            throw new Error(`WebGPU backend supports at most ${MAX_BODIES} bodies`);
        }
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error('No WebGPU adapter');
        this.device = await adapter.requestDevice();
        this.N = bodies.length;

        const bSize = MAX_BODIES * BODY_STRIDE;

        // 建立 Buffer
        this.bodyBuf = this.device.createBuffer({ size: bSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.paramsBuf = this.device.createBuffer({ size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.dtBuf = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        for (let i = 0; i < this.NUM_BUFFERS; i++) {
            this.readBuffers.push(this.device.createBuffer({ size: bSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
        }

        // 上傳天體資料 (pos.w = mass, vel.w = radius)
        const bd = new Float32Array(MAX_BODIES * 12);
        for (let i = 0; i < this.N; i++) {
            const b = bodies[i], o = i * 12;
            bd[o]=b.x; bd[o+1]=b.y; bd[o+2]=b.z; bd[o+3]=b.m;
            bd[o+4]=b.vx; bd[o+5]=b.vy; bd[o+6]=b.vz; bd[o+7]=b.radius || 0.0;
        }
        this.device.queue.writeBuffer(this.bodyBuf, 0, bd);

        // 上傳參數 (預設 subSteps=1)
        this._writeParams(1);

        // 初始化 dt buffer: cur=BASE_DT, next=BASE_DT
        const dtInit = new Uint32Array(2);
        const dv = new DataView(dtInit.buffer);
        dv.setFloat32(0, 0.0001, true); dv.setFloat32(4, 0.0001, true);
        this.device.queue.writeBuffer(this.dtBuf, 0, dtInit);

        // 編譯著色器 + 建立管線
        const mod = this.device.createShaderModule({ code: WGSL });
        const bgl = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        });
        const pl = this.device.createPipelineLayout({ bindGroupLayouts: [bgl] });
        
        for (const ep of ['simulateSubsteps', 'initAccel']) {
            this.pipelines[ep] = this.device.createComputePipeline({ layout: pl, compute: { module: mod, entryPoint: ep } });
        }
        
        this.bindGroup = this.device.createBindGroup({
            layout: bgl,
            entries: [
                { binding: 0, resource: { buffer: this.bodyBuf } },
                { binding: 1, resource: { buffer: this.paramsBuf } },
                { binding: 2, resource: { buffer: this.dtBuf } },
            ]
        });

        // 初始加速度計算
        const enc = this.device.createCommandEncoder();
        const p = enc.beginComputePass();
        p.setPipeline(this.pipelines.initAccel);
        p.setBindGroup(0, this.bindGroup);
        p.dispatchWorkgroups(1);
        p.end();
        this.device.queue.submit([enc.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }

    async step(subSteps) {
        // 更新 Uniform Buffer 中的 subSteps
        this._writeParams(subSteps);

        const enc = this.device.createCommandEncoder();
        
        // 只需要單次 Compute Pass 就可以跑完所有的 subSteps
        const p = enc.beginComputePass();
        p.setPipeline(this.pipelines.simulateSubsteps);
        p.setBindGroup(0, this.bindGroup);
        p.dispatchWorkgroups(1);
        p.end();

        const byteLen = this.N * BODY_STRIDE;
        
        const currentReadBuf = this.readBuffers[this.readIndex];
        const out = this.outPool[this.readIndex];
        this.readIndex = (this.readIndex + 1) % this.NUM_BUFFERS;

        enc.copyBufferToBuffer(this.bodyBuf, 0, currentReadBuf, 0, byteLen);
        this.device.queue.submit([enc.finish()]);

        const currentN = this.N; // capture for the closure
        const currentVersion = this.stateVersion;
        const readPromise = currentReadBuf.mapAsync(GPUMapMode.READ).then(() => {
            const raw = new Float32Array(currentReadBuf.getMappedRange());
            for (let i = 0; i < currentN; i++) {
                out.positions[i*3]   = raw[i*12];
                out.positions[i*3+1] = raw[i*12+1];
                out.positions[i*3+2] = raw[i*12+2];
                out.masses[i]    = raw[i*12+3];
                out.velocities[i*3]   = raw[i*12+4];
                out.velocities[i*3+1] = raw[i*12+5];
                out.velocities[i*3+2] = raw[i*12+6];
            }
            currentReadBuf.unmap();
            return { 
                positions: out.positions.subarray(0, currentN * 3), 
                velocities: out.velocities.subarray(0, currentN * 3), 
                masses: out.masses.subarray(0, currentN),
                version: currentVersion
            };
        }).catch(e => {
            console.error('GPU readback failed:', e);
            return { 
                positions: out.positions.subarray(0, currentN * 3), 
                velocities: out.velocities.subarray(0, currentN * 3), 
                masses: out.masses.subarray(0, currentN),
                version: currentVersion
            };
        });

        this.pendingReads.push(readPromise);

        if (this.pendingReads.length >= this.NUM_BUFFERS) {
            return await this.pendingReads.shift();
        } else {
            return { pended: true };
        }
    }

    async addBody(body) {
        if (this.N >= MAX_BODIES) return { ok: false, reason: `WebGPU 後端最多支援 ${MAX_BODIES} 個天體` };
        this.stateVersion++;
        const o = this.N * 12;
        const d = new Float32Array(12);
        d[0]=body.x; d[1]=body.y; d[2]=body.z; d[3]=body.m;
        d[4]=body.vx; d[5]=body.vy; d[6]=body.vz; d[7]=body.radius || 0.0;
        this.device.queue.writeBuffer(this.bodyBuf, this.N * BODY_STRIDE, d);
        this.N++;
        
        this._writeParams(1);
        
        // 重新計算加速度
        const enc = this.device.createCommandEncoder();
        const p = enc.beginComputePass();
        p.setPipeline(this.pipelines.initAccel);
        p.setBindGroup(0, this.bindGroup);
        p.dispatchWorkgroups(1);
        p.end();
        this.device.queue.submit([enc.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        return { ok: true };
    }

    updateSettings({ enableGR, cScale }) {
        this.stateVersion++;
        this._enableGR = enableGR;
        const c = 63239.7263 * (enableGR ? cScale : 1.0);
        this._C2 = c * c;
        // 我們不用在這裡傳 subSteps，因為 step() 呼叫時會更新
        this._writeParams(1); 
    }

    _writeParams(subSteps = 1) {
        const G = 4 * Math.PI * Math.PI;
        const buf = new ArrayBuffer(PARAMS_SIZE);
        const f = new Float32Array(buf);
        const u = new Uint32Array(buf);
        f[0] = G;           // G
        f[1] = 1e-10;       // EPSILON_SQ
        f[2] = 0.0001;      // BASE_DT
        f[3] = 1e-8;        // MIN_DT
        f[4] = 0.03;        // ETA
        f[5] = this._C2;    // C2
        u[6] = this.N;      // N
        u[7] = this._enableGR ? 1 : 0;
        u[8] = subSteps;    // subSteps
        // 9, 10, 11 保留作為 16-byte 對齊的 padding
        this.device.queue.writeBuffer(this.paramsBuf, 0, new Uint8Array(buf));
    }
}
