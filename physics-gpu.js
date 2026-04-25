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

@compute @workgroup_size(WG)
fn simulateSubsteps(@builtin(local_invocation_id) lid: vec3u) {
    let i = lid.x;
    let N = P.N;
    
    // 1. 將全域記憶體載入至 Shared Memory
    if (i < N) {
        sB[i] = B[i];
    }
    if (i == 0) {
        sDtCur = bitcast<f32>(dt.cur);
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
            if (any(abs(p) > vec3f(1e18)) || any(abs(v) > vec3f(1e18))) { p = vec3f(0.0); v = vec3f(0.0); }
            sB[i].pos = vec4f(p, sB[i].pos.w);
            sB[i].vel = vec4f(v, 0.0);
        }
        workgroupBarrier();

        // --- Accel ---
        if (i < N) {
            let pi = sB[i].pos; let vi = sB[i].vel; let mi = pi.w;
            var acc = vec3f(0.0);
            var mdt = P.BASE_DT;

            for (var j = 0u; j < N; j++) {
                if (j == i) { continue; }
                let pj = sB[j].pos; let vj = sB[j].vel; let mj = pj.w;
                let r = pj.xyz - pi.xyz;
                let d2 = dot(r,r) + P.EPS2;
                let d = sqrt(d2);
                let id3 = 1.0 / (d2 * d);

                mdt = min(mdt, P.ETA * sqrt(d * d2 / (P.G * (mi+mj))));
                let dv = vi.xyz - vj.xyz;
                let vr2 = dot(dv,dv);
                if (vr2 > 1e-30) { mdt = min(mdt, P.ETA * d / sqrt(vr2)); }

                var g = P.G * id3 * mj * r;

                // 1PN EIH GR (修正後的符號)
                if (P.grOn != 0u) {
                    let vi2 = dot(vi.xyz, vi.xyz); let vj2 = dot(vj.xyz, vj.xyz);
                    let vdv = dot(vi.xyz, vj.xyz);
                    let ndvj = dot(r, vj.xyz) / d;
                    
                    let t1 = vi2 + 2.0*vj2 - 4.0*vdv - 1.5*ndvj*ndvj - (5.0*P.G*mi + 4.0*P.G*mj)/d;
                    let t2 = dot(r, 4.0*vi.xyz - 3.0*vj.xyz);
                    let cf = P.G * mj / (P.C2 * d2 * d);
                    
                    var gr = cf * (r * t1 - (vi.xyz - vj.xyz) * t2);
                    
                    let gm = length(g); let grm = length(gr);
                    if (grm > gm && gm > 0.0) { gr *= gm/grm; }
                    g += gr;
                }
                acc += g;
            }
            sB[i].acc = vec4f(acc, 0.0);
            atomicMin(&sDtNext, bitcast<u32>(max(mdt, P.MIN_DT)));
        }
        workgroupBarrier();

        // --- Kick 2 ---
        if (i < N) {
            let step_f = sDtCur;
            let v = sB[i].vel.xyz + 0.5 * sB[i].acc.xyz * step_f;
            sB[i].vel = vec4f(v, 0.0);
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
    
    let pi = B[i].pos; let vi = B[i].vel; let mi = pi.w;
    var acc = vec3f(0.0);
    var mdt = P.BASE_DT;

    for (var j = 0u; j < N; j++) {
        if (j == i) { continue; }
        let pj = B[j].pos; let vj = B[j].vel; let mj = pj.w;
        let r = pj.xyz - pi.xyz;
        let d2 = dot(r,r) + P.EPS2;
        let d = sqrt(d2);
        let id3 = 1.0 / (d2 * d);

        mdt = min(mdt, P.ETA * sqrt(d * d2 / (P.G * (mi+mj))));
        let dv = vi.xyz - vj.xyz;
        let vr2 = dot(dv,dv);
        if (vr2 > 1e-30) { mdt = min(mdt, P.ETA * d / sqrt(vr2)); }

        var g = P.G * id3 * mj * r;

        if (P.grOn != 0u) {
            let vi2 = dot(vi.xyz, vi.xyz); let vj2 = dot(vj.xyz, vj.xyz);
            let vdv = dot(vi.xyz, vj.xyz);
            let ndvj = dot(r, vj.xyz) / d;
            
            let t1 = vi2 + 2.0*vj2 - 4.0*vdv - 1.5*ndvj*ndvj - (5.0*P.G*mi + 4.0*P.G*mj)/d;
            let t2 = dot(r, 4.0*vi.xyz - 3.0*vj.xyz);
            let cf = P.G * mj / (P.C2 * d2 * d);
            
            var gr = cf * (r * t1 - (vi.xyz - vj.xyz) * t2);
            
            let gm = length(g); let grm = length(gr);
            if (grm > gm && gm > 0.0) { gr *= gm/grm; }
            g += gr;
        }
        acc += g;
    }
    B[i].acc = vec4f(acc, 0.0);
    atomicMin(&dt.next, bitcast<u32>(max(mdt, P.MIN_DT)));
}
`;

// ────────────────────────── WebGPUPhysics 類 ──────────────────────────
export class WebGPUPhysics {
    constructor() {
        this.device = null;
        this.bodyBuf = null;
        this.paramsBuf = null;
        this.dtBuf = null;
        this.readBuf = null;
        this.pipelines = {};
        this.bindGroup = null;
        this.N = 0;
        this._enableGR = false;
        this._C2 = 63239.7263 * 63239.7263;
    }

    async init(bodies) {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error('No WebGPU adapter');
        this.device = await adapter.requestDevice();
        this.N = bodies.length;

        const bSize = MAX_BODIES * BODY_STRIDE;

        // 建立 Buffer
        this.bodyBuf = this.device.createBuffer({ size: bSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.paramsBuf = this.device.createBuffer({ size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.dtBuf = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.readBuf = this.device.createBuffer({ size: bSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

        // 上傳天體資料 (pos.w = mass)
        const bd = new Float32Array(MAX_BODIES * 12);
        for (let i = 0; i < this.N; i++) {
            const b = bodies[i], o = i * 12;
            bd[o]=b.x; bd[o+1]=b.y; bd[o+2]=b.z; bd[o+3]=b.m;
            bd[o+4]=b.vx; bd[o+5]=b.vy; bd[o+6]=b.vz;
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
        enc.copyBufferToBuffer(this.bodyBuf, 0, this.readBuf, 0, byteLen);
        this.device.queue.submit([enc.finish()]);

        await this.readBuf.mapAsync(GPUMapMode.READ);
        const raw = new Float32Array(this.readBuf.getMappedRange().slice(0));
        this.readBuf.unmap();

        const pos = new Float32Array(this.N * 3);
        for (let i = 0; i < this.N; i++) {
            pos[i*3]   = raw[i*12];
            pos[i*3+1] = raw[i*12+1];
            pos[i*3+2] = raw[i*12+2];
        }
        return pos;
    }

    addBody(body) {
        if (this.N >= MAX_BODIES) return;
        const o = this.N * 12;
        const d = new Float32Array(12);
        d[0]=body.x; d[1]=body.y; d[2]=body.z; d[3]=body.m;
        d[4]=body.vx; d[5]=body.vy; d[6]=body.vz;
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
    }

    updateSettings({ enableGR, cScale }) {
        this._enableGR = enableGR;
        const c = 63239.7263 * cScale;
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
        f[1] = 0.001;       // EPSILON_SQ
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
