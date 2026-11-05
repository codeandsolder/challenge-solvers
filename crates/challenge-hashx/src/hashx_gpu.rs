use std::{borrow::Cow, fmt::Write as _, sync::mpsc, time::Instant};

use anyhow::{Context, anyhow};
use hashwx::HashX;

use crate::{
    hashx::{Answer, Challenge, GpuOp, Instr, describe, valid_prefix},
    hashx_gpu_native::NativeGpuBackend,
};

const AMD_WG_SIZE: u32 = 64;
const NVIDIA_WG_SIZE: u32 = 32;
const SPECIALIZED_WG_SIZE: u32 = 64;
const SPECIALIZED_BATCH: u32 = 262_144;
const PERSISTENT_MAX_BATCH: u32 = 262_144;
const PERSISTENT_CROSSOVER_DIFFICULTY: u32 = 21;
const PROGRAM_BYTES: u64 = 512 * 16;
const NATIVE_MIN_DIFFICULTY: u32 = 1;
const NVIDIA_VULKAN_IMPLEMENTATION: &str = "vulkan-nvidia-u32-persistent";
const EXTMUL_VULKAN_IMPLEMENTATION: &str = "vulkan-extmul-persistent";
const SPECIALIZED_IMPLEMENTATION: &str = "wgsl-specialized";

pub struct GpuResult {
    pub answer: Answer,
    pub solve_ms: f64,
    pub implementation: &'static str,
}

pub async fn adapter_names() -> Vec<String> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let mut names = instance
        .enumerate_adapters(wgpu::Backends::VULKAN)
        .await
        .into_iter()
        .filter_map(|adapter| {
            let info = adapter.get_info();
            (adapter.features().contains(wgpu::Features::SHADER_INT64)
                && info.device_type != wgpu::DeviceType::Cpu)
                .then_some(info.name)
        })
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
}

pub struct GpuSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: String,
    params: wgpu::Buffer,
    program: wgpu::Buffer,
    solution: wgpu::Buffer,
    staging: wgpu::Buffer,
    persistent_pipeline: Option<wgpu::ComputePipeline>,
    persistent_bind_group: Option<wgpu::BindGroup>,
    persistent_wg_size: u32,
    native_backend: Option<NativeGpuBackend>,
    persistent_implementation: &'static str,
}

impl GpuSolver {
    #[expect(
        clippy::too_many_lines,
        reason = "GPU resource and pipeline construction is one audited initialization transaction"
    )]
    /// Initialize a compatible Vulkan device and optionally the native CUDA backend.
    ///
    /// # Errors
    /// Returns errors for missing GPU capabilities, device setup or pipeline creation.
    pub async fn create() -> anyhow::Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
            .await
            .context("no WebGPU/Vulkan adapter for HashX")?;
        let adapter_features = adapter.features();
        anyhow::ensure!(
            adapter_features.contains(wgpu::Features::SHADER_INT64),
            "HashX GPU backend requires shaderInt64"
        );
        let persistent_supported = adapter_features.contains(wgpu::Features::PASSTHROUGH_SHADERS);
        let required_features = if persistent_supported {
            wgpu::Features::SHADER_INT64 | wgpu::Features::PASSTHROUGH_SHADERS
        } else {
            wgpu::Features::SHADER_INT64
        };
        let info = adapter.get_info();
        let is_nvidia = info.vendor == 0x10de;
        let native_backend = match NativeGpuBackend::create_for_adapter(&info) {
            Ok(backend) => {
                if let Some(ref backend) = backend {
                    tracing::info!(
                        adapter = %backend.adapter(),
                        implementation = backend.implementation(),
                        "native HashX GPU backend ready"
                    );
                }
                backend
            }
            Err(error) => {
                tracing::warn!(%error, "native HashX GPU backend unavailable; using Vulkan fallback");
                None
            }
        };
        let persistent_implementation = if is_nvidia {
            NVIDIA_VULKAN_IMPLEMENTATION
        } else {
            EXTMUL_VULKAN_IMPLEMENTATION
        };
        let persistent_wg_size = if is_nvidia {
            NVIDIA_WG_SIZE
        } else {
            AMD_WG_SIZE
        };
        let adapter_name = format!("{} ({:?}, {})", info.name, info.backend, info.driver);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features,
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            })
            .await
            .context("request HashX GPU device")?;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hashx-params"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let program = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hashx-program"),
            size: PROGRAM_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let solution = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hashx-solution"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hashx-staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (persistent_pipeline, persistent_bind_group) = if persistent_supported {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("hashx-persistent-layout"),
                entries: &[
                    storage_layout_entry(0, true),
                    storage_layout_entry(1, true),
                    storage_layout_entry(2, false),
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hashx-persistent-pipeline-layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let module = create_persistent_shader_module(&device, is_nvidia, persistent_wg_size);
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(if is_nvidia {
                    "hashx-persistent-nvidia-u32-solve"
                } else {
                    "hashx-persistent-extmul-solve"
                }),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("solve"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hashx-persistent-bind"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: program.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: solution.as_entire_binding(),
                    },
                ],
            });
            (Some(pipeline), Some(bind_group))
        } else {
            (None, None)
        };

        Ok(Self {
            device,
            queue,
            adapter: adapter_name,
            params,
            program,
            solution,
            staging,
            persistent_pipeline,
            persistent_bind_group,
            persistent_wg_size,
            native_backend,
            persistent_implementation,
        })
    }

    /// Return the selected adapter name.
    #[must_use]
    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    /// List available Vulkan and native GPU implementations.
    #[must_use]
    pub fn implementations(&self) -> Vec<&'static str> {
        let mut implementations = Vec::with_capacity(3);
        if let Some(native) = self.native_backend.as_ref() {
            implementations.push(native.implementation());
        }
        if self.persistent_pipeline.is_some() {
            implementations.push(self.persistent_implementation);
        }
        implementations.push(SPECIALIZED_IMPLEMENTATION);
        implementations
    }

    /// Execute a GPU proof search with an optional explicitly selected backend.
    ///
    /// # Errors
    /// Rejects invalid requests or unsupported implementations and propagates GPU failures.
    pub fn solve_with_implementation(
        &mut self,
        challenge: &Challenge,
        requested: Option<&str>,
    ) -> anyhow::Result<GpuResult> {
        challenge.validate()?;
        let started = Instant::now();
        let seed = hex::decode(&challenge.data).context("decode HashX challenge")?;
        let (hashx, program) = describe(&seed)?;

        if let Some(requested) = requested {
            if requested == SPECIALIZED_IMPLEMENTATION {
                return self.solve_specialized(
                    challenge,
                    &hashx,
                    program.key(),
                    program.instructions(),
                    started,
                );
            }
            if requested == self.persistent_implementation {
                anyhow::ensure!(
                    self.persistent_pipeline.is_some(),
                    "HashX GPU implementation {requested} is unavailable on {}",
                    self.adapter
                );
                return self.solve_persistent(
                    challenge,
                    &hashx,
                    program.key(),
                    program.instructions(),
                    started,
                    self.persistent_implementation,
                );
            }
            if let Some(native) = self.native_backend.as_mut()
                && requested == native.implementation()
            {
                let implementation = native.implementation();
                let nonce = native.solve(&program, challenge.difficulty)?;
                return validated_result(
                    &hashx,
                    nonce,
                    challenge.difficulty,
                    started,
                    implementation,
                );
            }
            anyhow::bail!(
                "HashX GPU implementation {requested} is unavailable on {}; available: {}",
                self.adapter,
                self.implementations().join(", ")
            );
        }

        if challenge.difficulty >= NATIVE_MIN_DIFFICULTY {
            let native_result = self.native_backend.as_mut().map(|backend| {
                let implementation = backend.implementation();
                backend
                    .solve(&program, challenge.difficulty)
                    .map(|nonce| (nonce, implementation))
            });
            match native_result {
                None => {}
                Some(Ok((nonce, implementation))) => {
                    return validated_result(
                        &hashx,
                        nonce,
                        challenge.difficulty,
                        started,
                        implementation,
                    );
                }
                Some(Err(error)) => {
                    tracing::warn!(%error, "native HashX GPU solve failed; disabling backend and falling back to Vulkan");
                    self.native_backend = None;
                }
            }
        }

        if challenge.difficulty <= PERSISTENT_CROSSOVER_DIFFICULTY
            && self.persistent_pipeline.is_some()
        {
            self.solve_persistent(
                challenge,
                &hashx,
                program.key(),
                program.instructions(),
                started,
                self.persistent_implementation,
            )
        } else {
            self.solve_specialized(
                challenge,
                &hashx,
                program.key(),
                program.instructions(),
                started,
            )
        }
    }

    fn solve_persistent(
        &self,
        challenge: &Challenge,
        hashx: &HashX,
        key: &[u64; 4],
        program: &[Instr],
        started: Instant,
        implementation: &'static str,
    ) -> anyhow::Result<GpuResult> {
        let pipeline = self
            .persistent_pipeline
            .as_ref()
            .context("persistent HashX pipeline unavailable")?;
        let bind_group = self
            .persistent_bind_group
            .as_ref()
            .context("persistent HashX bind group unavailable")?;
        self.queue
            .write_buffer(&self.program, 0, bytemuck::cast_slice(program));
        self.queue
            .write_buffer(&self.solution, 0, &u32::MAX.to_le_bytes());

        let batch_size = persistent_batch(challenge.difficulty, self.persistent_wg_size);
        let mut base = 0_u64;
        while u32::try_from(base).is_ok() {
            let remaining = u64::from(u32::MAX) + 1 - base;
            let batch = u32::try_from(u64::from(batch_size).min(remaining))
                .context("persistent HashX batch exceeds u32")?;
            self.write_params(key, base, challenge.difficulty);
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(batch.div_ceil(self.persistent_wg_size), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&self.solution, 0, &self.staging, 0, 4);
            let submission = self.queue.submit([encoder.finish()]);
            self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })?;
            let nonce = self.read_solution()?;
            if nonce != u32::MAX {
                return validated_result(
                    hashx,
                    nonce,
                    challenge.difficulty,
                    started,
                    implementation,
                );
            }
            base += u64::from(batch);
        }
        Err(anyhow!("HashX GPU search exhausted u32 nonce space"))
    }

    fn solve_specialized(
        &self,
        challenge: &Challenge,
        hashx: &HashX,
        key: &[u64; 4],
        program: &[Instr],
        started: Instant,
    ) -> anyhow::Result<GpuResult> {
        let shader = specialized_shader(program)?;
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("hashx-specialized"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader)),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("hashx-specialized-solve"),
                layout: None,
                module: &module,
                entry_point: Some("solve"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hashx-specialized-bind"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.solution.as_entire_binding(),
                },
            ],
        });
        self.queue
            .write_buffer(&self.solution, 0, &u32::MAX.to_le_bytes());
        let mut base = 0_u64;
        while u32::try_from(base).is_ok() {
            let remaining = u64::from(u32::MAX) + 1 - base;
            let batch = u32::try_from(u64::from(SPECIALIZED_BATCH).min(remaining))
                .context("specialized HashX batch exceeds u32")?;
            self.write_params(key, base, challenge.difficulty);
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(batch.div_ceil(SPECIALIZED_WG_SIZE), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&self.solution, 0, &self.staging, 0, 4);
            let submission = self.queue.submit([encoder.finish()]);
            self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })?;
            let nonce = self.read_solution()?;
            if nonce != u32::MAX {
                return validated_result(
                    hashx,
                    nonce,
                    challenge.difficulty,
                    started,
                    SPECIALIZED_IMPLEMENTATION,
                );
            }
            base += u64::from(batch);
        }
        Err(anyhow!("HashX GPU search exhausted u32 nonce space"))
    }

    fn write_params(&self, key: &[u64; 4], base: u64, difficulty: u32) {
        let mut words = [0_u32; 16];
        for (index, value) in key.iter().copied().enumerate() {
            let (low, high) = split_u64(value);
            words[index * 2] = low;
            words[index * 2 + 1] = high;
        }
        words[8] = split_u64(base).0;
        words[9] = difficulty;
        self.queue
            .write_buffer(&self.params, 0, bytemuck::cast_slice(&words));
    }

    fn read_solution(&self) -> anyhow::Result<u32> {
        let slice = self.staging.slice(..4);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv().context("HashX map callback dropped")??;
        let mapped = slice.get_mapped_range().context("map HashX result")?;
        let bytes = mapped
            .get(..4)
            .context("HashX result buffer shorter than four bytes")?;
        let result = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        drop(mapped);
        self.staging.unmap();
        Ok(result)
    }
}

const fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[expect(
    unsafe_code,
    reason = "wgpu SPIR-V passthrough requires its audited unsafe module-construction API"
)]
fn create_persistent_shader_module(
    device: &wgpu::Device,
    is_nvidia: bool,
    workgroup_size: u32,
) -> wgpu::ShaderModule {
    let (bytes, label): (&[u8], &str) = if is_nvidia {
        (
            include_bytes!("hashx_nvidia_interpreter_solve.spv"),
            "hashx-persistent-nvidia-u32",
        )
    } else {
        (
            include_bytes!("hashx_blocksolve_extmul.spv"),
            "hashx-persistent-extmul",
        )
    };
    let words = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    // SAFETY: both embedded modules are generated from validated WGSL and checked by spirv-val.
    // RADV uses the extended-multiply fixed-block kernel; NVIDIA uses a warp-sized interpreter
    // whose 64-bit multiplies are explicitly decomposed into 32-bit operations. Passthrough is
    // required for the RADV extended-multiply operations and keeps both persistent paths uniform.
    unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some(label),
            entry_points: Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint {
                name: Cow::Borrowed("solve"),
                workgroup_size: (workgroup_size, 1, 1),
            }]),
            spirv: Some(Cow::Owned(words)),
            ..Default::default()
        })
    }
}

fn persistent_batch(difficulty: u32, workgroup_size: u32) -> u32 {
    let expected = 1_u32 << difficulty.min(PERSISTENT_CROSSOVER_DIFFICULTY);
    expected
        .div_ceil(4)
        .clamp(workgroup_size, PERSISTENT_MAX_BATCH)
}

const fn split_u64(value: u64) -> (u32, u32) {
    let bytes = value.to_le_bytes();
    (
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
    )
}

fn validated_result(
    hashx: &HashX,
    nonce: u32,
    difficulty: u32,
    started: Instant,
    implementation: &'static str,
) -> anyhow::Result<GpuResult> {
    let hash = hashx.hash_bytes(u64::from(nonce));
    anyhow::ensure!(
        valid_prefix(&hash, difficulty),
        "HashX GPU solver returned nonce with invalid difficulty"
    );
    Ok(GpuResult {
        answer: Answer {
            nonce: nonce.to_string(),
            hash: hex::encode(hash),
        },
        solve_ms: started.elapsed().as_secs_f64() * 1000.0,
        implementation,
    })
}

fn op_source(i: Instr) -> anyhow::Result<String> {
    Ok(match i.kind()? {
        GpuOp::Mul => format!("r[{d}u]=r[{d}u]*r[{s}u];\n", d = i.dst, s = i.src),
        GpuOp::UMulH => format!(
            "{{let x=umulhi(r[{d}u],r[{s}u]);r[{d}u]=x;mh=u32(x);}}\n",
            d = i.dst,
            s = i.src
        ),
        GpuOp::SMulH => format!(
            "{{let x=smulhi(r[{d}u],r[{s}u]);r[{d}u]=x;mh=u32(x);}}\n",
            d = i.dst,
            s = i.src
        ),
        GpuOp::AddShift => format!(
            "r[{d}u]=r[{d}u]+(r[{s}u]<<{n}u);\n",
            d = i.dst,
            s = i.src,
            n = i.imm
        ),
        GpuOp::AddConst => format!(
            "r[{d}u]=r[{d}u]+0x{v:016x}lu;\n",
            d = i.dst,
            v = i64::from(i.imm.cast_signed()).cast_unsigned()
        ),
        GpuOp::Sub => format!("r[{d}u]=r[{d}u]-r[{s}u];\n", d = i.dst, s = i.src),
        GpuOp::Xor => format!("r[{d}u]=r[{d}u]^r[{s}u];\n", d = i.dst, s = i.src),
        GpuOp::XorConst => format!(
            "r[{d}u]=r[{d}u]^0x{v:016x}lu;\n",
            d = i.dst,
            v = i64::from(i.imm.cast_signed()).cast_unsigned()
        ),
        GpuOp::Rotate => format!("r[{d}u]=ror(r[{d}u],{n}u);\n", d = i.dst, n = i.imm),
        GpuOp::Target | GpuOp::Branch => String::new(),
    })
}

fn specialized_shader(program: &[Instr]) -> anyhow::Result<String> {
    let mut body = String::new();
    let mut target = None;
    for (index, &instruction) in program.iter().enumerate() {
        match instruction.kind()? {
            GpuOp::Target => target = Some(index),
            GpuOp::Branch => {
                let start = target.context("HashX branch without target")? + 1;
                let _ = writeln!(
                    body,
                    "if(allow&&((mh&0x{:08x}u)==0u)){{allow=false;",
                    instruction.imm
                );
                for &replay in &program[start..index] {
                    body.push_str(&op_source(replay)?);
                }
                body.push_str("}\n");
            }
            _ => body.push_str(&op_source(instruction)?),
        }
    }
    Ok(format!(
        r"
struct Params{{w:array<u32,16>}};struct Out{{solution:atomic<u32>}};
@group(0) @binding(0) var<storage,read> params:Params;@group(0) @binding(1) var<storage,read_write> out:Out;
struct S{{v0:u64,v1:u64,v2:u64,v3:u64}};
fn rol(x:u64,n:u32)->u64{{return(x<<n)|(x>>(64u-n));}}fn ror(x:u64,n:u32)->u64{{return(x>>n)|(x<<(64u-n));}}
fn sip(q:S)->S{{var s=q;s.v0=s.v0+s.v1;s.v2=s.v2+s.v3;s.v1=rol(s.v1,13u);s.v3=rol(s.v3,16u);s.v1=s.v1^s.v0;s.v3=s.v3^s.v2;s.v0=rol(s.v0,32u);s.v2=s.v2+s.v1;s.v0=s.v0+s.v3;s.v1=rol(s.v1,17u);s.v3=rol(s.v3,21u);s.v1=s.v1^s.v2;s.v3=s.v3^s.v0;s.v2=rol(s.v2,32u);return s;}}
fn key()->S{{return S(u64(params.w[0])|(u64(params.w[1])<<32u),u64(params.w[2])|(u64(params.w[3])<<32u),u64(params.w[4])|(u64(params.w[5])<<32u),u64(params.w[6])|(u64(params.w[7])<<32u));}}
fn init(k:S,n:u64)->array<u64,8>{{var s=k;s.v1=s.v1^0xeelu;s.v3=s.v3^n;s=sip(s);s=sip(s);s.v0=s.v0^n;s.v2=s.v2^0xeelu;s=sip(s);s=sip(s);s=sip(s);s=sip(s);var t=s;t.v1=t.v1^0xddlu;t=sip(t);t=sip(t);t=sip(t);t=sip(t);return array<u64,8>(s.v0,s.v1,s.v2,s.v3,t.v0,t.v1,t.v2,t.v3);}}
fn umulhi(a:u64,b:u64)->u64{{let m=0xffffffffu;let a0=u64(u32(a&u64(m)));let a1=a>>32u;let b0=u64(u32(b&u64(m)));let b1=b>>32u;let p0=a0*b0;let p1=a0*b1;let p2=a1*b0;let p3=a1*b1;let carry=((p0>>32u)+(p1&u64(m))+(p2&u64(m)))>>32u;return p3+(p1>>32u)+(p2>>32u)+carry;}}
fn smulhi(a:u64,b:u64)->u64{{var h=umulhi(a,b);if((a>>63u)!=0lu){{h=h-b;}}if((b>>63u)!=0lu){{h=h-a;}}return h;}}
fn run(r0:array<u64,8>)->array<u64,8>{{var r=r0;var allow=true;var mh=0u;{body}return r;}}
fn hx(n:u64)->array<u64,4>{{let k=key();let r=run(init(k,n));var x=S(r[0]+k.v0,r[1]+k.v1,r[2],r[3]);var y=S(r[4],r[5],r[6]+k.v2,r[7]+k.v3);x=sip(x);y=sip(y);return array<u64,4>(x.v0^y.v0,x.v1^y.v1,x.v2^y.v2,x.v3^y.v3);}}
fn bswap(x:u32)->u32{{return((x&0xffu)<<24u)|((x&0xff00u)<<8u)|((x&0xff0000u)>>8u)|((x&0xff000000u)>>24u);}}
@compute @workgroup_size(64) fn solve(@builtin(global_invocation_id) gid:vec3<u32>){{if(atomicLoad(&out.solution)!=0xffffffffu){{return;}}let nonce=params.w[8]+gid.x;let h=hx(u64(nonce));let d=params.w[9];let p0=bswap(u32(h[0]));let p1=bswap(u32(h[0]>>32u));var ok=false;if(d<=32u){{ok=(d==0u)||((p0>>(32u-d))==0u);}}else{{ok=(p0==0u)&&((p1>>(64u-d))==0u);}}if(ok){{_=atomicMin(&out.solution,nonce);}}}}
"
    ))
}
