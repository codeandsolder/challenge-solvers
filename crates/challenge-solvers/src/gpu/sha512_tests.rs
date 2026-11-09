//! Physical GPU verification of a general SHA-512/SHA-384 compression core.
//! No ALTCHA-specific message restrictions: tests multi-block arbitrary bytes.
use std::{
    borrow::Cow,
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    time::Duration,
};

use sha2::{Digest, Sha384, Sha512};

struct ThreadWaker(std::thread::Thread);
impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut Context::from_waker(&waker)) {
            Poll::Ready(result) => return result,
            Poll::Pending => std::thread::park_timeout(Duration::from_millis(100)),
        }
    }
}

const IV_SHA512: [u64; 8] = [
    0x6a09e667f3bcc908_u64,
    0xbb67ae8584caa73b_u64,
    0x3c6ef372fe94f82b_u64,
    0xa54ff53a5f1d36f1_u64,
    0x510e527fade682d1_u64,
    0x9b05688c2b3e6c1f_u64,
    0x1f83d9abfb41bd6b_u64,
    0x5be0cd19137e2179_u64,
];
const IV_SHA384: [u64; 8] = [
    0xcbbb9d5dc1059ed8_u64,
    0x629a292a367cd507_u64,
    0x9159015a3070dd17_u64,
    0x152fecd8f70e5939_u64,
    0x67332667ffc00b31_u64,
    0x8eb44a8768581511_u64,
    0xdb0c2e0d64f98fa7_u64,
    0x47b5481dbefa4fa4_u64,
];

fn padding(input: &[u8]) -> Vec<[u8; 128]> {
    let mut bytes = input.to_vec();
    bytes.push(0x80);
    while bytes.len() % 128 != 112 {
        bytes.push(0);
    }
    bytes.extend_from_slice(&((input.len() as u128) * 8).to_be_bytes());
    bytes
        .chunks_exact(128)
        .map(|chunk| chunk.try_into().expect("padded block"))
        .collect()
}

fn solve_block(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    state: [u64; 8],
    block: &[u8; 128],
) -> [u64; 8] {
    let mut inputs = Vec::with_capacity(48);
    for value in state {
        inputs.push((value >> 32) as u32);
        inputs.push(value as u32);
    }
    for chunk in block.chunks_exact(8) {
        let value = u64::from_be_bytes(chunk.try_into().expect("eight bytes"));
        inputs.push((value >> 32) as u32);
        inputs.push(value as u32);
    }
    let input = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sha512-test-input"),
        size: (inputs.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sha512-test-output"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sha512-test-readback"),
        size: 64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input, 0, bytemuck::cast_slice(&inputs));
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sha512-test-bind"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sha512-test"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sha512-test"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 64);
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).expect("notify GPU result");
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("wait GPU completion");
    receiver
        .recv()
        .expect("map response")
        .expect("map GPU bytes");
    let mapped = slice.get_mapped_range().expect("mapped GPU SHA-512 output");
    let words: Vec<u32> = mapped
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("word")))
        .collect();
    let result =
        std::array::from_fn(|i| (u64::from(words[i * 2]) << 32) | u64::from(words[i * 2 + 1]));
    drop(mapped);
    staging.unmap();
    result
}

#[test]
#[ignore = "requires physical Vulkan GPU with SHADER_INT64"]
fn sha512_384_native_u64_gpu_compression_matches_sha2() {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapters = block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
    let wanted = std::env::var("WGPU_ADAPTER_NAME")
        .unwrap_or_default()
        .to_lowercase();
    let adapter = adapters
        .into_iter()
        .find(|a| {
            let info = a.get_info();
            info.device_type != wgpu::DeviceType::Cpu && info.name.to_lowercase().contains(&wanted)
        })
        .expect("requested physical Vulkan GPU");
    let info = adapter.get_info();
    assert!(
        adapter.features().contains(wgpu::Features::SHADER_INT64),
        "{} lacks shaderInt64",
        info.name
    );
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::SHADER_INT64,
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..wgpu::DeviceDescriptor::default()
    }))
    .expect("create u64-capable Vulkan device");
    let code = format!(
        "{}\n{}",
        include_str!("../shaders/sha512_core.wgsl"),
        r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(1)
fn main() {
 var state:array<u64,8>;
 var block:array<u64,16>;
 for(var i=0u;i<8u;i=i+1u){
   state[i]=(u64(input[i*2u])<<32u)|u64(input[i*2u+1u]);
 }
 for(var i=0u;i<16u;i=i+1u){
   block[i]=(u64(input[16u+i*2u])<<32u)|u64(input[17u+i*2u]);
 }
 let digest=sha512_compress(state,block);
 for(var i=0u;i<8u;i=i+1u){
   output[i*2u]=u32(digest[i]>>32u);
   output[i*2u+1u]=u32(digest[i]);
 }
}
"#
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sha512-gpu-primitive-test"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(code)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sha512-gpu-primitive-test"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let cases: Vec<Vec<u8>> = vec![
        vec![],
        b"abc".to_vec(),
        vec![0x11; 1],
        vec![0x22; 55],
        vec![0x33; 111],
        vec![0x44; 112],
        vec![0x55; 127],
        vec![0x66; 128],
        vec![0x77; 129],
        vec![0x88; 255],
        vec![0x99; 512],
    ];
    for message in &cases {
        for (kind, iv, expected) in [
            ("sha512", IV_SHA512, Sha512::digest(message).to_vec()),
            ("sha384", IV_SHA384, Sha384::digest(message).to_vec()),
        ] {
            let mut state = iv;
            for block in padding(message) {
                state = solve_block(&device, &queue, &pipeline, state, &block);
            }
            let got = state
                .iter()
                .flat_map(|word| word.to_be_bytes())
                .collect::<Vec<_>>();
            assert_eq!(
                &got[..expected.len()],
                expected,
                "GPU {} message len {}",
                kind,
                message.len()
            );
        }
    }
    println!(
        "verified SHA-384/SHA-512 {} message shapes on {}",
        cases.len(),
        info.name
    );
}
