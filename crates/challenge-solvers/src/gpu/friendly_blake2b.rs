//! Fixed-block BLAKE2b-256 GPU primitive for Friendly Captcha v1.
//!
//! A single two-dimensional GPU dispatch searches independent bank IDs
//! across a shared contiguous counter window, with one atomic result per bank.
//! The 128-byte final BLAKE2b block and unsigned comparisons match RustCrypto.

use super::{GpuContext, GpuError, WG_SIZE};
use std::borrow::Cow;

/// The full set of u8 bank selectors, including 255 for primitive testing.
const MAX_BANKS: usize = 256;
const RESULT_BYTES_PER_BANK: u64 = 16;
const PARAM_PREFIX_WORDS: usize = 37;

/// Initialized native-u64 compute pipeline and per-bank result buffers.
pub struct FriendlyBlake2bGpu<'a> {
    gpu: &'a GpuContext,
    pipeline: wgpu::ComputePipeline,
    input: wgpu::Buffer,
    output: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

impl<'a> FriendlyBlake2bGpu<'a> {
    /// Compile one native-u64 pipeline and allocate reusable GPU resources.
    ///
    /// # Errors
    /// Returns a hardware capability error if integer u64 shader math is unavailable.
    pub fn new(gpu: &'a GpuContext) -> Result<Self, GpuError> {
        if !gpu.device.features().contains(wgpu::Features::SHADER_INT64) {
            return Err(GpuError::new("Friendly BLAKE2b requires GPU SHADER_INT64"));
        }
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("friendly-blake2b-256-banks"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shaders/friendly_blake2b.wgsl"
                ))),
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("friendly-blake2b-256-banks"),
                layout: None,
                module: &shader,
                entry_point: Some("solve"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let input = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("friendly-blake2b-bank-input"),
            size: ((PARAM_PREFIX_WORDS + MAX_BANKS) * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let result_bytes = MAX_BANKS as u64 * RESULT_BYTES_PER_BANK;
        let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("friendly-blake2b-bank-output"),
            size: result_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("friendly-blake2b-bank-readback"),
            size: result_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("friendly-blake2b-banks"),
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
        Ok(Self {
            gpu,
            pipeline,
            input,
            output,
            staging,
            bind,
        })
    }

    /// Per-bank maximum candidate count for a single native GPU dispatch.
    #[must_use]
    pub fn batch_capacity(&self) -> u64 {
        u64::from(self.gpu.max_wgs) * u64::from(WG_SIZE)
    }

    /// Search a contiguous u32 counter window across independent bank IDs
    /// in one GPU submission and one readback, returning a result per bank.
    ///
    /// Every reported nonce belongs to the requested window; the consuming
    /// protocol must independently validate each candidate with RustCrypto.
    ///
    /// # Errors
    /// Rejects empty or duplicate bank selectors, overlarge counter windows,
    /// device workgroup-limit violations and GPU readback errors.
    pub async fn scan_banks(
        &mut self,
        padded: &[u8; 128],
        banks: &[u8],
        base: u32,
        count: u32,
        threshold: u32,
    ) -> Result<Vec<Option<u32>>, GpuError> {
        let domain_end = u64::from(base) + u64::from(count);
        if banks.is_empty()
            || banks.len() > MAX_BANKS
            || count == 0
            || domain_end > u64::from(u32::MAX) + 1
            || u64::from(count) > self.batch_capacity()
            || banks.len() as u32 > self.gpu.max_wgs
        {
            return Err(GpuError::new(
                "Friendly bank count or counter window exceeds GPU/u32 domain",
            ));
        }
        let mut seen = [false; MAX_BANKS];
        for &bank in banks {
            if std::mem::replace(&mut seen[usize::from(bank)], true) {
                return Err(GpuError::new(
                    "Friendly bank selector repeated in one dispatch",
                ));
            }
        }
        let mut packed = [0u32; PARAM_PREFIX_WORDS + MAX_BANKS];
        for (slot, bytes) in packed[..32].iter_mut().zip(padded.as_chunks::<4>().0) {
            *slot = u32::from_le_bytes(*bytes);
        }
        packed[32] = base;
        packed[33] = count;
        packed[34] = threshold;
        packed[36] = u32::try_from(banks.len())
            .map_err(|_| GpuError::new("Friendly bank selector count overflow"))?;
        for (slot, &bank) in packed[PARAM_PREFIX_WORDS..].iter_mut().zip(banks) {
            *slot = u32::from(bank);
        }
        let result_bytes = banks.len() as u64 * RESULT_BYTES_PER_BANK;
        self.gpu
            .queue
            .write_buffer(&self.input, 0, bytemuck::cast_slice(&packed));
        self.gpu.queue.write_buffer(
            &self.output,
            0,
            &[0_u8; MAX_BANKS * RESULT_BYTES_PER_BANK as usize][..result_bytes as usize],
        );
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("friendly-blake2b-banks"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("friendly-blake2b-bank-search"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups(
                count.div_ceil(WG_SIZE),
                u32::try_from(banks.len())
                    .map_err(|_| GpuError::new("Friendly bank selector count overflow"))?,
                1,
            );
        }
        encoder.copy_buffer_to_buffer(&self.output, 0, &self.staging, 0, result_bytes);
        self.gpu.queue.submit([encoder.finish()]);
        let mapped_slice = self.staging.slice(0..result_bytes);
        let (tx, rx) = futures_channel::oneshot::channel();
        mapped_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|err| GpuError::new(format!("poll Friendly GPU results: {err}")))?;
        rx.await
            .map_err(|_| GpuError::new("Friendly GPU map callback dropped"))?
            .map_err(|err| GpuError::new(format!("map Friendly results: {err}")))?;
        let mapped = mapped_slice
            .get_mapped_range()
            .map_err(|err| GpuError::new(format!("read mapped Friendly result: {err}")))?;
        let result_words = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
        drop(mapped);
        self.staging.unmap();
        let mut solved = Vec::with_capacity(banks.len());
        for word in result_words.chunks_exact(4) {
            if word[0] == 0 {
                solved.push(None);
            } else {
                let nonce = word[1];
                if u64::from(nonce) < u64::from(base) || u64::from(nonce) >= domain_end {
                    return Err(GpuError::new("Friendly GPU nonce outside bank window"));
                }
                solved.push(Some(nonce));
            }
        }
        Ok(solved)
    }

    /// Convenience adapter for a single solution bank.
    ///
    /// # Errors
    /// Propagates window validation, device submission and readback errors.
    pub async fn scan(
        &mut self,
        padded: &[u8; 128],
        bank: u8,
        base: u32,
        count: u32,
        threshold: u32,
    ) -> Result<Option<u32>, GpuError> {
        let mut outcomes = self
            .scan_banks(padded, &[bank], base, count, threshold)
            .await?;
        outcomes
            .pop()
            .ok_or_else(|| GpuError::new("missing single bank result"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blake2::{Blake2b256, Digest};
    use std::{hint::black_box, time::Instant};

    fn puzzle_input() -> [u8; 128] {
        let mut input = [0u8; 128];
        for (i, byte) in input[..64].iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
        input[12] = 1;
        input[14] = 16;
        input[15] = 128;
        input
    }

    fn cpu_matches(padded: &[u8; 128], bank: u8, nonce: u32, threshold: u32) -> bool {
        let mut candidate = *padded;
        candidate[120] = bank;
        candidate[124..128].copy_from_slice(&nonce.to_le_bytes());
        let digest = Blake2b256::digest(candidate);
        u32::from_le_bytes(digest[..4].try_into().expect("digest prefix")) < threshold
    }

    #[test]
    #[ignore = "requires a physical SHADER_INT64 GPU"]
    fn friendly_blake2b_gpu_matches_independent_cpu() -> anyhow::Result<()> {
        let gpu = super::super::sha512_tests::block_on(GpuContext::create())?;
        let mut solver = FriendlyBlake2bGpu::new(&gpu)?;
        let input = puzzle_input();
        for bank in [0, 1, 7, 255] {
            for base in [0, 1, 4_096, 65_534, u32::MAX - 127] {
                let result = super::super::sha512_tests::block_on(solver.scan(
                    &input,
                    bank,
                    base,
                    128,
                    u32::MAX,
                ))?;
                let nonce = result.expect("most BLAKE2b digests should match u32::MAX");
                assert!(cpu_matches(&input, bank, nonce, u32::MAX));
                assert!(u64::from(nonce) >= u64::from(base));
                assert!(u64::from(nonce) < u64::from(base) + 128);
            }
        }
        assert_eq!(
            super::super::sha512_tests::block_on(solver.scan(&input, 1, 0, 256, 0))?,
            None
        );
        // Cross-bank independence and staging alignment across all u8 bank IDs.
        let all_banks: Vec<u8> = (0..=u8::MAX).collect();
        let all = super::super::sha512_tests::block_on(solver.scan_banks(
            &input,
            &all_banks,
            u32::MAX - 127,
            128,
            u32::MAX,
        ))?;
        assert_eq!(all.len(), all_banks.len());
        for (bank, candidate) in all_banks.into_iter().zip(all) {
            let candidate = candidate.expect("every bank should find a wide-threshold proof");
            assert!(cpu_matches(&input, bank, candidate, u32::MAX));
            assert!(candidate >= u32::MAX - 127);
        }
        assert!(
            super::super::sha512_tests::block_on(solver.scan_banks(
                &input,
                &[1, 7, 255],
                0,
                1024,
                0
            ))?
            .iter()
            .all(Option::is_none)
        );
        assert!(
            super::super::sha512_tests::block_on(solver.scan_banks(
                &input,
                &[7, 7],
                0,
                64,
                u32::MAX
            ))
            .is_err()
        );

        let threshold = 1u32 << 24;
        let found =
            super::super::sha512_tests::block_on(solver.scan(&input, 5, 0, 65_536, threshold))?
                .expect("expected one 8-bit prefix hit in a 65k nonce window");
        assert!(cpu_matches(&input, 5, found, threshold));
        println!(
            "FRIENDLY_GPU_PROOF_PASS adapter={} nonce={found}",
            gpu.adapter()
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a physical SHADER_INT64 GPU"]
    fn friendly_blake2b_gpu_fixed_work_vs_rustcrypto() -> anyhow::Result<()> {
        let gpu = super::super::sha512_tests::block_on(GpuContext::create())?;
        let start_init = Instant::now();
        let mut solver = FriendlyBlake2bGpu::new(&gpu)?;
        let init_ms = start_init.elapsed().as_secs_f64() * 1e3;
        let input = puzzle_input();
        let bank = 3;
        let mut cpu_input = input;
        cpu_input[120] = bank;
        for count in [16_384u32, 65_536, 262_144] {
            for trial in 0..5 {
                let started = Instant::now();
                let result =
                    super::super::sha512_tests::block_on(solver.scan(&input, bank, 0, count, 1))?;
                assert!(result.is_none());
                let gpu_ns = started.elapsed().as_nanos();
                let started = Instant::now();
                for nonce in 0..count {
                    cpu_input[124..128].copy_from_slice(&nonce.to_le_bytes());
                    black_box(Blake2b256::digest(black_box(&cpu_input)));
                }
                let cpu_ns = started.elapsed().as_nanos();
                println!(
                    "FRIENDLY_GPU_BENCH {{\"adapter\":\"{}\",\"count\":{},\"trial\":{},\"gpu_ns\":{},\"cpu_ns\":{},\"gpu_rate\":{},\"cpu_rate\":{},\"init_ms\":{}}}",
                    gpu.adapter(),
                    count,
                    trial,
                    gpu_ns,
                    cpu_ns,
                    f64::from(count) * 1e9 / gpu_ns as f64,
                    f64::from(count) * 1e9 / cpu_ns as f64,
                    init_ms
                );
            }
        }
        Ok(())
    }
}
