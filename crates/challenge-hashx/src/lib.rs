#![deny(unsafe_code)]

//! Reusable HashX proof validation, native-JIT search and optional GPU backends.
//!
//! This crate owns HashX program construction and execution. Scheduling,
//! requests, routing and GPU work queues belong to the calling service.

/// HashX challenge parsing, shared validation and generated IR.
pub mod hashx;
/// Native HashX JIT search over the protocol's nonce space.
pub mod hashx_cpu_jit;
/// Vulkan and native SM75 CUDA backends.
#[cfg(feature = "gpu")]
pub mod hashx_gpu;
#[cfg(feature = "gpu")]
mod hashx_gpu_native;
#[cfg(feature = "gpu")]
mod hashx_sm75;
#[cfg(feature = "gpu")]
mod hashx_sm75_cuda;
