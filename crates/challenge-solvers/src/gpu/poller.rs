//! On-demand native GPU polling runs on one dedicated sleeping thread per device.
//! Async callers await the poll result without blocking their executor.

use super::GpuError;
use futures_channel::oneshot;
use std::{sync::mpsc, thread};

type PollReply = oneshot::Sender<Result<(), GpuError>>;

pub(super) struct NativeGpuPoller {
    sender: mpsc::Sender<PollReply>,
}

impl NativeGpuPoller {
    pub(super) fn new(device: wgpu::Device) -> Result<Self, GpuError> {
        let (sender, receiver) = mpsc::channel::<PollReply>();
        thread::Builder::new()
            .name("challenge-wgpu-poll".to_owned())
            .spawn(move || {
                while let Ok(done) = receiver.recv() {
                    let result = device
                        .poll(wgpu::PollType::wait_indefinitely())
                        .map(|_| ())
                        .map_err(|e| GpuError::new(format!("poll GPU: {e}")));
                    let _ = done.send(result);
                }
            })
            .map_err(|e| GpuError::new(format!("spawn GPU polling thread: {e}")))?;
        Ok(Self { sender })
    }

    pub(super) async fn poll(&self) -> Result<(), GpuError> {
        let (sender, receiver) = oneshot::channel();
        self.sender
            .send(sender)
            .map_err(|_| GpuError::new("GPU polling thread has stopped"))?;
        receiver
            .await
            .map_err(|_| GpuError::new("GPU polling thread failed"))?
    }
}
