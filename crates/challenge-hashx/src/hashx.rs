use anyhow::Context;
#[cfg(feature = "gpu")]
use anyhow::anyhow;
#[cfg(feature = "gpu")]
use hashwx::HashXInstruction;
use hashwx::{HashX, HashXError, HashXProgram};
use serde::Deserialize;

use challenge_protocols::anubis::Answer;

#[derive(Debug, Clone, Deserialize)]
pub struct Challenge {
    pub data: String,
    pub difficulty: u32,
}

impl Challenge {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            (1..=64).contains(&self.difficulty),
            "difficulty must be in 1..=64"
        );
        let bytes = hex::decode(&self.data).context("HashX challenge data must be hex")?;
        anyhow::ensure!(
            !bytes.is_empty() && bytes.len() <= 4096,
            "HashX challenge data length out of range"
        );
        Ok(())
    }
}

#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum GpuOp {
    Mul = 0,
    UMulH = 1,
    SMulH = 2,
    AddShift = 3,
    AddConst = 4,
    Sub = 5,
    Xor = 6,
    XorConst = 7,
    Rotate = 8,
    Target = 9,
    Branch = 10,
}

#[cfg(feature = "gpu")]
impl From<GpuOp> for u32 {
    fn from(value: GpuOp) -> Self {
        value as Self
    }
}

#[cfg(feature = "gpu")]
impl TryFrom<u32> for GpuOp {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Mul),
            1 => Ok(Self::UMulH),
            2 => Ok(Self::SMulH),
            3 => Ok(Self::AddShift),
            4 => Ok(Self::AddConst),
            5 => Ok(Self::Sub),
            6 => Ok(Self::Xor),
            7 => Ok(Self::XorConst),
            8 => Ok(Self::Rotate),
            9 => Ok(Self::Target),
            10 => Ok(Self::Branch),
            other => Err(anyhow!("unknown HashX GPU opcode {other}")),
        }
    }
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "gpu", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct Instr {
    pub(crate) op: u32,
    pub(crate) dst: u32,
    pub(crate) src: u32,
    pub(crate) imm: u32,
}

#[cfg(feature = "gpu")]
impl Instr {
    pub(crate) fn kind(self) -> anyhow::Result<GpuOp> {
        self.op.try_into()
    }
}

#[cfg(feature = "gpu")]
#[derive(Clone, Debug)]
pub struct GpuProgram {
    key: [u64; 4],
    instructions: Vec<Instr>,
}

#[cfg(feature = "gpu")]
impl GpuProgram {
    fn new(key: [u64; 4], instructions: Vec<Instr>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            instructions.len() == 512,
            "expected 512 HashX instructions, got {}",
            instructions.len()
        );
        let mut target = None;
        for (index, instruction) in instructions.iter().copied().enumerate() {
            let kind = instruction.kind()?;
            match kind {
                GpuOp::Mul | GpuOp::UMulH | GpuOp::SMulH | GpuOp::Sub | GpuOp::Xor => {
                    anyhow::ensure!(
                        instruction.dst < 8 && instruction.src < 8,
                        "HashX GPU register out of range at instruction {index}"
                    );
                }
                GpuOp::AddShift => {
                    anyhow::ensure!(
                        instruction.dst < 8 && instruction.src < 8,
                        "HashX GPU register out of range at instruction {index}"
                    );
                    anyhow::ensure!(
                        instruction.imm < 64,
                        "HashX GPU AddShift amount out of range at instruction {index}"
                    );
                }
                GpuOp::AddConst | GpuOp::XorConst | GpuOp::Rotate => {
                    anyhow::ensure!(
                        instruction.dst < 8,
                        "HashX GPU destination register out of range at instruction {index}"
                    );
                    if kind == GpuOp::Rotate {
                        anyhow::ensure!(
                            instruction.imm < 64,
                            "HashX GPU rotate amount out of range at instruction {index}"
                        );
                    }
                }
                GpuOp::Target => target = Some(index),
                GpuOp::Branch => {
                    let target_index =
                        target.context("HashX GPU branch without preceding target")?;
                    anyhow::ensure!(
                        target_index < index,
                        "HashX GPU branch target is not before branch at instruction {index}"
                    );
                    target = None;
                }
            }
        }
        anyhow::ensure!(target.is_none(), "HashX GPU target without matching branch");
        Ok(Self { key, instructions })
    }

    pub(crate) const fn key(&self) -> &[u64; 4] {
        &self.key
    }

    pub(crate) fn instructions(&self) -> &[Instr] {
        &self.instructions
    }
}

#[cfg(feature = "gpu")]
const fn prefix_mask(difficulty: u32) -> u32 {
    if difficulty >= 32 {
        return u32::MAX;
    }
    let mut mask = 0_u32;
    let full = difficulty / 8;
    let rem = difficulty % 8;
    let mut i = 0;
    while i < full {
        mask |= 0xff << (8 * i);
        i += 1;
    }
    if rem != 0 {
        mask |= ((0xff << (8 - rem)) & 0xff) << (8 * full);
    }
    mask
}

#[cfg(feature = "gpu")]
pub fn difficulty_masks(difficulty: u32) -> (u32, u32) {
    let low = prefix_mask(difficulty.min(32));
    let high = if difficulty > 32 {
        prefix_mask(difficulty - 32)
    } else {
        0
    };
    (low, high)
}

pub fn build_compiled_seed(seed: &[u8]) -> anyhow::Result<(Vec<u8>, HashX, HashXProgram)> {
    let mut candidate = seed.to_vec();
    let mut counter = 0_u32;
    loop {
        match HashXProgram::generate(&candidate) {
            Ok(program) => {
                let compiled = HashX::compile_program(&program).context("compile HashX program")?;
                return Ok((candidate, compiled, program));
            }
            Err(HashXError::ProgramConstraints) => {}
            Err(error) => return Err(error).context("generate HashX program"),
        }
        candidate.clear();
        candidate.extend_from_slice(seed);
        candidate.extend_from_slice(&counter.to_le_bytes());
        counter = counter.wrapping_add(1);
        anyhow::ensure!(counter != 0, "HashX seed search exhausted");
    }
}

#[cfg(feature = "gpu")]
pub fn describe(seed: &[u8]) -> anyhow::Result<(HashX, GpuProgram)> {
    let (_usable_seed, compiled, generated) = build_compiled_seed(seed)?;
    let mut instructions = Vec::with_capacity(generated.instructions().len());
    for instruction in generated.instructions() {
        let instruction = match instruction {
            HashXInstruction::Mul { dst, src } => Instr {
                op: u32::from(GpuOp::Mul),
                dst: u32::from(dst.index()),
                src: u32::from(src.index()),
                imm: 0,
            },
            HashXInstruction::UMulH { dst, src } => Instr {
                op: u32::from(GpuOp::UMulH),
                dst: u32::from(dst.index()),
                src: u32::from(src.index()),
                imm: 0,
            },
            HashXInstruction::SMulH { dst, src } => Instr {
                op: u32::from(GpuOp::SMulH),
                dst: u32::from(dst.index()),
                src: u32::from(src.index()),
                imm: 0,
            },
            HashXInstruction::AddShift {
                dst,
                src,
                left_shift,
            } => Instr {
                op: u32::from(GpuOp::AddShift),
                dst: u32::from(dst.index()),
                src: u32::from(src.index()),
                imm: u32::from(*left_shift),
            },
            HashXInstruction::AddConst { dst, src } => Instr {
                op: u32::from(GpuOp::AddConst),
                dst: u32::from(dst.index()),
                src: 0,
                imm: (*src).cast_unsigned(),
            },
            HashXInstruction::Sub { dst, src } => Instr {
                op: u32::from(GpuOp::Sub),
                dst: u32::from(dst.index()),
                src: u32::from(src.index()),
                imm: 0,
            },
            HashXInstruction::Xor { dst, src } => Instr {
                op: u32::from(GpuOp::Xor),
                dst: u32::from(dst.index()),
                src: u32::from(src.index()),
                imm: 0,
            },
            HashXInstruction::XorConst { dst, src } => Instr {
                op: u32::from(GpuOp::XorConst),
                dst: u32::from(dst.index()),
                src: 0,
                imm: (*src).cast_unsigned(),
            },
            HashXInstruction::Rotate { dst, right_rotate } => Instr {
                op: u32::from(GpuOp::Rotate),
                dst: u32::from(dst.index()),
                src: 0,
                imm: u32::from(*right_rotate),
            },
            HashXInstruction::Target => Instr {
                op: u32::from(GpuOp::Target),
                dst: 0,
                src: 0,
                imm: 0,
            },
            HashXInstruction::Branch { mask } => Instr {
                op: u32::from(GpuOp::Branch),
                dst: 0,
                src: 0,
                imm: *mask,
            },
        };
        instructions.push(instruction);
    }
    Ok((
        compiled,
        GpuProgram::new(generated.register_key(), instructions)?,
    ))
}

pub fn verify(challenge: &Challenge, answer: &Answer) -> anyhow::Result<bool> {
    challenge.validate()?;
    let seed = hex::decode(&challenge.data).context("decode HashX challenge")?;
    let (_usable_seed, verifier, _program) = build_compiled_seed(&seed)?;
    let nonce = answer
        .nonce
        .parse::<u64>()
        .context("parse HashX answer nonce")?;
    let hash = verifier.hash_bytes(nonce);
    Ok(hex::encode(hash) == answer.hash && valid_prefix(&hash, challenge.difficulty))
}

pub fn valid_prefix(hash: &[u8; 32], difficulty: u32) -> bool {
    if difficulty > 256 {
        return false;
    }
    let Ok(whole) = usize::try_from(difficulty / 8) else {
        return false;
    };
    let rem = difficulty % 8;
    hash[..whole].iter().all(|&byte| byte == 0)
        && (rem == 0 || hash[whole] & (0xff_u8 << (8 - rem)) == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_prefix_rejects_difficulty_wider_than_the_digest() {
        assert!(!valid_prefix(&[0_u8; 32], 257));
        assert!(valid_prefix(&[0_u8; 32], 256));
    }

    #[test]
    fn rejected_seed_retries_with_one_authoritative_program() -> anyhow::Result<()> {
        let seed = [0xf9_u8, 0x05, 0x00, 0x00];
        assert!(matches!(
            HashXProgram::generate(&seed),
            Err(HashXError::ProgramConstraints)
        ));

        let (usable_seed, compiled, program) = build_compiled_seed(&seed)?;
        assert_ne!(usable_seed.as_slice(), seed);
        assert!(usable_seed.starts_with(&seed));
        assert_eq!(usable_seed.len(), seed.len() + size_of::<u32>());

        let interpreted = HashX::interpret_program(&program);
        for nonce in [0, 1, 123_456, u64::MAX] {
            assert_eq!(compiled.hash_bytes(nonce), interpreted.hash_bytes(nonce));
        }
        Ok(())
    }
}
