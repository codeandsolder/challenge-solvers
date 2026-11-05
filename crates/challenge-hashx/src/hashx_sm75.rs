// SM75 instruction encodings and layout details are derived from TuringAs.
// See THIRD_PARTY_NOTICES.md for attribution and its MIT license.

use crate::hashx::{GpuOp, GpuProgram, Instr};
use anyhow::Context;

const TEXT_OFFSET: usize = 0x380;
const TEXT_SIZE: usize = 0x20000;
const TEXT_WORDS: usize = TEXT_SIZE / 16;
// HashX generation has a fixed 512-instruction schedule: 160 Mul, 32 wide-mul,
// 16 Target/Branch pairs, and 288 normal ops. Even pessimistically expanding every
// wide-mul as signed, every normal op to its longest form, and all 16 one-shot replay
// regions, generated code stays far below this 8192-word shell. Dead words follow EXIT.
const TEMPLATE: &[u8] = include_bytes!("hashx_sm75_solve_template.cubin");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pred {
    idx: u8,
    neg: bool,
}

impl Pred {
    const PT: Self = Self { idx: 7, neg: false };
    const NPT: Self = Self { idx: 7, neg: true };
    const P0: Self = Self { idx: 0, neg: false };
    const P1: Self = Self { idx: 1, neg: false };
    const P2: Self = Self { idx: 2, neg: false };
    const P3: Self = Self { idx: 3, neg: false };
    const P5: Self = Self { idx: 5, neg: false };
    const P6: Self = Self { idx: 6, neg: false };

    fn bits(self) -> u8 {
        self.idx | (u8::from(self.neg) << 3)
    }
}

#[derive(Clone, Copy, Debug)]
enum Reg {
    R(u8),
    Z,
}

impl Reg {
    const fn bits(self) -> u8 {
        match self {
            Self::R(v) => v,
            Self::Z => 0xff,
        }
    }
}

const fn r(v: u8) -> Reg {
    Reg::R(v)
}

#[derive(Clone, Copy, Debug)]
enum Src1 {
    Reg(Reg),
    Imm(u32),
    Const { bank: u32, offset: u32 },
}

#[derive(Clone, Copy, Debug)]
struct Ctrl {
    wait: u8,
    read: u8,
    write: u8,
    yield_hint: bool,
    stall: u8,
}

impl Ctrl {
    const fn standard(stall: u8) -> Self {
        Self {
            wait: 0,
            read: 7,
            write: 7,
            yield_hint: false,
            stall,
        }
    }

    fn encode(self) -> u16 {
        u16::from(self.stall)
            | (u16::from(!self.yield_hint) << 4)
            | (u16::from(self.write) << 5)
            | (u16::from(self.read) << 8)
            | (u16::from(self.wait) << 11)
    }
}

fn base(opcode: u16, guard: Pred, ctrl: Ctrl) -> u128 {
    (u128::from(opcode) << 64)
        | (u128::from(guard.bits()) << 76)
        | (u128::from(ctrl.encode()) << 41)
}

fn set_icr(code: &mut u128, mode: u8) {
    *code &= !(1u128 << 73);
    *code |= u128::from(mode) << 72;
}

fn put_src1(code: &mut u128, src: Src1) {
    match src {
        Src1::Reg(reg) => *code |= u128::from(reg.bits()) << 96,
        Src1::Imm(imm) => {
            set_icr(code, 0x8);
            *code |= u128::from(imm) << 96;
        }
        Src1::Const { bank, offset } => {
            set_icr(code, 0xa);
            *code |= (u128::from(offset) << 6) << 96;
            *code |= (u128::from(bank) << 6) << 112;
        }
    }
}

fn put_pred(code: &mut u128, pred: Pred, shift: u32) {
    *code |= u128::from(pred.bits()) << shift;
}

fn mov(rd: Reg, src: Src1, guard: Pred, stall: u8) -> u128 {
    let mut code = base(0x202, guard, Ctrl::standard(stall));
    code |= u128::from(rd.bits()) << 80;
    put_src1(&mut code, src);
    code | 0xf00
}

#[expect(
    clippy::too_many_arguments,
    reason = "SASS instruction fields map one-to-one to encoded operands"
)]
fn iadd3(
    rd: Reg,
    pd0: Option<Pred>,
    pd1: Option<Pred>,
    rs0: Reg,
    rs0_neg: bool,
    src1: Src1,
    rs2: Reg,
    ps0: Option<Pred>,
    ps1: Option<Pred>,
    x: bool,
    guard: Pred,
    stall: u8,
) -> u128 {
    let mut code = base(0x210, guard, Ctrl::standard(stall));
    code |= u128::from(rd.bits()) << 80;
    code |= u128::from(rs0.bits()) << 88;
    code |= u128::from(rs2.bits());
    if rs0_neg {
        code |= 1u128 << 8;
    }
    if x {
        code |= 1u128 << 10;
    }
    match pd0 {
        Some(p) => put_pred(&mut code, p, 17),
        None => code |= 7u128 << 17,
    }
    match pd1 {
        Some(p) => put_pred(&mut code, p, 20),
        None => code |= 7u128 << 20,
    }
    match ps0 {
        Some(p) => put_pred(&mut code, p, 23),
        None => code |= 15u128 << 23,
    }
    match ps1 {
        Some(p) => put_pred(&mut code, p, 13),
        None => code |= 15u128 << 13,
    }
    put_src1(&mut code, src1);
    code
}

#[expect(
    clippy::too_many_arguments,
    reason = "SASS instruction fields map one-to-one to encoded operands"
)]
fn imad(
    wide: bool,
    x: bool,
    rd: Reg,
    pd0: Option<Pred>,
    rs0: Reg,
    src1: Src1,
    rs2: Reg,
    ps0: Option<Pred>,
    guard: Pred,
    ctrl: Ctrl,
) -> u128 {
    let mut code = base(if wide { 0x225 } else { 0x224 }, guard, ctrl);
    code |= u128::from(rd.bits()) << 80;
    code |= u128::from(rs0.bits()) << 88;
    code |= u128::from(rs2.bits());
    if x {
        code |= 1u128 << 10;
    }
    match pd0 {
        Some(p) => put_pred(&mut code, p, 17),
        None => code |= 7u128 << 17,
    }
    match ps0 {
        Some(p) => put_pred(&mut code, p, 23),
        None => code |= 15u128 << 23,
    }
    put_src1(&mut code, src1);
    code
}

#[derive(Clone, Copy)]
enum Cmp {
    Lt,
    Eq,
    Ne,
}

impl Cmp {
    const fn bits(self) -> u128 {
        match self {
            Self::Lt => 1u128 << 12,
            Self::Eq => 2u128 << 12,
            Self::Ne => 5u128 << 12,
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "SASS instruction fields map one-to-one to encoded operands"
)]
fn isetp(
    cmp: Cmp,
    u32_type: bool,
    pd0: Pred,
    pd1: Pred,
    rs0: Reg,
    src1: Src1,
    ps0: Pred,
    guard: Pred,
    stall: u8,
) -> u128 {
    let mut code = base(0x20c, guard, Ctrl::standard(stall));
    code |= cmp.bits();
    if !u32_type {
        code |= 1u128 << 9;
    }
    code |= 7u128 << 4;
    put_pred(&mut code, pd0, 17);
    put_pred(&mut code, pd1, 20);
    code |= u128::from(rs0.bits()) << 88;
    put_src1(&mut code, src1);
    put_pred(&mut code, ps0, 23);
    code
}

#[expect(
    clippy::too_many_arguments,
    reason = "SASS instruction fields map one-to-one to encoded operands"
)]
fn lop3(
    rd: Reg,
    rs0: Reg,
    src1: Src1,
    rs2: Reg,
    lut: u8,
    ps0: Pred,
    guard: Pred,
    stall: u8,
) -> u128 {
    let mut code = base(0x212, guard, Ctrl::standard(stall));
    code |= 7u128 << 17;
    code |= u128::from(rd.bits()) << 80;
    code |= u128::from(rs0.bits()) << 88;
    code |= u128::from(rs2.bits());
    code |= u128::from(lut) << 8;
    put_src1(&mut code, src1);
    put_pred(&mut code, ps0, 23);
    code
}

#[derive(Clone, Copy)]
enum ShfKind {
    LeftU64Hi,
    LeftU32,
    RightU64,
}

fn shf(kind: ShfKind, rd: Reg, rs0: Reg, amount: u8, rs2: Reg, guard: Pred, stall: u8) -> u128 {
    let mut code = base(0x219, guard, Ctrl::standard(stall));
    code |= u128::from(rd.bits()) << 80;
    code |= u128::from(rs0.bits()) << 88;
    code |= u128::from(rs2.bits());
    match kind {
        ShfKind::LeftU64Hi => code |= 1u128 << 16,
        ShfKind::LeftU32 => code |= 6u128 << 8,
        ShfKind::RightU64 => code |= 1u128 << 12,
    }
    put_src1(&mut code, Src1::Imm(u32::from(amount)));
    code
}

fn sel(rd: Reg, rs0: Reg, src1: Src1, ps0: Pred, guard: Pred, stall: u8) -> u128 {
    let mut code = base(0x207, guard, Ctrl::standard(stall));
    code |= u128::from(rd.bits()) << 80;
    code |= u128::from(rs0.bits()) << 88;
    put_src1(&mut code, src1);
    put_pred(&mut code, ps0, 23);
    code
}

#[derive(Clone, Copy)]
enum SpecialReg {
    TidX,
    CtaidX,
}

fn s2r(rd: Reg, sr: SpecialReg, write_barrier: u8) -> u128 {
    let ctrl = Ctrl {
        wait: 0,
        read: 7,
        write: write_barrier,
        yield_hint: false,
        stall: 5,
    };
    let mut code = base(0x919, Pred::PT, ctrl);
    code |= u128::from(rd.bits()) << 80;
    code |= (match sr {
        SpecialReg::TidX => 33u128,
        SpecialReg::CtaidX => 37u128,
    }) << 8;
    code
}

fn red_min_u32_gpu_strong(addr: Reg, src: Reg, guard: Pred, stall: u8) -> u128 {
    let mut code = base(0x98e, guard, Ctrl::standard(stall));
    code |= 1u128 << 8;
    code |= 1u128 << 23;
    code |= 2u128 << 15;
    code |= 2u128 << 13;
    code |= 1u128 << 20;
    code |= u128::from(addr.bits()) << 88;
    code |= u128::from(src.bits()) << 96;
    code
}

fn nop(stall: u8) -> u128 {
    base(0x918, Pred::PT, Ctrl::standard(stall))
}

fn tail_nop() -> u128 {
    base(
        0x918,
        Pred::PT,
        Ctrl {
            wait: 0,
            read: 7,
            write: 7,
            yield_hint: true,
            stall: 0,
        },
    )
}

fn exit(stall: u8) -> u128 {
    let mut code = base(0x94d, Pred::PT, Ctrl::standard(stall));
    put_pred(&mut code, Pred::PT, 23);
    code
}

fn tail_bra() -> u128 {
    let ctrl = Ctrl {
        wait: 0,
        read: 7,
        write: 7,
        yield_hint: true,
        stall: 0,
    };
    let mut code = base(0x947, Pred::PT, ctrl);
    set_icr(&mut code, 0x8);
    code |= (0xffff_fff0u128) << 96;
    code |= 0x3ffffu128;
    put_pred(&mut code, Pred::PT, 23);
    code
}

struct Emitter {
    words: Vec<u128>,
}

impl Emitter {
    fn new() -> Self {
        Self {
            words: Vec::with_capacity(TEXT_WORDS),
        }
    }
    fn emit(&mut self, word: u128) {
        self.words.push(word);
    }
    fn mov(&mut self, d: u8, src: Src1, guard: Pred) {
        self.emit(mov(r(d), src, guard, 5));
    }
    fn lop(&mut self, d: u8, a: u8, b: Src1, lut: u8, guard: Pred) {
        self.emit(lop3(r(d), r(a), b, Reg::Z, lut, Pred::NPT, guard, 5));
    }
    fn copy64(&mut self, dlo: u8, dhi: u8, slo: u8, shi: u8, guard: Pred) {
        self.mov(dlo, Src1::Reg(r(slo)), guard);
        self.mov(dhi, Src1::Reg(r(shi)), guard);
    }
    fn xor64_reg(&mut self, dlo: u8, dhi: u8, slo: u8, shi: u8, guard: Pred) {
        self.lop(dlo, dlo, Src1::Reg(r(slo)), 0x3c, guard);
        self.lop(dhi, dhi, Src1::Reg(r(shi)), 0x3c, guard);
    }
    fn xor64_const(&mut self, dlo: u8, dhi: u8, value: u64, guard: Pred) {
        let (lo, hi) = split_u64(value);
        if lo != 0 {
            self.lop(dlo, dlo, Src1::Imm(lo), 0x3c, guard);
        }
        if hi != 0 {
            self.lop(dhi, dhi, Src1::Imm(hi), 0x3c, guard);
        }
    }
    fn add64_reg(&mut self, dlo: u8, dhi: u8, slo: u8, shi: u8, guard: Pred) {
        self.emit(iadd3(
            r(dlo),
            Some(Pred::P0),
            None,
            r(dlo),
            false,
            Src1::Reg(r(slo)),
            Reg::Z,
            None,
            None,
            false,
            guard,
            4,
        ));
        self.emit(iadd3(
            r(dhi),
            None,
            None,
            r(dhi),
            false,
            Src1::Reg(r(shi)),
            Reg::Z,
            Some(Pred::P0),
            Some(Pred::NPT),
            true,
            guard,
            5,
        ));
    }
    fn add64_const(&mut self, dlo: u8, dhi: u8, value: u64, guard: Pred) {
        let (lo, hi) = split_u64(value);
        self.emit(iadd3(
            r(dlo),
            Some(Pred::P0),
            None,
            r(dlo),
            false,
            Src1::Imm(lo),
            Reg::Z,
            None,
            None,
            false,
            guard,
            4,
        ));
        self.emit(iadd3(
            r(dhi),
            None,
            None,
            r(dhi),
            false,
            Src1::Imm(hi),
            Reg::Z,
            Some(Pred::P0),
            Some(Pred::NPT),
            true,
            guard,
            5,
        ));
    }
    fn sub64(&mut self, dlo: u8, dhi: u8, slo: u8, shi: u8, guard: Pred) {
        self.emit(lop3(
            r(30),
            Reg::Z,
            Src1::Reg(r(shi)),
            Reg::Z,
            0x33,
            Pred::NPT,
            guard,
            5,
        ));
        self.emit(iadd3(
            r(dlo),
            Some(Pred::P0),
            None,
            r(slo),
            true,
            Src1::Reg(r(dlo)),
            Reg::Z,
            None,
            None,
            false,
            guard,
            4,
        ));
        self.emit(iadd3(
            r(dhi),
            None,
            None,
            r(dhi),
            false,
            Src1::Reg(r(30)),
            Reg::Z,
            Some(Pred::P0),
            Some(Pred::NPT),
            true,
            guard,
            5,
        ));
    }
    fn shl64_tmp(&mut self, slo: u8, shi: u8, n: u8, guard: Pred) {
        if n == 0 {
            self.copy64(24, 25, slo, shi, guard);
            return;
        }
        assert!(n < 32);
        self.emit(shf(ShfKind::LeftU64Hi, r(25), r(slo), n, r(shi), guard, 5));
        self.emit(shf(ShfKind::LeftU32, r(24), r(slo), n, Reg::Z, guard, 5));
    }
    fn ror64(&mut self, dlo: u8, dhi: u8, n: u8, guard: Pred) {
        let n = n % 64;
        if n == 0 {
            return;
        }
        if n == 32 {
            self.mov(24, Src1::Reg(r(dlo)), guard);
            self.mov(dlo, Src1::Reg(r(dhi)), guard);
            self.mov(dhi, Src1::Reg(r(24)), guard);
            return;
        }
        let k = n % 32;
        if n < 32 {
            self.emit(shf(ShfKind::RightU64, r(24), r(dlo), k, r(dhi), guard, 5));
            self.emit(shf(ShfKind::RightU64, r(25), r(dhi), k, r(dlo), guard, 5));
        } else {
            self.emit(shf(ShfKind::RightU64, r(24), r(dhi), k, r(dlo), guard, 5));
            self.emit(shf(ShfKind::RightU64, r(25), r(dlo), k, r(dhi), guard, 5));
        }
        self.mov(dlo, Src1::Reg(r(24)), guard);
        self.mov(dhi, Src1::Reg(r(25)), guard);
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "register-pair multiply lowering keeps explicit physical operands"
    )]
    fn mullo(&mut self, dlo: u8, dhi: u8, alo: u8, ahi: u8, blo: u8, bhi: u8, guard: Pred) {
        self.emit(imad(
            true,
            false,
            r(24),
            None,
            r(alo),
            Src1::Reg(r(blo)),
            Reg::Z,
            None,
            guard,
            Ctrl::standard(5),
        ));
        self.emit(imad(
            false,
            false,
            r(26),
            None,
            r(alo),
            Src1::Reg(r(bhi)),
            Reg::Z,
            None,
            guard,
            Ctrl::standard(5),
        ));
        self.emit(imad(
            false,
            false,
            r(27),
            None,
            r(ahi),
            Src1::Reg(r(blo)),
            Reg::Z,
            None,
            guard,
            Ctrl::standard(5),
        ));
        self.emit(iadd3(
            r(25),
            None,
            None,
            r(25),
            false,
            Src1::Reg(r(26)),
            r(27),
            None,
            None,
            false,
            guard,
            4,
        ));
        self.mov(dlo, Src1::Reg(r(24)), guard);
        self.mov(dhi, Src1::Reg(r(25)), guard);
    }
    fn umul_core(&mut self, alo: u8, ahi: u8, blo: u8, bhi: u8, guard: Pred) {
        self.emit(imad(
            true,
            false,
            r(24),
            None,
            r(blo),
            Src1::Reg(r(ahi)),
            Reg::Z,
            None,
            guard,
            Ctrl::standard(5),
        ));
        self.emit(imad(
            true,
            false,
            r(26),
            None,
            r(blo),
            Src1::Reg(r(alo)),
            Reg::Z,
            None,
            guard,
            Ctrl::standard(5),
        ));
        self.emit(imad(
            true,
            false,
            r(24),
            Some(Pred::P0),
            r(bhi),
            Src1::Reg(r(alo)),
            r(24),
            None,
            guard,
            Ctrl::standard(5),
        ));
        self.emit(iadd3(
            Reg::Z,
            Some(Pred::P1),
            None,
            r(27),
            false,
            Src1::Reg(r(24)),
            Reg::Z,
            None,
            None,
            false,
            guard,
            4,
        ));
        self.emit(iadd3(
            r(27),
            None,
            None,
            Reg::Z,
            false,
            Src1::Reg(Reg::Z),
            Reg::Z,
            Some(Pred::P0),
            Some(Pred::NPT),
            true,
            guard,
            5,
        ));
        self.mov(26, Src1::Reg(r(25)), guard);
        self.emit(imad(
            true,
            true,
            r(26),
            None,
            r(bhi),
            Src1::Reg(r(ahi)),
            r(26),
            Some(Pred::P1),
            guard,
            Ctrl::standard(8),
        ));
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "register-pair multiply lowering keeps explicit physical operands"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "signed high-half multiply lowering is kept linear to preserve SASS dataflow auditability"
    )]
    fn mulhi(
        &mut self,
        dlo: u8,
        dhi: u8,
        alo: u8,
        ahi: u8,
        blo: u8,
        bhi: u8,
        signed: bool,
        guard: Pred,
    ) {
        self.umul_core(alo, ahi, blo, bhi, guard);
        if signed {
            self.emit(isetp(
                Cmp::Lt,
                false,
                Pred::P2,
                Pred::PT,
                r(ahi),
                Src1::Reg(Reg::Z),
                Pred::PT,
                Pred::PT,
                6,
            ));
            self.emit(iadd3(
                r(28),
                Some(Pred::P0),
                None,
                r(blo),
                true,
                Src1::Reg(r(26)),
                Reg::Z,
                None,
                None,
                false,
                Pred::PT,
                4,
            ));
            self.emit(lop3(
                r(30),
                Reg::Z,
                Src1::Reg(r(bhi)),
                Reg::Z,
                0x33,
                Pred::NPT,
                Pred::PT,
                5,
            ));
            self.emit(iadd3(
                r(29),
                None,
                None,
                r(27),
                false,
                Src1::Reg(r(30)),
                Reg::Z,
                Some(Pred::P0),
                Some(Pred::NPT),
                true,
                Pred::PT,
                5,
            ));
            self.emit(sel(r(26), r(28), Src1::Reg(r(26)), Pred::P2, Pred::PT, 4));
            self.emit(sel(r(27), r(29), Src1::Reg(r(27)), Pred::P2, Pred::PT, 4));
            self.emit(isetp(
                Cmp::Lt,
                false,
                Pred::P2,
                Pred::PT,
                r(bhi),
                Src1::Reg(Reg::Z),
                Pred::PT,
                Pred::PT,
                6,
            ));
            self.emit(iadd3(
                r(28),
                Some(Pred::P0),
                None,
                r(alo),
                true,
                Src1::Reg(r(26)),
                Reg::Z,
                None,
                None,
                false,
                Pred::PT,
                4,
            ));
            self.emit(lop3(
                r(30),
                Reg::Z,
                Src1::Reg(r(ahi)),
                Reg::Z,
                0x33,
                Pred::NPT,
                Pred::PT,
                5,
            ));
            self.emit(iadd3(
                r(29),
                None,
                None,
                r(27),
                false,
                Src1::Reg(r(30)),
                Reg::Z,
                Some(Pred::P0),
                Some(Pred::NPT),
                true,
                Pred::PT,
                5,
            ));
            self.emit(sel(r(26), r(28), Src1::Reg(r(26)), Pred::P2, Pred::PT, 4));
            self.emit(sel(r(27), r(29), Src1::Reg(r(27)), Pred::P2, Pred::PT, 4));
        }
        self.mov(dlo, Src1::Reg(r(26)), guard);
        self.mov(dhi, Src1::Reg(r(27)), guard);
        self.mov(4, Src1::Reg(r(dlo)), guard);
    }
    fn sip(&mut self, regs: [u8; 4], guard: Pred) {
        let [a, b, c, d] = regs;
        let (alo, ahi) = pair(a);
        let (blo, bhi) = pair(b);
        let (clo, chi) = pair(c);
        let (dlo, dhi) = pair(d);
        self.add64_reg(alo, ahi, blo, bhi, guard);
        self.add64_reg(clo, chi, dlo, dhi, guard);
        self.ror64(blo, bhi, 64 - 13, guard);
        self.ror64(dlo, dhi, 64 - 16, guard);
        self.xor64_reg(blo, bhi, alo, ahi, guard);
        self.xor64_reg(dlo, dhi, clo, chi, guard);
        self.ror64(alo, ahi, 32, guard);
        self.add64_reg(clo, chi, blo, bhi, guard);
        self.add64_reg(alo, ahi, dlo, dhi, guard);
        self.ror64(blo, bhi, 64 - 17, guard);
        self.ror64(dlo, dhi, 64 - 21, guard);
        self.xor64_reg(blo, bhi, clo, chi, guard);
        self.xor64_reg(dlo, dhi, alo, ahi, guard);
        self.ror64(clo, chi, 32, guard);
    }
    fn op_instr(&mut self, ins: &Instr, guard: Pred) -> anyhow::Result<()> {
        let d = u8::try_from(ins.dst).context("HashX destination register exceeds u8")?;
        let src = u8::try_from(ins.src).context("HashX source register exceeds u8")?;
        match ins.kind()? {
            GpuOp::Mul => {
                let (dlo, dhi) = pair(d);
                let (slo, shi) = pair(src);
                self.mullo(dlo, dhi, dlo, dhi, slo, shi, guard);
            }
            GpuOp::UMulH | GpuOp::SMulH => {
                let (dlo, dhi) = pair(d);
                let (slo, shi) = pair(src);
                self.mulhi(
                    dlo,
                    dhi,
                    dlo,
                    dhi,
                    slo,
                    shi,
                    ins.kind()? == GpuOp::SMulH,
                    guard,
                );
            }
            GpuOp::AddShift => {
                let (slo, shi) = pair(src);
                self.shl64_tmp(
                    slo,
                    shi,
                    u8::try_from(ins.imm).context("HashX shift/rotate immediate exceeds u8")?,
                    guard,
                );
                let (dlo, dhi) = pair(d);
                self.add64_reg(dlo, dhi, 24, 25, guard);
            }
            GpuOp::AddConst => {
                let (dlo, dhi) = pair(d);
                self.add64_const(dlo, dhi, imm64(ins.imm), guard);
            }
            GpuOp::Sub => {
                let (dlo, dhi) = pair(d);
                let (slo, shi) = pair(src);
                self.sub64(dlo, dhi, slo, shi, guard);
            }
            GpuOp::Xor => {
                let (dlo, dhi) = pair(d);
                let (slo, shi) = pair(src);
                self.xor64_reg(dlo, dhi, slo, shi, guard);
            }
            GpuOp::XorConst => {
                let (dlo, dhi) = pair(d);
                self.xor64_const(dlo, dhi, imm64(ins.imm), guard);
            }
            GpuOp::Rotate => {
                let (dlo, dhi) = pair(d);
                self.ror64(
                    dlo,
                    dhi,
                    u8::try_from(ins.imm).context("HashX shift/rotate immediate exceeds u8")?,
                    guard,
                );
            }
            GpuOp::Target | GpuOp::Branch => {}
        }
        Ok(())
    }
}

const fn pair(i: u8) -> (u8, u8) {
    (8 + 2 * i, 9 + 2 * i)
}
fn imm64(v: u32) -> u64 {
    i64::from(v.cast_signed()).cast_unsigned()
}

const fn split_u64(value: u64) -> (u32, u32) {
    let bytes = value.to_le_bytes();
    (
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
    )
}

#[expect(
    clippy::too_many_lines,
    reason = "the generated SASS instruction stream is intentionally linear for instruction-order auditability"
)]
fn generate(source: &GpuProgram) -> anyhow::Result<Vec<u128>> {
    let key = source.key();
    let program = source.instructions();
    let mut e = Emitter::new();
    e.emit(mov(
        r(0),
        Src1::Const {
            bank: 0,
            offset: 0x160,
        },
        Pred::PT,
        5,
    ));
    e.emit(mov(
        r(1),
        Src1::Const {
            bank: 0,
            offset: 0x164,
        },
        Pred::PT,
        5,
    ));
    e.emit(mov(
        r(2),
        Src1::Const {
            bank: 0,
            offset: 0x168,
        },
        Pred::PT,
        5,
    ));
    e.emit(mov(
        r(6),
        Src1::Const {
            bank: 0,
            offset: 0x16c,
        },
        Pred::PT,
        5,
    ));
    e.emit(mov(
        r(7),
        Src1::Const {
            bank: 0,
            offset: 0x170,
        },
        Pred::PT,
        5,
    ));
    e.emit(s2r(r(3), SpecialReg::TidX, 1));
    e.emit(s2r(r(5), SpecialReg::CtaidX, 2));
    e.emit(imad(
        false,
        false,
        r(3),
        None,
        r(5),
        Src1::Imm(32),
        r(3),
        None,
        Pred::PT,
        Ctrl {
            wait: 6,
            read: 7,
            write: 7,
            yield_hint: false,
            stall: 5,
        },
    ));
    e.emit(iadd3(
        r(2),
        None,
        None,
        r(2),
        false,
        Src1::Reg(r(3)),
        Reg::Z,
        None,
        None,
        false,
        Pred::PT,
        4,
    ));
    for (i, &key) in key.iter().enumerate() {
        let pair_index = u8::try_from(i).context("HashX key register index exceeds u8")?;
        let (lo, hi) = pair(pair_index);
        let (key_lo, key_hi) = split_u64(key);
        e.mov(lo, Src1::Imm(key_lo), Pred::PT);
        e.mov(hi, Src1::Imm(key_hi), Pred::PT);
    }
    e.xor64_const(pair(1).0, pair(1).1, 0xee, Pred::PT);
    e.lop(pair(3).0, pair(3).0, Src1::Reg(r(2)), 0x3c, Pred::PT);
    for _ in 0..2 {
        e.sip([0, 1, 2, 3], Pred::PT);
    }
    e.lop(pair(0).0, pair(0).0, Src1::Reg(r(2)), 0x3c, Pred::PT);
    e.xor64_const(pair(2).0, pair(2).1, 0xee, Pred::PT);
    for _ in 0..4 {
        e.sip([0, 1, 2, 3], Pred::PT);
    }
    for i in 0..4 {
        let (dlo, dhi) = pair(i + 4);
        let (slo, shi) = pair(i);
        e.copy64(dlo, dhi, slo, shi, Pred::PT);
    }
    e.xor64_const(pair(5).0, pair(5).1, 0xdd, Pred::PT);
    for _ in 0..4 {
        e.sip([4, 5, 6, 7], Pred::PT);
    }
    e.emit(isetp(
        Cmp::Eq,
        true,
        Pred::P5,
        Pred::PT,
        Reg::Z,
        Src1::Reg(Reg::Z),
        Pred::PT,
        Pred::PT,
        6,
    ));
    e.mov(4, Src1::Reg(Reg::Z), Pred::PT);
    let mut target: Option<usize> = None;
    for (index, ins) in program.iter().enumerate() {
        match ins.kind()? {
            GpuOp::Target => target = Some(index),
            GpuOp::Branch => {
                let target = target.context("HashX branch without target")?;
                e.lop(5, 4, Src1::Imm(ins.imm), 0xc0, Pred::PT);
                e.emit(isetp(
                    Cmp::Eq,
                    true,
                    Pred::P6,
                    Pred::PT,
                    r(5),
                    Src1::Reg(Reg::Z),
                    Pred::P5,
                    Pred::PT,
                    6,
                ));
                e.emit(isetp(
                    Cmp::Ne,
                    true,
                    Pred::P5,
                    Pred::PT,
                    r(5),
                    Src1::Reg(Reg::Z),
                    Pred::P5,
                    Pred::PT,
                    8,
                ));
                for replay in &program[target + 1..index] {
                    if !matches!(replay.kind()?, GpuOp::Target | GpuOp::Branch) {
                        e.op_instr(replay, Pred::P6)?;
                    }
                }
            }
            _ => e.op_instr(ins, Pred::PT)?,
        }
    }
    e.add64_const(pair(0).0, pair(0).1, key[0], Pred::PT);
    e.add64_const(pair(1).0, pair(1).1, key[1], Pred::PT);
    e.add64_const(pair(6).0, pair(6).1, key[2], Pred::PT);
    e.add64_const(pair(7).0, pair(7).1, key[3], Pred::PT);
    e.sip([0, 1, 2, 3], Pred::PT);
    e.sip([4, 5, 6, 7], Pred::PT);
    e.mov(26, Src1::Reg(r(pair(0).0)), Pred::PT);
    e.mov(27, Src1::Reg(r(pair(0).1)), Pred::PT);
    e.lop(26, 26, Src1::Reg(r(pair(4).0)), 0x3c, Pred::PT);
    e.lop(27, 27, Src1::Reg(r(pair(4).1)), 0x3c, Pred::PT);
    e.lop(28, 26, Src1::Reg(r(6)), 0xc0, Pred::PT);
    e.lop(29, 27, Src1::Reg(r(7)), 0xc0, Pred::PT);
    e.lop(28, 28, Src1::Reg(r(29)), 0xfc, Pred::PT);
    e.emit(isetp(
        Cmp::Eq,
        true,
        Pred::P3,
        Pred::PT,
        r(28),
        Src1::Reg(Reg::Z),
        Pred::PT,
        Pred::PT,
        6,
    ));
    for _ in 0..6 {
        e.emit(nop(1));
    }
    e.emit(red_min_u32_gpu_strong(r(0), r(2), Pred::P3, 8));
    e.emit(exit(2));
    let natural_len = e.words.len();
    anyhow::ensure!(
        natural_len < TEXT_WORDS - 5,
        "sm_75 HashX kernel too large: {natural_len} words"
    );
    while e.words.len() < TEXT_WORDS - 5 {
        e.emit(nop(1));
    }
    e.emit(tail_bra());
    while e.words.len() < TEXT_WORDS {
        e.emit(tail_nop());
    }
    anyhow::ensure!(e.words.len() == TEXT_WORDS, "sm_75 text size mismatch");
    Ok(e.words)
}

fn patch_words(cubin: &mut [u8], words: &[u128]) {
    let text = &mut cubin[TEXT_OFFSET..TEXT_OFFSET + TEXT_SIZE];
    for (i, &word) in words.iter().enumerate() {
        let slot = &mut text[i * 16..(i + 1) * 16];
        let bytes = word.to_le_bytes();
        slot[..8].copy_from_slice(&bytes[8..]);
        slot[8..].copy_from_slice(&bytes[..8]);
    }
}

pub fn template_solve_cubin() -> Vec<u8> {
    TEMPLATE.to_vec()
}

pub fn build_solve_cubin(program: &GpuProgram) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        TEMPLATE.len() >= TEXT_OFFSET + TEXT_SIZE,
        "embedded sm_75 cubin template is truncated"
    );
    let words = generate(program)?;
    let mut cubin = TEMPLATE.to_vec();
    patch_words(&mut cubin, &words);
    Ok(cubin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn difficulty_masks_match_hash_byte_order() {
        assert_eq!(crate::hashx::difficulty_masks(1), (0x0000_0080, 0));
        assert_eq!(crate::hashx::difficulty_masks(8), (0x0000_00ff, 0));
        assert_eq!(crate::hashx::difficulty_masks(12), (0x0000_f0ff, 0));
        assert_eq!(crate::hashx::difficulty_masks(32), (u32::MAX, 0));
        assert_eq!(crate::hashx::difficulty_masks(40), (u32::MAX, 0x0000_00ff));
        assert_eq!(crate::hashx::difficulty_masks(64), (u32::MAX, u32::MAX));
    }

    #[test]
    fn generated_programs_fit_fixed_text() -> anyhow::Result<()> {
        let exit_word = exit(2);
        let mut max_words = 0_usize;
        let mut max_seed = 0_u32;
        let cases = std::env::var("HASHX_SM75_FIT_CASES")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(1024);
        for seed in 0_u32..cases {
            let seed_bytes = seed.to_le_bytes();
            let (_, program) = crate::hashx::describe(&seed_bytes)?;
            let words = generate(&program)?;
            let used = words
                .iter()
                .position(|&word| word == exit_word)
                .context("generated sm_75 kernel has no EXIT")?
                + 1;
            if used > max_words {
                max_words = used;
                max_seed = seed;
            }
        }
        eprintln!("max generated sm_75 words={max_words} seed={max_seed}");
        assert!(max_words < TEXT_WORDS - 5);
        Ok(())
    }

    #[test]
    fn reference_programs_match_turingas_cubins() -> anyhow::Result<()> {
        use sha2::{Digest, Sha256};

        let cases: &[(&[u8], &str)] = &[
            (
                &[0x00],
                "002b9b44c1482648bc01cb51c4cb88a3dba8780ad08475c9ebf35c0f0d453677",
            ),
            (
                &[0x01],
                "db2400150539d67a262a9e82844618a47ca6beb177abb8cd54650b254e42aa3d",
            ),
            (
                &[0xde, 0xad, 0xbe, 0xef],
                "e2a56ec702a7106d097ec8e83f560e6a09a7f87c36d478eb8c2b314d40de9625",
            ),
            (
                &[0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef],
                "54cde4df9b7f850d6d568239b0c35226024ca53ba227c126c48fbe130f9bfca4",
            ),
        ];
        for (seed, expected) in cases {
            let (_, program) = crate::hashx::describe(seed)?;
            let cubin = build_solve_cubin(&program)?;
            assert_eq!(hex::encode(Sha256::digest(cubin)), *expected);
        }
        Ok(())
    }
}
