#!/usr/bin/env bash
# Reproduce the inert SM86 ELF shell with CUDA ptxas 13.1.115.
# Usage: build-sm86-template.sh /path/to/ptxas /absolute/path/to/output.cubin
set -euo pipefail
if [[ $# != 2 ]]; then echo "usage: $0 PTXAS OUTPUT_CUBIN" >&2; exit 2; fi
ptxas="$(realpath "$1")"
output="$(realpath -m "$2")"
scratch="$(mktemp -d "$(dirname "$output")/sm86-template.XXXXXXXX")"
trap 'rm -rf -- "$scratch"' EXIT
cd "$scratch"
cat > template.ptx <<'PTX'
.version 8.0
.target sm_86
.address_size 64
.visible .entry kern(
  .param .u64 solution,
  .param .u32 base,
  .param .u32 mask0,
  .param .u32 mask1
)
{
  .reg .u64 %addr;
  .reg .u32 %r<48>;
  ld.param.u64 %addr, [solution];
  ld.param.u32 %r0, [base];
  ld.param.u32 %r1, [mask0];
  ld.param.u32 %r2, [mask1];
PTX
# Volatile instructions force the register allocation and text capacity.
# None of these instructions survives into the resulting inert shell.
for ((i=3;i<30;i++)); do printf '  ld.volatile.global.u32 %%r%d, [%%addr];\n' "$i"; done >> template.ptx
for ((i=3;i<30;i++)); do printf '  st.volatile.global.u32 [%%addr], %%r%d;\n' "$i"; done >> template.ptx
for ((i=0;i<8192;i++)); do printf '  st.volatile.global.u32 [%%addr], %%r0;\n'; done >> template.ptx
printf '  st.volatile.global.u32 [%%addr], %%r1;\n  st.volatile.global.u32 [%%addr], %%r2;\n  ret;\n}\n' >> template.ptx
"$ptxas" -O3 -arch=sm_86 -v template.ptx -o template.raw.cubin
# Refuse a changed toolkit layout rather than patching arbitrary offsets.
readelf -SW template.raw.cubin > sections.txt 2>/dev/null
if ! rg -q '\.text\.kern\s+PROGBITS\s+0+\s+000700\s+020500\s' sections.txt ||
   ! rg -q '\.nv\.info\.kern\s+LOPROC\+0\s+0+\s+00047c\s+00006c\s' sections.txt; then
    echo 'unexpected SM86 ELF layout; inspect the new compiler output before updating the template' >&2
    exit 1
fi
expect_bytes() {
    local offset="$1" count="$2" expected="$3"
    local actual
    actual="$(od -An -v -tx1 -j "$offset" -N "$count" template.raw.cubin | tr -d '[:space:]')"
    [[ "$actual" == "$expected" ]] || { echo "unexpected compiler bytes at $offset" >&2; exit 1; }
}
expect_bytes $((0x4e0)) 8 041c0400f0030200
expect_bytes $((0x700+0x203f0)) 16 4d790000000000000000800300ea0f00
expect_bytes $((0x700+0x20400)) 16 47790000f0ffffffffff830300c00f00
expect_bytes $((0x700+0x20410)) 16 18790000000000000000000000c00f00
cp template.raw.cubin template.cubin
{
    dd if=template.raw.cubin bs=1 skip=$((0x700+0x203f0)) count=16 status=none
    for ((i=1;i<0x20400/16;i++)); do
        printf '\x18\x79\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc0\x0f\x00'
    done
} | dd of=template.cubin bs=16 seek=$((0x700/16)) conv=notrunc status=none
printf '\x00\x00\x00\x00' | dd of=template.cubin bs=1 seek=$((0x4e4)) conv=notrunc status=none
cp template.cubin "$output"
sha256sum "$output"
