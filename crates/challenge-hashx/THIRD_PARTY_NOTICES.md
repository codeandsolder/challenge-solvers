# Third-party notices

## TuringAs

`src/hashx_sm75.rs` derives NVIDIA SM75 instruction encodings and encoding layout details from TuringAs:
https://github.com/daadaada/turingas

MIT License

Copyright (c) 2020 Da Yan @ HKUST

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## hashwx-rs

Challenge-disasterer uses `codeandsolder/hashwx-rs` for the production HashWX and HashX native JIT implementations and for the immutable generated HashX IR consumed by accelerator backends:
https://github.com/codeandsolder/hashwx-rs

The crate combines original Rust backend work with compatibility code derived from tevador HashWX and the Tor Project Arti HashX implementation. It is distributed under LGPL-3.0-only; the corresponding license text is included at `THIRD_PARTY_LICENSES/hashwx-rs-LGPL-3.0-only.txt`, and detailed provenance is maintained in the dependency repository.
