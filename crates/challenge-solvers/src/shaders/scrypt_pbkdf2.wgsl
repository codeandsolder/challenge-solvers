// RFC 7914 PBKDF2-HMAC-SHA256 pre-/post-ROMix stages.
// The shared unrolled SHA-256 compression implementation is injected by Rust.
__SHA256_CORE__

struct KdfParams {
 n: u32,
 r: u32,
 p: u32,
 salt_len: u32,
 salt_blocks: u32,
 key_blocks: u32,
 b_words: u32,
 candidates: u32,
};
@group(0) @binding(0) var<uniform> params:KdfParams;
// HMAC SHA256 chaining states after the respective ipad/opad compression;
// P[16*candidate..16*candidate+7]: ipad, +8..+15: opad SHA states.
// P[16*candidates..]: shared padded salt||INT_BE32(1) U1 template.
@group(0) @binding(1) var<storage,read> P:array<u32>;
// ROMix B is a sequence of LE32 words, matching RFC byte representation.
@group(0) @binding(2) var<storage,read_write> B:array<u32>;
// The final derived key is stored as a BE32 digest word per u32.
@group(0) @binding(3) var<storage,read_write> Key:array<u32>;

fn bswap(x:u32)->u32 {
 return (x<<24u)|((x<<8u)&0x00ff0000u)|((x>>8u)&0x0000ff00u)|(x>>24u);
}

fn outer(inner:array<u32,8>,candidate:u32)->array<u32,8> {
 var w:array<u32,16>;
 for(var i=0u;i<16u;i++){w[i]=0u;}
 for(var i=0u;i<8u;i++){w[i]=inner[i];}
 w[8]=0x80000000u;
 w[15]=768u;
 let base=candidate*16u+8u;
 let opad=array<u32,8>(P[base],P[base+1u],P[base+2u],P[base+3u],P[base+4u],P[base+5u],P[base+6u],P[base+7u]);
 return compress(opad,w);
}

// First PBKDF2 invocation expands salt into B, 32-byte block at a time.
@compute @workgroup_size(64)
fn initial(@builtin(global_invocation_id) gid:vec3<u32>) {
 let digest_index=gid.x;
 let digests_per_candidate=params.b_words/8u;
 let candidate=digest_index/digests_per_candidate;
 if(candidate>=params.candidates){return;}
 let block_index=digest_index%digests_per_candidate+1u;
 let base=candidate*16u;
 var inner=array<u32,8>(P[base],P[base+1u],P[base+2u],P[base+3u],P[base+4u],P[base+5u],P[base+6u],P[base+7u]);
 for(var block=0u;block<params.salt_blocks;block++){
  var w:array<u32,16>;
  for(var i=0u;i<16u;i++){
   w[i]=P[16u*params.candidates+block*16u+i];
   let word_pos=block*64u+i*4u;
   if(word_pos<params.salt_len+4u&&word_pos+4u>params.salt_len) {
    for(var j=0u;j<4u;j++){
     let pos=word_pos+j;
     if(pos>=params.salt_len&&pos<params.salt_len+4u) {
      let shift=(3u-j)*8u;
      let index_shift=(3u-(pos-params.salt_len))*8u;
      let byte=(block_index>>index_shift)&255u;
      w[i]=(w[i]&~(255u<<shift))|(byte<<shift);
     }
    }
   }
  }
  inner=compress(inner,w);
 }
 let digest=outer(inner,candidate);
 // RFC B is byte-oriented; storing little-endian u32 also matches ROMix.
 for(var i=0u;i<8u;i++){
  B[digest_index*8u+i]=bswap(digest[i]);
 }
}

// Final PBKDF2 invocation hashes all mixed B blocks and produces the
// requested key. Each output block is independent; no host KDF work.
@compute @workgroup_size(64)
fn finish(@builtin(global_invocation_id) gid:vec3<u32>) {
 let key_index=gid.x;
 let candidate=key_index/params.key_blocks;
 if(candidate>=params.candidates){return;}
 let block_index=key_index%params.key_blocks+1u;
 let base=candidate*16u;
 var inner=array<u32,8>(P[base],P[base+1u],P[base+2u],P[base+3u],P[base+4u],P[base+5u],P[base+6u],P[base+7u]);
 for(var block=0u;block<params.b_words/16u;block++){
  var w:array<u32,16>;
  for(var i=0u;i<16u;i++){
   w[i]=bswap(B[candidate*params.b_words+block*16u+i]);
  }
  inner=compress(inner,w);
 }
 // B length is always a multiple of 128 bytes.  The 4-byte PBKDF2
 // index and SHA padding therefore require exactly one final block.
 var tail:array<u32,16>;
 for(var i=0u;i<16u;i++){tail[i]=0u;}
 tail[0]=block_index;
 tail[1]=0x80000000u;
 let message_len=params.b_words*4u+68u;
 tail[14]=message_len>>29u;
 tail[15]=message_len<<3u;
 inner=compress(inner,tail);
 let digest=outer(inner,candidate);
 for(var i=0u;i<8u;i++){Key[key_index*8u+i]=digest[i];}
}
