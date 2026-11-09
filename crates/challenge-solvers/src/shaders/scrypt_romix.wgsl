// RFC 7914 scrypt ROMix, one independent 128*r-byte chunk per GPU lane.
// All Salsa20/8 arithmetic operates on little-endian u32 words.
struct Params {
 n: u32,
 r: u32,
 lanes: u32,
 offset: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage,read_write> state: array<u32>;
@group(0) @binding(2) var<storage,read_write> scratch: array<u32>;
@group(0) @binding(3) var<storage,read_write> mixed: array<u32>;

// Four independent Salsa20 quarter-rounds per vec4; lane permutations
// turn the four column quarters into four row quarters without dynamic
// indexing of a sixteen-word mutable WGSL array.
fn rotl4(value: vec4<u32>, bits: u32) -> vec4<u32> {
 return (value << vec4<u32>(bits)) | (value >> vec4<u32>(32u-bits));
}

// Salsa20/8 over one 64-byte block (RFC7914), returning input + 8 rounds.
// Each vector component is one independent quarter-round, exposing ILP to
// the backend and eliminating dynamically indexed round-state arrays.
fn salsa20_8(input: array<u32,16>) -> array<u32,16> {
 var a=vec4<u32>(input[0],input[5],input[10],input[15]);
 var b=vec4<u32>(input[4],input[9],input[14],input[3]);
 var c=vec4<u32>(input[8],input[13],input[2],input[7]);
 var d=vec4<u32>(input[12],input[1],input[6],input[11]);
 for(var round=0u;round<4u;round++) {
  // Column round: four parallel quarter-rounds.
  b ^= rotl4(a+d,7u);
  c ^= rotl4(b+a,9u);
  d ^= rotl4(c+b,13u);
  a ^= rotl4(d+c,18u);

  // Column -> row permutation. The a-vector is unchanged.
  var rb=d.yzwx;
  var rc=c.zwxy;
  var rd=b.wxyz;

  // Row round: four parallel quarter-rounds.
  rb ^= rotl4(a+rd,7u);
  rc ^= rotl4(rb+a,9u);
  rd ^= rotl4(rc+rb,13u);
  a ^= rotl4(rd+rc,18u);

  // Row -> column permutation, inverse to the one above.
  b=rd.yzwx;
  c=rc.zwxy;
  d=rb.wxyz;
 }
 return array<u32,16>(
  a.x+input[0],d.y+input[1],c.z+input[2],b.w+input[3],
  b.x+input[4],a.y+input[5],d.z+input[6],c.w+input[7],
  c.x+input[8],b.y+input[9],a.z+input[10],d.w+input[11],
  d.x+input[12],c.y+input[13],b.z+input[14],a.w+input[15]
 );
}

fn blockmix(lane: u32) {
 let words_per_lane=32u*params.r;
 let base=(lane+params.offset)*words_per_lane;
 var x: array<u32,16>;
 for(var j=0u;j<16u;j++){
  x[j]=state[base+(2u*params.r-1u)*16u+j];
 }
 for(var i=0u;i<2u*params.r;i++){
  for(var j=0u;j<16u;j++){x[j]^=state[base+i*16u+j];}
  x=salsa20_8(x);
  let destination=(i/2u)+select(0u,params.r,(i&1u)!=0u);
  for(var j=0u;j<16u;j++){mixed[lane*words_per_lane+destination*16u+j]=x[j];}
 }
 for(var i=0u;i<words_per_lane;i++){state[base+i]=mixed[lane*words_per_lane+i];}
}

@compute @workgroup_size(64)
fn fill(@builtin(global_invocation_id) gid:vec3<u32>) {
 let lane=gid.x;
 if(lane>=params.lanes){return;}
 let words_per_lane=32u*params.r;
 let base=(lane+params.offset)*words_per_lane;
 for(var round=0u;round<params.n;round++){
  let address=(lane*params.n+round)*words_per_lane;
  for(var word=0u;word<words_per_lane;word++){
   scratch[address+word]=state[base+word];
  }
  blockmix(lane);
 }
}

@compute @workgroup_size(64)
fn mix(@builtin(global_invocation_id) gid:vec3<u32>) {
 let lane=gid.x;
 if(lane>=params.lanes){return;}
 let words_per_lane=32u*params.r;
 let base=(lane+params.offset)*words_per_lane;
 for(var round=0u;round<params.n;round++){
  // Integerify: the little-endian 64-bit integer beginning at the last
  // 64-byte chunk. N is a u32 power of two, so only its low 32 bits matter.
  let index=state[base+(2u*params.r-1u)*16u]&(params.n-1u);
  let address=(lane*params.n+index)*words_per_lane;
  for(var word=0u;word<words_per_lane;word++){
   state[base+word]^=scratch[address+word];
  }
  blockmix(lane);
 }
}
