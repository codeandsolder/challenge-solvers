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

// RFC7914 requires N to be a power of two >=2, so each phase consists
// of complete two-round pairs. Pin each round's input/output storage
// direction outside the word loop rather than selecting it on every read
// and write. Scratch XOR is fused into BlockMix's input words.
// The two directed functions intentionally share the same Salsa20/8 core.

fn blockmix_state_to_mixed(lane:u32,v_base:u32,xor_v:bool) {
 let words=32u*params.r;
 let base=(lane+params.offset)*words;
 let aux=lane*words;
 let last=(2u*params.r-1u)*16u;
 var x:array<u32,16>;
 for(var j=0u;j<16u;j++) {
  x[j]=state[base+last+j];
  if(xor_v){x[j]^=scratch[v_base+last+j];}
 }
 for(var i=0u;i<2u*params.r;i++) {
  for(var j=0u;j<16u;j++) {
   let pos=i*16u+j;
   var word=state[base+pos];
   if(xor_v){word^=scratch[v_base+pos];}
   x[j]^=word;
  }
  x=salsa20_8(x);
  let destination=(i/2u)+select(0u,params.r,(i&1u)!=0u);
  for(var j=0u;j<16u;j++) {
   mixed[aux+destination*16u+j]=x[j];
  }
 }
}

fn blockmix_mixed_to_state(lane:u32,v_base:u32,xor_v:bool) {
 let words=32u*params.r;
 let base=(lane+params.offset)*words;
 let aux=lane*words;
 let last=(2u*params.r-1u)*16u;
 var x:array<u32,16>;
 for(var j=0u;j<16u;j++) {
  x[j]=mixed[aux+last+j];
  if(xor_v){x[j]^=scratch[v_base+last+j];}
 }
 for(var i=0u;i<2u*params.r;i++) {
  for(var j=0u;j<16u;j++) {
   let pos=i*16u+j;
   var word=mixed[aux+pos];
   if(xor_v){word^=scratch[v_base+pos];}
   x[j]^=word;
  }
  x=salsa20_8(x);
  let destination=(i/2u)+select(0u,params.r,(i&1u)!=0u);
  for(var j=0u;j<16u;j++) {
   state[base+destination*16u+j]=x[j];
  }
 }
}

@compute @workgroup_size(64)
fn fill(@builtin(global_invocation_id) gid:vec3<u32>) {
 let lane=gid.x;
 if(lane>=params.lanes){return;}
 let words=32u*params.r;
 let base=(lane+params.offset)*words;
 let aux=lane*words;
 // Two rounds are needed to return state to B. Scratch V is populated
 // from the corresponding input before each BlockMix.
 for(var round=0u;round<params.n;round+=2u) {
  let first=(lane*params.n+round)*words;
  for(var word=0u;word<words;word++) {
   scratch[first+word]=state[base+word];
  }
  blockmix_state_to_mixed(lane,0u,false);

  let second=first+words;
  for(var word=0u;word<words;word++) {
   scratch[second+word]=mixed[aux+word];
  }
  blockmix_mixed_to_state(lane,0u,false);
 }
}

@compute @workgroup_size(64)
fn mix(@builtin(global_invocation_id) gid:vec3<u32>) {
 let lane=gid.x;
 if(lane>=params.lanes){return;}
 let words=32u*params.r;
 let base=(lane+params.offset)*words;
 let aux=lane*words;
 let last=(2u*params.r-1u)*16u;
 for(var round=0u;round<params.n;round+=2u) {
  let first_index=state[base+last]&(params.n-1u);
  let first=(lane*params.n+first_index)*words;
  blockmix_state_to_mixed(lane,first,true);

  let second_index=mixed[aux+last]&(params.n-1u);
  let second=(lane*params.n+second_index)*words;
  blockmix_mixed_to_state(lane,second,true);
 }
}
