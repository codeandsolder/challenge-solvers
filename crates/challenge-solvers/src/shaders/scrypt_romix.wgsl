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

fn rotl(x: u32, shift: u32) -> u32 {
 return (x << shift) | (x >> (32u - shift));
}

// Salsa20/8 core, exactly four column+row double rounds.
fn salsa20_8(input: array<u32,16>) -> array<u32,16> {
 var x = input;
 for(var round=0u;round<4u;round++) {
  x[ 4] ^= rotl(x[ 0]+x[12], 7u);
  x[ 8] ^= rotl(x[ 4]+x[ 0], 9u);
  x[12] ^= rotl(x[ 8]+x[ 4],13u);
  x[ 0] ^= rotl(x[12]+x[ 8],18u);
  x[ 9] ^= rotl(x[ 5]+x[ 1], 7u);
  x[13] ^= rotl(x[ 9]+x[ 5], 9u);
  x[ 1] ^= rotl(x[13]+x[ 9],13u);
  x[ 5] ^= rotl(x[ 1]+x[13],18u);
  x[14] ^= rotl(x[10]+x[ 6], 7u);
  x[ 2] ^= rotl(x[14]+x[10], 9u);
  x[ 6] ^= rotl(x[ 2]+x[14],13u);
  x[10] ^= rotl(x[ 6]+x[ 2],18u);
  x[ 3] ^= rotl(x[15]+x[11], 7u);
  x[ 7] ^= rotl(x[ 3]+x[15], 9u);
  x[11] ^= rotl(x[ 7]+x[ 3],13u);
  x[15] ^= rotl(x[11]+x[ 7],18u);

  x[ 1] ^= rotl(x[ 0]+x[ 3], 7u);
  x[ 2] ^= rotl(x[ 1]+x[ 0], 9u);
  x[ 3] ^= rotl(x[ 2]+x[ 1],13u);
  x[ 0] ^= rotl(x[ 3]+x[ 2],18u);
  x[ 6] ^= rotl(x[ 5]+x[ 4], 7u);
  x[ 7] ^= rotl(x[ 6]+x[ 5], 9u);
  x[ 4] ^= rotl(x[ 7]+x[ 6],13u);
  x[ 5] ^= rotl(x[ 4]+x[ 7],18u);
  x[11] ^= rotl(x[10]+x[ 9], 7u);
  x[ 8] ^= rotl(x[11]+x[10], 9u);
  x[ 9] ^= rotl(x[ 8]+x[11],13u);
  x[10] ^= rotl(x[ 9]+x[ 8],18u);
  x[12] ^= rotl(x[15]+x[14], 7u);
  x[13] ^= rotl(x[12]+x[15], 9u);
  x[14] ^= rotl(x[13]+x[12],13u);
  x[15] ^= rotl(x[14]+x[13],18u);
 }
 for(var i=0u;i<16u;i++){x[i]+=input[i];}
 return x;
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
