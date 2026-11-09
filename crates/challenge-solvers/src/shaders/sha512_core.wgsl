// SHA-512/384 80-round native-u64 compression (SHADER_INT64).
// Round constants are stored as u32 halves to avoid constant u64 shift parsing.
const SHA512_K_HI:array<u32,80>=array<u32,80>(
 0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
 0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
 0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
 0xca273eceu, 0xd186b8c7u, 0xeada7dd6u, 0xf57d4f7fu,
 0x06f067aau, 0x0a637dc5u, 0x113f9804u, 0x1b710b35u,
 0x28db77f5u, 0x32caab7bu, 0x3c9ebe0au, 0x431d67c4u,
 0x4cc5d4beu, 0x597f299cu, 0x5fcb6fabu, 0x6c44198cu
);
const SHA512_K_LO:array<u32,80>=array<u32,80>(
 0xd728ae22u, 0x23ef65cdu, 0xec4d3b2fu, 0x8189dbbcu,
 0xf348b538u, 0xb605d019u, 0xaf194f9bu, 0xda6d8118u,
 0xa3030242u, 0x45706fbeu, 0x4ee4b28cu, 0xd5ffb4e2u,
 0xf27b896fu, 0x3b1696b1u, 0x25c71235u, 0xcf692694u,
 0x9ef14ad2u, 0x384f25e3u, 0x8b8cd5b5u, 0x77ac9c65u,
 0x592b0275u, 0x6ea6e483u, 0xbd41fbd4u, 0x831153b5u,
 0xee66dfabu, 0x2db43210u, 0x98fb213fu, 0xbeef0ee4u,
 0x3da88fc2u, 0x930aa725u, 0xe003826fu, 0x0a0e6e70u,
 0x46d22ffcu, 0x5c26c926u, 0x5ac42aedu, 0x9d95b3dfu,
 0x8baf63deu, 0x3c77b2a8u, 0x47edaee6u, 0x1482353bu,
 0x4cf10364u, 0xbc423001u, 0xd0f89791u, 0x0654be30u,
 0xd6ef5218u, 0x5565a910u, 0x5771202au, 0x32bbd1b8u,
 0xb8d2d0c8u, 0x5141ab53u, 0xdf8eeb99u, 0xe19b48a8u,
 0xc5c95a63u, 0xe3418acbu, 0x7763e373u, 0xd6b2b8a3u,
 0x5defb2fcu, 0x43172f60u, 0xa1f0ab72u, 0x1a6439ecu,
 0x23631e28u, 0xde82bde9u, 0xb2c67915u, 0xe372532bu,
 0xea26619cu, 0x21c0c207u, 0xcde0eb1eu, 0xee6ed178u,
 0x72176fbau, 0xa2c898a6u, 0xbef90daeu, 0x131c471bu,
 0x23047d84u, 0x40c72493u, 0x15c9bebcu, 0x9c100d4cu,
 0xcb3e42b6u, 0xfc657e2au, 0x3ad6faecu, 0x4a475817u
);
fn sha512_ror(x:u64,n:u32)->u64{
 return (x>>n)|(x<<(64u-n));
}
fn sha512_compress(state:array<u64,8>,block:array<u64,16>)->array<u64,8>{
 var w:array<u64,80>;
 for(var i=0u;i<16u;i=i+1u){w[i]=block[i];}
 for(var i=16u;i<80u;i=i+1u){
  let s0=sha512_ror(w[i-15u],1u)^sha512_ror(w[i-15u],8u)^(w[i-15u]>>7u);
  let s1=sha512_ror(w[i-2u],19u)^sha512_ror(w[i-2u],61u)^(w[i-2u]>>6u);
  w[i]=w[i-16u]+s0+w[i-7u]+s1;
 }
 var a=state[0];var b=state[1];var c=state[2];var d=state[3];
 var e=state[4];var f=state[5];var g=state[6];var h=state[7];
 for(var i=0u;i<80u;i=i+1u){
  let s1=sha512_ror(e,14u)^sha512_ror(e,18u)^sha512_ror(e,41u);
  let ch=(e&f)^(~e&g);
  let round_k=(u64(SHA512_K_HI[i])<<32u)|u64(SHA512_K_LO[i]);
  let t1=h+s1+ch+round_k+w[i];
  let s0=sha512_ror(a,28u)^sha512_ror(a,34u)^sha512_ror(a,39u);
  let maj=(a&b)^(a&c)^(b&c);
  let t2=s0+maj;
  h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
 }
 return array<u64,8>(
  state[0]+a,state[1]+b,state[2]+c,state[3]+d,
  state[4]+e,state[5]+f,state[6]+g,state[7]+h
 );
}
