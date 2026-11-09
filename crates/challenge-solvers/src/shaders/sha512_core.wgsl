// SHA-512 compression shared by SHA-512 and SHA-384.
// Uses native shader u64 operations; request SHADER_INT64 before compiling.
// No protocol framing is assumed: accepts any chaining state and 1024-bit block.
const SHA512_K:array<u64,80>=array<u64,80>(
 ((u64(0x428a2f98u)<<u64(32u))|u64(0xd728ae22u)), ((u64(0x71374491u)<<u64(32u))|u64(0x23ef65cdu)), ((u64(0xb5c0fbcfu)<<u64(32u))|u64(0xec4d3b2fu)), ((u64(0xe9b5dba5u)<<u64(32u))|u64(0x8189dbbcu)),
 ((u64(0x3956c25bu)<<u64(32u))|u64(0xf348b538u)), ((u64(0x59f111f1u)<<u64(32u))|u64(0xb605d019u)), ((u64(0x923f82a4u)<<u64(32u))|u64(0xaf194f9bu)), ((u64(0xab1c5ed5u)<<u64(32u))|u64(0xda6d8118u)),
 ((u64(0xd807aa98u)<<u64(32u))|u64(0xa3030242u)), ((u64(0x12835b01u)<<u64(32u))|u64(0x45706fbeu)), ((u64(0x243185beu)<<u64(32u))|u64(0x4ee4b28cu)), ((u64(0x550c7dc3u)<<u64(32u))|u64(0xd5ffb4e2u)),
 ((u64(0x72be5d74u)<<u64(32u))|u64(0xf27b896fu)), ((u64(0x80deb1feu)<<u64(32u))|u64(0x3b1696b1u)), ((u64(0x9bdc06a7u)<<u64(32u))|u64(0x25c71235u)), ((u64(0xc19bf174u)<<u64(32u))|u64(0xcf692694u)),
 ((u64(0xe49b69c1u)<<u64(32u))|u64(0x9ef14ad2u)), ((u64(0xefbe4786u)<<u64(32u))|u64(0x384f25e3u)), ((u64(0x0fc19dc6u)<<u64(32u))|u64(0x8b8cd5b5u)), ((u64(0x240ca1ccu)<<u64(32u))|u64(0x77ac9c65u)),
 ((u64(0x2de92c6fu)<<u64(32u))|u64(0x592b0275u)), ((u64(0x4a7484aau)<<u64(32u))|u64(0x6ea6e483u)), ((u64(0x5cb0a9dcu)<<u64(32u))|u64(0xbd41fbd4u)), ((u64(0x76f988dau)<<u64(32u))|u64(0x831153b5u)),
 ((u64(0x983e5152u)<<u64(32u))|u64(0xee66dfabu)), ((u64(0xa831c66du)<<u64(32u))|u64(0x2db43210u)), ((u64(0xb00327c8u)<<u64(32u))|u64(0x98fb213fu)), ((u64(0xbf597fc7u)<<u64(32u))|u64(0xbeef0ee4u)),
 ((u64(0xc6e00bf3u)<<u64(32u))|u64(0x3da88fc2u)), ((u64(0xd5a79147u)<<u64(32u))|u64(0x930aa725u)), ((u64(0x06ca6351u)<<u64(32u))|u64(0xe003826fu)), ((u64(0x14292967u)<<u64(32u))|u64(0x0a0e6e70u)),
 ((u64(0x27b70a85u)<<u64(32u))|u64(0x46d22ffcu)), ((u64(0x2e1b2138u)<<u64(32u))|u64(0x5c26c926u)), ((u64(0x4d2c6dfcu)<<u64(32u))|u64(0x5ac42aedu)), ((u64(0x53380d13u)<<u64(32u))|u64(0x9d95b3dfu)),
 ((u64(0x650a7354u)<<u64(32u))|u64(0x8baf63deu)), ((u64(0x766a0abbu)<<u64(32u))|u64(0x3c77b2a8u)), ((u64(0x81c2c92eu)<<u64(32u))|u64(0x47edaee6u)), ((u64(0x92722c85u)<<u64(32u))|u64(0x1482353bu)),
 ((u64(0xa2bfe8a1u)<<u64(32u))|u64(0x4cf10364u)), ((u64(0xa81a664bu)<<u64(32u))|u64(0xbc423001u)), ((u64(0xc24b8b70u)<<u64(32u))|u64(0xd0f89791u)), ((u64(0xc76c51a3u)<<u64(32u))|u64(0x0654be30u)),
 ((u64(0xd192e819u)<<u64(32u))|u64(0xd6ef5218u)), ((u64(0xd6990624u)<<u64(32u))|u64(0x5565a910u)), ((u64(0xf40e3585u)<<u64(32u))|u64(0x5771202au)), ((u64(0x106aa070u)<<u64(32u))|u64(0x32bbd1b8u)),
 ((u64(0x19a4c116u)<<u64(32u))|u64(0xb8d2d0c8u)), ((u64(0x1e376c08u)<<u64(32u))|u64(0x5141ab53u)), ((u64(0x2748774cu)<<u64(32u))|u64(0xdf8eeb99u)), ((u64(0x34b0bcb5u)<<u64(32u))|u64(0xe19b48a8u)),
 ((u64(0x391c0cb3u)<<u64(32u))|u64(0xc5c95a63u)), ((u64(0x4ed8aa4au)<<u64(32u))|u64(0xe3418acbu)), ((u64(0x5b9cca4fu)<<u64(32u))|u64(0x7763e373u)), ((u64(0x682e6ff3u)<<u64(32u))|u64(0xd6b2b8a3u)),
 ((u64(0x748f82eeu)<<u64(32u))|u64(0x5defb2fcu)), ((u64(0x78a5636fu)<<u64(32u))|u64(0x43172f60u)), ((u64(0x84c87814u)<<u64(32u))|u64(0xa1f0ab72u)), ((u64(0x8cc70208u)<<u64(32u))|u64(0x1a6439ecu)),
 ((u64(0x90befffau)<<u64(32u))|u64(0x23631e28u)), ((u64(0xa4506cebu)<<u64(32u))|u64(0xde82bde9u)), ((u64(0xbef9a3f7u)<<u64(32u))|u64(0xb2c67915u)), ((u64(0xc67178f2u)<<u64(32u))|u64(0xe372532bu)),
 ((u64(0xca273eceu)<<u64(32u))|u64(0xea26619cu)), ((u64(0xd186b8c7u)<<u64(32u))|u64(0x21c0c207u)), ((u64(0xeada7dd6u)<<u64(32u))|u64(0xcde0eb1eu)), ((u64(0xf57d4f7fu)<<u64(32u))|u64(0xee6ed178u)),
 ((u64(0x06f067aau)<<u64(32u))|u64(0x72176fbau)), ((u64(0x0a637dc5u)<<u64(32u))|u64(0xa2c898a6u)), ((u64(0x113f9804u)<<u64(32u))|u64(0xbef90daeu)), ((u64(0x1b710b35u)<<u64(32u))|u64(0x131c471bu)),
 ((u64(0x28db77f5u)<<u64(32u))|u64(0x23047d84u)), ((u64(0x32caab7bu)<<u64(32u))|u64(0x40c72493u)), ((u64(0x3c9ebe0au)<<u64(32u))|u64(0x15c9bebcu)), ((u64(0x431d67c4u)<<u64(32u))|u64(0x9c100d4cu)),
 ((u64(0x4cc5d4beu)<<u64(32u))|u64(0xcb3e42b6u)), ((u64(0x597f299cu)<<u64(32u))|u64(0xfc657e2au)), ((u64(0x5fcb6fabu)<<u64(32u))|u64(0x3ad6faecu)), ((u64(0x6c44198cu)<<u64(32u))|u64(0x4a475817u))
);
fn sha512_ror(x:u64,n:u32)->u64{
 return (x>>u64(n))|(x<<u64(64u-n));
}
fn sha512_compress(state:array<u64,8>,block:array<u64,16>)->array<u64,8>{
 var w:array<u64,80>;
 for(var i=0u;i<16u;i=i+1u){w[i]=block[i];}
 for(var i=16u;i<80u;i=i+1u){
  let s0=sha512_ror(w[i-15u],1u)^sha512_ror(w[i-15u],8u)^(w[i-15u]>>u64(7u));
  let s1=sha512_ror(w[i-2u],19u)^sha512_ror(w[i-2u],61u)^(w[i-2u]>>u64(6u));
  w[i]=w[i-16u]+s0+w[i-7u]+s1;
 }
 var a=state[0];var b=state[1];var c=state[2];var d=state[3];
 var e=state[4];var f=state[5];var g=state[6];var h=state[7];
 for(var i=0u;i<80u;i=i+1u){
  let s1=sha512_ror(e,14u)^sha512_ror(e,18u)^sha512_ror(e,41u);
  let ch=(e&f)^(~e&g);
  let t1=h+s1+ch+SHA512_K[i]+w[i];
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
