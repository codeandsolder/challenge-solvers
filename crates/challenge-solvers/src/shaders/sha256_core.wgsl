const IV:array<u32,8>=array<u32,8>(0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u);
fn rotr(x:u32,n:u32)->u32{return (x>>n)|(x<<(32u-n));}
fn compress(s:array<u32,8>,block:array<u32,16>)->array<u32,8>{
 var w0=block[0];var w1=block[1];var w2=block[2];var w3=block[3];var w4=block[4];var w5=block[5];var w6=block[6];var w7=block[7];
 var w8=block[8];var w9=block[9];var w10=block[10];var w11=block[11];var w12=block[12];var w13=block[13];var w14=block[14];var w15=block[15];
 var a=s[0];var b=s[1];var c=s[2];var d=s[3];var e=s[4];var f=s[5];var g=s[6];var h=s[7];
 __SHA256_ROUNDS__
 return array<u32,8>(s[0]+a,s[1]+b,s[2]+c,s[3]+d,s[4]+e,s[5]+f,s[6]+g,s[7]+h);
}
