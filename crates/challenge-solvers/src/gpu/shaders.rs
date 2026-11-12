//! Protocol WGSL shader construction. Shader bodies are intentionally unchanged.
use super::*;

fn sha256_core_shader_source() -> String {
    let mut rounds = String::new();
    for (t, k) in crate::sha256::K32.iter().enumerate() {
        let i = t & 15;
        if t >= 16 {
            append_fmt(
                &mut rounds,
                format_args!(
                    "w{i}=(rotr(w{},17u)^rotr(w{},19u)^(w{}>>10u))+w{}+(rotr(w{},7u)^rotr(w{},18u)^(w{}>>3u))+w{i};\n",
                    (t - 2) & 15,
                    (t - 2) & 15,
                    (t - 2) & 15,
                    (t - 7) & 15,
                    (t - 15) & 15,
                    (t - 15) & 15,
                    (t - 15) & 15
                ),
            );
        }
        append_fmt(
            &mut rounds,
            format_args!(
                "{{let t1=h+(rotr(e,6u)^rotr(e,11u)^rotr(e,25u))+(g^(e&(f^g)))+0x{k:08x}u+w{i};let t2=(rotr(a,2u)^rotr(a,13u)^rotr(a,22u))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}}\n"
            ),
        );
    }
    include_str!("shaders/sha256_core.wgsl").replace("__SHA256_ROUNDS__", &rounds)
}

pub fn altcha_shader_source() -> String {
    include_str!("shaders/altcha.wgsl")
        .replace("__SHA256_CORE__", &sha256_core_shader_source())
        .replace("__WORKGROUP_SIZE__", &WG_SIZE.to_string())
}

pub fn cerberus_shader_source() -> String {
    const PERM: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
    let mut schedule = [[0_usize; 16]; 7];
    schedule[0] = core::array::from_fn(|i| i);
    for round in 1..7 {
        schedule[round] = core::array::from_fn(|i| schedule[round - 1][PERM[i]]);
    }
    let mut rounds = String::new();
    let gs = [
        (0, 4, 8, 12, 0, 1),
        (1, 5, 9, 13, 2, 3),
        (2, 6, 10, 14, 4, 5),
        (3, 7, 11, 15, 6, 7),
        (0, 5, 10, 15, 8, 9),
        (1, 6, 11, 12, 10, 11),
        (2, 7, 8, 13, 12, 13),
        (3, 4, 9, 14, 14, 15),
    ];
    for sched in &schedule {
        for &(a, b, c, d, x, y) in &gs {
            let mx = sched[x];
            let my = sched[y];
            append_fmt(
                &mut rounds,
                format_args!(
                    "v[{a}]=v[{a}]+v[{b}]+m[{mx}];v[{d}]=rotr(v[{d}]^v[{a}],16u);v[{c}]=v[{c}]+v[{d}];v[{b}]=rotr(v[{b}]^v[{c}],12u);v[{a}]=v[{a}]+v[{b}]+m[{my}];v[{d}]=rotr(v[{d}]^v[{a}],8u);v[{c}]=v[{c}]+v[{d}];v[{b}]=rotr(v[{b}]^v[{c}],7u);\n"
                ),
            );
        }
    }
    format!(
        r"
struct OutBuf {{ flag:atomic<u32>,nonce_lo:u32,nonce_hi:u32,hash:array<u32,8> }};
@group(0) @binding(0) var<storage,read> P:array<u32>;
@group(0) @binding(1) var<storage,read_write> R:OutBuf;
fn rotr(x:u32,n:u32)->u32{{return (x>>n)|(x<<(32u-n));}}
fn compress(cv:array<u32,8>,m:array<u32,16>,blen:u32,flags:u32)->array<u32,8>{{
 var v=array<u32,16>(cv[0],cv[1],cv[2],cv[3],cv[4],cv[5],cv[6],cv[7],0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0u,0u,blen,flags);
 {rounds}
 return array<u32,8>(v[0]^v[8],v[1]^v[9],v[2]^v[10],v[3]^v[11],v[4]^v[12],v[5]^v[13],v[6]^v[14],v[7]^v[15]);
}}
@compute @workgroup_size({WG_SIZE})
fn solve(@builtin(global_invocation_id) gid:vec3<u32>){{
 let first=gid.x*{STEPS}u;
 var decimal_m=array<u32,16>(P[16],P[17],P[18],P[19],P[20],P[21],P[22],P[23],P[24],P[25],P[26],P[27],P[28],P[29],P[30],P[31]);
 if(P[0]==0u&&first<P[2]){{
  var x=P[1]+first;for(var j=0u;j<9u;j=j+1u){{let digit=x%10u;x=x/10u;let pos=P[32]+8u-j;let wi=pos>>2u;let sh=(pos&3u)*8u;decimal_m[wi]=(decimal_m[wi]&~(0xffu<<sh))|((0x30u+digit)<<sh);}}
 }}
 for(var step=0u;step<{STEPS}u;step=step+1u){{
  if((step&{}u)==0u&&atomicLoad(&R.flag)!=0u){{return;}}
  let idx=first+step;if(idx>=P[2]){{return;}}let candidate=P[1]+idx;
  let cv=array<u32,8>(P[8],P[9],P[10],P[11],P[12],P[13],P[14],P[15]);
  var m:array<u32,16>;var nlo=candidate;var nhi=0u;var h:array<u32,8>;
  if(P[0]==0u){{
   m=decimal_m;h=compress(cv,m,P[33],P[34]);nlo=P[35]+candidate;let carry=select(0u,1u,nlo<P[35]);nhi=P[36]+carry;
   if(step+1u<{STEPS}u&&idx+1u<P[2]){{var j=0u;loop{{let pos=P[32]+8u-j;let wi=pos>>2u;let sh=(pos&3u)*8u;let digit=(decimal_m[wi]>>sh)&0xffu;if(digit<0x39u){{decimal_m[wi]=(decimal_m[wi]&~(0xffu<<sh))|((digit+1u)<<sh);break;}}decimal_m[wi]=(decimal_m[wi]&~(0xffu<<sh))|(0x30u<<sh);j=j+1u;if(j==9u){{break;}}}}}}
  }}else{{
   m=array<u32,16>(P[16],candidate,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);h=compress(cv,m,8u,10u);nhi=P[16];
  }}
  if((h[0]&P[3])==0u&&(h[1]&P[4])==0u){{if(atomicExchange(&R.flag,1u)==0u){{R.nonce_lo=nlo;R.nonce_hi=nhi;for(var i=0u;i<8u;i=i+1u){{R.hash[i]=h[i];}}}}}}
 }}
}}
",
        CANCEL_INTERVAL - 1
    )
}

fn multi_layout_rounds(start: usize) -> String {
    let mut rounds = String::new();
    for (t, k) in crate::sha256::K32.iter().enumerate().skip(start) {
        let i = t & 15;
        if t >= 16 {
            append_fmt(
                &mut rounds,
                format_args!(
                    "w{i}=(rotr(w{},17u)^rotr(w{},19u)^(w{}>>10u))+w{}+(rotr(w{},7u)^rotr(w{},18u)^(w{}>>3u))+w{i};\n",
                    (t - 2) & 15,
                    (t - 2) & 15,
                    (t - 2) & 15,
                    (t - 7) & 15,
                    (t - 15) & 15,
                    (t - 15) & 15,
                    (t - 15) & 15
                ),
            );
        }
        append_fmt(
            &mut rounds,
            format_args!(
                "{{let t1=h+(rotr(e,6u)^rotr(e,11u)^rotr(e,25u))+(g^(e&(f^g)))+0x{k:08x}u+w{i};let t2=(rotr(a,2u)^rotr(a,13u)^rotr(a,22u))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}}\n"
            ),
        );
    }
    rounds
}

pub fn multi_layout_shader_source() -> String {
    let rounds = multi_layout_rounds(0);
    let rounds9 = multi_layout_rounds(9);
    let rounds13 = multi_layout_rounds(13);
    format!(
        r"
const IV:array<u32,8>=array<u32,8>(0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u);
override MODE:u32;
struct OutBuf {{ flag:atomic<u32>,nonce_lo:u32,nonce_hi:u32,hash:array<u32,8> }};
@group(0) @binding(0) var<storage,read> P:array<u32>;
@group(0) @binding(1) var<storage,read_write> R:OutBuf;
fn rotr(x:u32,n:u32)->u32{{return (x>>n)|(x<<(32u-n));}}
fn less64(ah:u32,al:u32,bh:u32,bl:u32)->bool{{return ah<bh||(ah==bh&&al<bl);}}
fn greater64(ah:u32,al:u32,bh:u32,bl:u32)->bool{{return ah>bh||(ah==bh&&al>bl);}}
fn target_ok(h:array<u32,8>)->bool{{
 let ty=P[1];
 if(ty==8u){{
  let bits=P[7];if(P[6]!=0u||bits>256u){{return false;}}let full_words=bits>>5u;
  for(var i=0u;i<8u;i=i+1u){{if(i<full_words&&h[i]!=0u){{return false;}}}}
  let remaining=bits&31u;if(remaining==0u){{return true;}}
  return (h[full_words]>>(32u-remaining))==0u;
 }}
 let full=(P[2]&1u)!=0u;
 if(ty==1u){{return select(h[0]<P[6],less64(h[0],h[1],P[6],P[7]),full);}}
 if(ty==2u){{return select(h[0]>P[6],greater64(h[0],h[1],P[6],P[7]),full);}}
 let hi=(h[0]&P[8])==(P[6]&P[8]);return hi&&select(true,(h[1]&P[9])==(P[7]&P[9]),full);
}}
fn compress(s:array<u32,8>,block:array<u32,16>)->array<u32,8>{{
 var w0=block[0];var w1=block[1];var w2=block[2];var w3=block[3];var w4=block[4];var w5=block[5];var w6=block[6];var w7=block[7];
 var w8=block[8];var w9=block[9];var w10=block[10];var w11=block[11];var w12=block[12];var w13=block[13];var w14=block[14];var w15=block[15];
 var a=s[0];var b=s[1];var c=s[2];var d=s[3];var e=s[4];var f=s[5];var g=s[6];var h=s[7];
 {rounds}
 return array<u32,8>(s[0]+a,s[1]+b,s[2]+c,s[3]+d,s[4]+e,s[5]+f,s[6]+g,s[7]+h);
}}
fn compress9(feed:array<u32,8>,s:array<u32,8>,block:array<u32,16>)->array<u32,8>{{
 var w0=block[0];var w1=block[1];var w2=block[2];var w3=block[3];var w4=block[4];var w5=block[5];var w6=block[6];var w7=block[7];
 var w8=block[8];var w9=block[9];var w10=block[10];var w11=block[11];var w12=block[12];var w13=block[13];var w14=block[14];var w15=block[15];
 var a=s[0];var b=s[1];var c=s[2];var d=s[3];var e=s[4];var f=s[5];var g=s[6];var h=s[7];
 {rounds9}
 return array<u32,8>(feed[0]+a,feed[1]+b,feed[2]+c,feed[3]+d,feed[4]+e,feed[5]+f,feed[6]+g,feed[7]+h);
}}
fn compress13(feed:array<u32,8>,s:array<u32,8>,block:array<u32,16>)->array<u32,8>{{
 var w0=block[0];var w1=block[1];var w2=block[2];var w3=block[3];var w4=block[4];var w5=block[5];var w6=block[6];var w7=block[7];
 var w8=block[8];var w9=block[9];var w10=block[10];var w11=block[11];var w12=block[12];var w13=block[13];var w14=block[14];var w15=block[15];
 var a=s[0];var b=s[1];var c=s[2];var d=s[3];var e=s[4];var f=s[5];var g=s[6];var h=s[7];
 {rounds13}
 return array<u32,8>(feed[0]+a,feed[1]+b,feed[2]+c,feed[3]+d,feed[4]+e,feed[5]+f,feed[6]+g,feed[7]+h);
}}
fn publish(nlo:u32,nhi:u32,h:array<u32,8>){{if(target_ok(h)&&atomicExchange(&R.flag,1u)==0u){{R.nonce_lo=nlo;R.nonce_hi=nhi;for(var i=0u;i<8u;i=i+1u){{R.hash[i]=h[i];}}}}}}
@compute @workgroup_size({WG_SIZE})
fn solve(@builtin(global_invocation_id) gid:vec3<u32>){{
 let first=gid.x*{STEPS}u;
 var decimal_block=array<u32,16>(P[24],P[25],P[26],P[27],P[28],P[29],P[30],P[31],P[32],P[33],P[34],P[35],P[36],P[37],P[38],P[39]);
 var binary_blocks=array<u32,32>(P[24],P[25],P[26],P[27],P[28],P[29],P[30],P[31],P[32],P[33],P[34],P[35],P[36],P[37],P[38],P[39],P[40],P[41],P[42],P[43],P[44],P[45],P[46],P[47],P[48],P[49],P[50],P[51],P[52],P[53],P[54],P[55]);
 if(MODE==1u&&first<P[5]){{
  var x=P[3]+first;
  for(var j=0u;j<9u;j=j+1u){{let digit=x%10u;x=x/10u;let pos=62u-j;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;decimal_block[wi]=(decimal_block[wi]&~(0xffu<<sh))|((0x30u+digit)<<sh);}}
 }}
 if(MODE==3u&&first<P[5]){{
  let blo=P[3]+first;let bc=select(0u,1u,blo<P[3]);let bhi=P[4]+bc;
  for(var j=0u;j<P[11];j=j+1u){{let byte=select((blo>>(j*8u))&0xffu,(bhi>>((j-4u)*8u))&0xffu,j>=4u);let pos=P[12]+j;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;binary_blocks[wi]=(binary_blocks[wi]&~(0xffu<<sh))|(byte<<sh);}}
 }}
 for(var step=0u;step<{STEPS}u;step=step+1u){{
  if((step&{}u)==0u&&atomicLoad(&R.flag)!=0u){{return;}}
  let idx=first+step;if(idx>=P[5]){{return;}}
  let clo=P[3]+idx;let carry=select(0u,1u,clo<P[3]);let chi=P[4]+carry;let mode=MODE;
  if(mode==1u){{
   let st=array<u32,8>(P[16],P[17],P[18],P[19],P[20],P[21],P[22],P[23]);let hs=array<u32,8>(P[44],P[45],P[46],P[47],P[48],P[49],P[50],P[51]);let mid=compress13(st,hs,decimal_block);
   var tail=array<u32,16>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,P[42],P[43]);let h=compress(mid,tail);
   let nlo=P[40]+clo;let nc=select(0u,1u,nlo<P[40]);publish(nlo,P[41]+nc,h);
   if(step+1u<{STEPS}u&&idx+1u<P[5]){{
    var j=0u;loop{{let pos=62u-j;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;let digit=(decimal_block[wi]>>sh)&0xffu;if(digit<0x39u){{decimal_block[wi]=(decimal_block[wi]&~(0xffu<<sh))|((digit+1u)<<sh);break;}}decimal_block[wi]=(decimal_block[wi]&~(0xffu<<sh))|(0x30u<<sh);j=j+1u;if(j==9u){{break;}}}}
   }}
  }} else if(mode==2u){{
   let block=array<u32,16>(P[16],P[17],P[18],P[19],P[20],P[21],P[22],P[23],P[24],clo,0x80000000u,0u,0u,0u,0u,320u);
   let hs=array<u32,8>(P[40],P[41],P[42],P[43],P[44],P[45],P[46],P[47]);publish(clo,P[24],compress9(IV,hs,block));
  }} else if(mode==3u){{
   let b0=array<u32,16>(binary_blocks[0],binary_blocks[1],binary_blocks[2],binary_blocks[3],binary_blocks[4],binary_blocks[5],binary_blocks[6],binary_blocks[7],binary_blocks[8],binary_blocks[9],binary_blocks[10],binary_blocks[11],binary_blocks[12],binary_blocks[13],binary_blocks[14],binary_blocks[15]);
   let st=array<u32,8>(P[16],P[17],P[18],P[19],P[20],P[21],P[22],P[23]);var h=compress(st,b0);
   if(P[10]==2u){{let b1=array<u32,16>(binary_blocks[16],binary_blocks[17],binary_blocks[18],binary_blocks[19],binary_blocks[20],binary_blocks[21],binary_blocks[22],binary_blocks[23],binary_blocks[24],binary_blocks[25],binary_blocks[26],binary_blocks[27],binary_blocks[28],binary_blocks[29],binary_blocks[30],binary_blocks[31]);h=compress(h,b1);}}
   publish(clo,chi,h);
   if(step+1u<{STEPS}u&&idx+1u<P[5]){{var j=0u;loop{{let pos=P[12]+j;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;let byte=(binary_blocks[wi]>>sh)&0xffu;if(byte<0xffu){{binary_blocks[wi]=(binary_blocks[wi]&~(0xffu<<sh))|((byte+1u)<<sh);break;}}binary_blocks[wi]=binary_blocks[wi]&~(0xffu<<sh);j=j+1u;if(j==P[11]){{break;}}}}}}
  }}
 }}
}}
",
        CANCEL_INTERVAL - 1
    )
}

pub fn generic_single_block_shader_source() -> String {
    let mut rounds = String::new();
    for (t, k) in crate::sha256::K32.iter().enumerate() {
        let i = t & 15;
        if t >= 16 {
            append_fmt(
                &mut rounds,
                format_args!(
                    "w{i}=(rotr(w{},17u)^rotr(w{},19u)^(w{}>>10u))+w{}+(rotr(w{},7u)^rotr(w{},18u)^(w{}>>3u))+w{i};\n",
                    (t - 2) & 15,
                    (t - 2) & 15,
                    (t - 2) & 15,
                    (t - 7) & 15,
                    (t - 15) & 15,
                    (t - 15) & 15,
                    (t - 15) & 15
                ),
            );
        }
        append_fmt(
            &mut rounds,
            format_args!(
                "{{let t1=h+(rotr(e,6u)^rotr(e,11u)^rotr(e,25u))+(g^(e&(f^g)))+0x{k:08x}u+w{i};let t2=(rotr(a,2u)^rotr(a,13u)^rotr(a,22u))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}}\n"
            ),
        );
    }
    format!(
        r"
struct OutBuf {{ flag:atomic<u32>,nonce_lo:u32,nonce_hi:u32,hash:array<u32,8> }};
@group(0) @binding(0) var<storage,read> P:array<u32>;
@group(0) @binding(1) var<storage,read_write> R:OutBuf;
fn rotr(x:u32,n:u32)->u32{{return (x>>n)|(x<<(32u-n));}}
fn less64(ah:u32,al:u32,bh:u32,bl:u32)->bool{{return ah<bh||(ah==bh&&al<bl);}}
fn greater64(ah:u32,al:u32,bh:u32,bl:u32)->bool{{return ah>bh||(ah==bh&&al>bl);}}
fn target_ok(h:array<u32,8>)->bool{{
 let ty=P[32];
 if(ty==8u){{
  let bits=P[29];if(P[28]!=0u||bits>256u){{return false;}}let full_words=bits>>5u;
  for(var i=0u;i<8u;i=i+1u){{if(i<full_words&&h[i]!=0u){{return false;}}}}
  let remaining=bits&31u;if(remaining==0u){{return true;}}
  return (h[full_words]>>(32u-remaining))==0u;
 }}
 let full=(P[33]&2u)!=0u;
 if(ty==1u){{return select(h[0]<P[28],less64(h[0],h[1],P[28],P[29]),full);}}
 if(ty==2u){{return select(h[0]>P[28],greater64(h[0],h[1],P[28],P[29]),full);}}
 let hi=(h[0]&P[30])==(P[28]&P[30]);
 return hi&&select(true,(h[1]&P[31])==(P[29]&P[31]),full);
}}
@compute @workgroup_size({WG_SIZE})
fn solve(@builtin(global_invocation_id) gid:vec3<u32>){{
 let first=P[24]+gid.x*{STEPS}u;if(first>=P[34]){{return;}}
 var words=array<u32,16>(P[8],P[9],P[10],P[11],P[12],P[13],P[14],P[15],P[16],P[17],P[18],P[19],P[20],P[21],P[22],P[23]);
 // Format once per invocation. The remaining candidates are generated by an in-place ASCII decimal odometer.
 var x=first;
 for(var j=0u;j<9u;j=j+1u){{
  let digit=x%10u;x=x/10u;let pos=P[25]+8u-j;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;
  words[wi]=(words[wi]&~(0xffu<<sh))|((0x30u+digit)<<sh);
 }}
 let last_pos=P[25]+8u;let last_wi=last_pos>>2u;let last_sh=(3u-(last_pos&3u))*8u;
 for(var step=0u;step<{STEPS}u;step=step+1u){{
  if((step&{}u)==0u&&atomicLoad(&R.flag)!=0u){{return;}}
  let candidate=first+step;if(candidate>=P[34]){{return;}}
  let trailing_zero=((words[last_wi]>>last_sh)&0xffu)==0x30u;
  if(!((P[33]&1u)!=0u&&trailing_zero)){{
   var w0=words[0];var w1=words[1];var w2=words[2];var w3=words[3];var w4=words[4];var w5=words[5];var w6=words[6];var w7=words[7];
   var w8=words[8];var w9=words[9];var w10=words[10];var w11=words[11];var w12=words[12];var w13=words[13];var w14=words[14];var w15=words[15];
   var a=P[0];var b=P[1];var c=P[2];var d=P[3];var e=P[4];var f=P[5];var g=P[6];var h=P[7];
   {rounds}
   let out=array<u32,8>(P[0]+a,P[1]+b,P[2]+c,P[3]+d,P[4]+e,P[5]+f,P[6]+g,P[7]+h);
   if(target_ok(out)){{if(atomicExchange(&R.flag,1u)==0u){{
    let nlo=P[26]+candidate;let carry=select(0u,1u,nlo<P[26]);R.nonce_lo=nlo;R.nonce_hi=P[27]+carry;
    for(var i=0u;i<8u;i=i+1u){{R.hash[i]=out[i];}}
   }}}}
  }}
  if(step+1u<{STEPS}u&&candidate+1u<P[34]){{
   var j=0u;
   loop{{
    let pos=P[25]+8u-j;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;let digit=(words[wi]>>sh)&0xffu;
    if(digit<0x39u){{words[wi]=(words[wi]&~(0xffu<<sh))|((digit+1u)<<sh);break;}}
    words[wi]=(words[wi]&~(0xffu<<sh))|(0x30u<<sh);j=j+1u;if(j==9u){{break;}}
   }}
  }}
 }}
}}
",
        CANCEL_INTERVAL - 1
    )
}

pub fn shader_source() -> String {
    let mut rounds = String::new();
    for (t, k) in crate::sha256::K32.iter().enumerate() {
        let i = t & 15;
        if t >= 16 {
            append_fmt(
                &mut rounds,
                format_args!(
                    "w{i}=(rotr(w{},17u)^rotr(w{},19u)^(w{}>>10u))+w{}+(rotr(w{},7u)^rotr(w{},18u)^(w{}>>3u))+w{i};\n",
                    (t - 2) & 15,
                    (t - 2) & 15,
                    (t - 2) & 15,
                    (t - 7) & 15,
                    (t - 15) & 15,
                    (t - 15) & 15,
                    (t - 15) & 15
                ),
            );
        }
        append_fmt(
            &mut rounds,
            format_args!(
                "{{let t1=h+(rotr(e,6u)^rotr(e,11u)^rotr(e,25u))+(g^(e&(f^g)))+0x{k:08x}u+w{i};let t2=(rotr(a,2u)^rotr(a,13u)^rotr(a,22u))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}}\n"
            ),
        );
    }
    format!(
        r"
struct InParams {{ s0:u32,s1:u32,s2:u32,s3:u32,s4:u32,s5:u32,s6:u32,s7:u32,base_lo:u32,base_hi:u32,bitlen:u32,mask0:u32,mask1:u32,batch_size:u32 }};
struct OutBuf {{ flag:atomic<u32>,nonce_lo:u32,nonce_hi:u32,hash:array<u32,8> }};
@group(0) @binding(0) var<uniform> P:InParams;
@group(0) @binding(1) var<storage,read_write> R:OutBuf;
fn rotr(x:u32,n:u32)->u32{{return (x>>n)|(x<<(32u-n));}}
@compute @workgroup_size({WG_SIZE})
fn solve(@builtin(global_invocation_id) gid:vec3<u32>){{
 let local_first=gid.x*{STEPS}u;if(local_first>=P.batch_size){{return;}}
 let gid_lo=gid.x<<7u;let gid_hi=gid.x>>25u;let base_lo=P.base_lo+gid_lo;var base_hi=P.base_hi+gid_hi;if(base_lo<P.base_lo){{base_hi=base_hi+1u;}}
 let d2=(base_lo>>6u)&7u;let d3=(base_lo>>9u)&7u;let d4=(base_lo>>12u)&7u;let d5=(base_lo>>15u)&7u;let d6=(base_lo>>18u)&7u;let d7=(base_lo>>21u)&7u;let d8=(base_lo>>24u)&7u;let d9=(base_lo>>27u)&7u;let d10=((base_lo>>30u)|(base_hi<<2u))&7u;let d11=(base_hi>>1u)&7u;let d12=(base_hi>>4u)&7u;let d13=(base_hi>>7u)&7u;let d14=(base_hi>>10u)&7u;let d15=(base_hi>>13u)&7u;let d16=(base_hi>>16u)&7u;let d17=(base_hi>>19u)&7u;
 let nw0=0x31303030u|(d17<<16u)|(d16<<8u)|d15;let nw1=0x30303030u|(d14<<24u)|(d13<<16u)|(d12<<8u)|d11;let nw2=0x30303030u|(d10<<24u)|(d9<<16u)|(d8<<8u)|d7;let nw3=0x30303030u|(d6<<24u)|(d5<<16u)|(d4<<8u)|d3;var nw4=0x30303080u|(d2<<24u);
 const GM:array<u32,7>=array<u32,7>(0x00000100u,0x00000200u,0x00000400u,0x00010000u,0x00020000u,0x00040000u,0x01000000u);
 for(var step=0u;step<{STEPS}u;step=step+1u){{
  if(local_first+step>=P.batch_size){{return;}}
  if((step&{}u)==0u && atomicLoad(&R.flag)!=0u){{return;}}
  var w0=nw0;var w1=nw1;var w2=nw2;var w3=nw3;var w4=nw4;
  var w5=0u;var w6=0u;var w7=0u;var w8=0u;var w9=0u;var w10=0u;var w11=0u;var w12=0u;var w13=0u;var w14=0u;var w15=P.bitlen;
  var a=P.s0;var b=P.s1;var c=P.s2;var d=P.s3;var e=P.s4;var f=P.s5;var g=P.s6;var h=P.s7;
  {rounds}
  let r0=P.s0+a;let r1=P.s1+b;
  if((r0&P.mask0)==0u&&(r1&P.mask1)==0u){{if(atomicExchange(&R.flag,1u)==0u){{let gray=step^(step>>1u);let clo=base_lo+gray;let carry=select(0u,1u,clo<base_lo);R.nonce_lo=clo;R.nonce_hi=base_hi+carry;R.hash[0]=r0;R.hash[1]=r1;R.hash[2]=P.s2+c;R.hash[3]=P.s3+d;R.hash[4]=P.s4+e;R.hash[5]=P.s5+f;R.hash[6]=P.s6+g;R.hash[7]=P.s7+h;}}}}
  let ns=step+1u;if(ns<{STEPS}u){{nw4=nw4^GM[countTrailingZeros(ns)];}}
 }}
}}
",
        CANCEL_INTERVAL - 1
    )
}
