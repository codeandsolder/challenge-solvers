// ALTCHA SHA-256/PBKDF2 WGPU kernel; populated with unrolled SHA rounds and WG size.
struct OutBuf { flag:atomic<u32>,nonce_lo:u32,nonce_hi:u32,hash:array<u32,8> };
@group(0) @binding(0) var<storage,read> P:array<u32>;
@group(0) @binding(1) var<storage,read_write> R:OutBuf;
__SHA256_CORE__
fn target_ok(out:array<u32,8>)->bool{
 let ty=P[3];
 if(ty==4u&&P[12]!=0u){
  // Compare each hex nibble, including odd lengths and all 32 digest bytes.
  for(var i=0u;i<min(P[12],64u);i=i+1u){
   let actual=(out[i/8u]>>((7u-i%8u)*4u))&15u;
   let expected=(P[32u+i/8u]>>((7u-i%8u)*4u))&15u;
   if(actual!=expected){return false;}
  }
  return true;
 }
 let h0=out[0];let h1=out[1];
 if(ty==1u){return h0<P[4]||(h0==P[4]&&h1<P[5]);}
 if(ty==2u){return h0>P[4]||(h0==P[4]&&h1>P[5]);}
 return (h0&P[6])==(P[4]&P[6])&&(h1&P[7])==(P[5]&P[7]);
}
fn hash_prefix(h:array<u32,8>,len:u32)->array<u32,8>{
 var b:array<u32,16>;for(var i=0u;i<16u;i=i+1u){b[i]=0u;}
 for(var j=0u;j<len;j=j+1u){let wi=j>>2u;let sh=(3u-(j&3u))*8u;let byte=(h[wi]>>sh)&0xffu;b[wi]=b[wi]|(byte<<sh);}
 let wi=len>>2u;let sh=(3u-(len&3u))*8u;b[wi]=b[wi]|(0x80u<<sh);b[15]=len*8u;return compress(IV,b);
}
fn hmac(ipad:array<u32,8>,opad:array<u32,8>,msg:array<u32,8>,words:u32,tail_word:u32,bytes:u32)->array<u32,8>{
 var b:array<u32,16>;for(var i=0u;i<16u;i=i+1u){b[i]=0u;}for(var i=0u;i<words;i=i+1u){b[i]=msg[i];}if(words<8u){b[words]=tail_word;}let padpos=bytes;let pwi=padpos>>2u;let psh=(3u-(padpos&3u))*8u;b[pwi]=b[pwi]|(0x80u<<psh);b[15]=(64u+bytes)*8u;let inner=compress(ipad,b);
 var o:array<u32,16>;for(var i=0u;i<16u;i=i+1u){o[i]=0u;}for(var i=0u;i<8u;i=i+1u){o[i]=inner[i];}o[8]=0x80000000u;o[15]=768u;return compress(opad,o);
}
// Finish SHA-256(salt || nonce || counter) with an arbitrary-length fixed
// prefix. P[40..47] is the SHA chaining state after its full 64-byte blocks;
// P[48..63] holds the remaining 0..63 prefix bytes. The binary or decimal
// counter adds <=16 bytes, so SHA padding requires at most two more blocks.
fn sha256_suffix(candidate:u32,suffix:array<u32,16>,suffix_len:u32)->array<u32,8>{
 var first:array<u32,16>;
 var second:array<u32,16>;
 for(var i=0u;i<16u;i=i+1u){first[i]=0u;second[i]=0u;}
 let remainder=P[15];
 for(var i=0u;i<remainder;i=i+1u){
  let byte=(P[48u+i/4u]>>((3u-i%4u)*8u))&255u;
  first[i/4u]=first[i/4u]|(byte<<((3u-i%4u)*8u));
 }
 for(var j=0u;j<suffix_len;j=j+1u){
  var byte:u32;
  if(P[11]!=0u){
   byte=suffix[suffix_len-1u-j];
  }else{
   byte=(candidate>>((3u-j)*8u))&255u;
  }
  let pos=remainder+j;
  if(pos<64u){
   first[pos/4u]=first[pos/4u]|(byte<<((3u-pos%4u)*8u));
  }else{
   let offset=pos-64u;
   second[offset/4u]=second[offset/4u]|(byte<<((3u-offset%4u)*8u));
  }
 }
 let final_pos=remainder+suffix_len;
 if(final_pos<64u){
  first[final_pos/4u]=first[final_pos/4u]|(0x80u<<((3u-final_pos%4u)*8u));
 }else{
  let offset=final_pos-64u;
  second[offset/4u]=second[offset/4u]|(0x80u<<((3u-offset%4u)*8u));
 }
 // Total bit length, preserving high words across 32-bit overflow.
 let byte_lo=P[13]+suffix_len;
 let byte_hi=P[14]+select(0u,1u,byte_lo<P[13]);
 let bit_lo=byte_lo<<3u;
 let bit_hi=(byte_hi<<3u)|(byte_lo>>29u);
 if(final_pos<56u){first[14]=bit_hi;first[15]=bit_lo;}
 else{second[14]=bit_hi;second[15]=bit_lo;}
 let state=array<u32,8>(P[40],P[41],P[42],P[43],P[44],P[45],P[46],P[47]);
 let intermediate=compress(state,first);
 if(final_pos<56u){return intermediate;}
 return compress(intermediate,second);
}
// PBKDF2-HMAC-SHA256 with an arbitrary-length nonce||counter password and
// arbitrary binary salt. Passwords >64 bytes become SHA256(password) as
// required by HMAC, using the same nonce prefix-midstate and exact 64-bit
// final length encoding as the general SHA path.
//
// P[64..] contains fully padded U1 inner message blocks for salt || INT(1).
// They are candidate-independent, but each candidate must compress them
// from its own ipad-derived chaining state.
fn pbkdf2_general(candidate:u32,suffix:array<u32,16>,suffix_len:u32,block_index:u32)->array<u32,8>{
 var key:array<u32,16>;
 for(var i=0u;i<16u;i=i+1u){key[i]=0u;}
 let nonce_len=P[13];
 if(P[14]==0u&&nonce_len<=64u-suffix_len){
  // Raw nonce||counter is itself a valid HMAC key (<=64 bytes).
  // All nonce bytes are present in P[48..63] for this branch.
  for(var i=0u;i<nonce_len;i=i+1u){
   let byte=(P[48u+i/4u]>>((3u-i%4u)*8u))&255u;
   key[i/4u]=key[i/4u]|(byte<<((3u-i%4u)*8u));
  }
  for(var j=0u;j<suffix_len;j=j+1u){
   var byte:u32;
   if(P[11]!=0u){byte=suffix[suffix_len-1u-j];}
   else{byte=(candidate>>((3u-j)*8u))&255u;}
   let pos=nonce_len+j;
   key[pos/4u]=key[pos/4u]|(byte<<((3u-pos%4u)*8u));
  }
 }else{
  // HMAC requires a long candidate password to be SHA256 hashed first.
  let hashed=sha256_suffix(candidate,suffix,suffix_len);
  for(var i=0u;i<8u;i=i+1u){key[i]=hashed[i];}
 }
 var ib:array<u32,16>;
 var ob:array<u32,16>;
 for(var i=0u;i<16u;i=i+1u){
  ib[i]=key[i]^0x36363636u;
  ob[i]=key[i]^0x5c5c5c5cu;
 }
 let ipad=compress(IV,ib);
 let opad=compress(IV,ob);
 var inner=ipad;
 for(var block=0u;block<P[25];block=block+1u){
  var w:array<u32,16>;
  for(var i=0u;i<16u;i=i+1u){
   w[i]=P[64u+block*16u+i];
   if(block_index!=1u){
    // The 4-byte PBKDF2 INT(block_index) follows the arbitrary salt.
    // It may cross word and compression block boundaries.
    let word_pos=block*64u+i*4u;
    if(word_pos<P[26]+4u&&word_pos+4u>P[26]){
     for(var j=0u;j<4u;j=j+1u){
      let pos=word_pos+j;
      if(pos>=P[26]&&pos<P[26]+4u){
       let shift=(3u-j)*8u;
       let index_shift=(3u-(pos-P[26]))*8u;
       let byte=(block_index>>index_shift)&255u;
       w[i]=(w[i]&~(255u<<shift))|(byte<<shift);
      }
     }
    }
   }
  }
  inner=compress(inner,w);
 }
 var outer:array<u32,16>;
 for(var i=0u;i<16u;i=i+1u){outer[i]=0u;}
 for(var i=0u;i<8u;i=i+1u){outer[i]=inner[i];}
 outer[8]=0x80000000u;outer[15]=768u;
 var u=compress(opad,outer);
 var acc=u;
 for(var iteration=1u;iteration<P[8];iteration=iteration+1u){
  u=hmac(ipad,opad,u,8u,0u,32u);
  for(var i=0u;i<8u;i=i+1u){acc[i]=acc[i]^u[i];}
 }
 return acc;
}
@compute @workgroup_size(__WORKGROUP_SIZE__)
fn solve(@builtin(global_invocation_id) gid:vec3<u32>){
 let idx=gid.x;if(idx>=P[2]||atomicLoad(&R.flag)!=0u){return;}
 let candidate=P[1]+idx;
 let carry=select(0u,1u,candidate<P[1]);
 let candidate_hi=P[10]+carry;
 var suffix:array<u32,16>;
 var suffix_len=0u;
 if(P[11]!=0u){
  // Divide a JavaScript-safe 53-bit integer by ten without WGSL u64.
  // The intermediate 10*65536 + 65535 remains representable in u32.
  var hi=candidate_hi;var lo=candidate;
  loop{
   let qhi=hi/10u;
   let top=(hi%10u)*65536u+(lo>>16u);
   let middle=top/10u;
   let bottom=(top%10u)*65536u+(lo&0xffffu);
   suffix[suffix_len]=0x30u+bottom%10u;
   suffix_len=suffix_len+1u;
   hi=qhi;lo=(middle<<16u)|(bottom/10u);
   if((hi|lo)==0u){break;}
  }
 }else{
  suffix_len=4u;
 }
 var out:array<u32,8>;
 if(P[0]==0u){
  if(P[24]!=0u){
   out=sha256_suffix(candidate,suffix,suffix_len);
  }else{
  var b=array<u32,16>(P[16],P[17],P[18],P[19],P[20],P[21],P[22],P[23],0u,0u,0u,0u,0u,0u,0u,0u);
  if(P[11]!=0u){
   for(var j=0u;j<suffix_len;j=j+1u){
    let pos=32u+j;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;
    b[wi]=b[wi]|(suffix[suffix_len-1u-j]<<sh);
   }
   let pos=32u+suffix_len;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;
   b[wi]=b[wi]|(0x80u<<sh);b[15]=pos*8u;
  }else{
   b[8]=candidate;b[9]=0x80000000u;b[15]=288u;
  }
  out=compress(IV,b);
  }
  for(var r=1u;r<P[8];r=r+1u){out=hash_prefix(out,32u);}
 }else if(P[24]==2u){
  out=pbkdf2_general(candidate,suffix,suffix_len,1u);
 }else{
  var ib:array<u32,16>;var ob:array<u32,16>;for(var i=0u;i<16u;i=i+1u){ib[i]=0x36363636u;ob[i]=0x5c5c5c5cu;}
  for(var i=0u;i<4u;i=i+1u){ib[i]=ib[i]^P[20+i];ob[i]=ob[i]^P[20+i];}
  if(P[11]!=0u){
   for(var j=0u;j<suffix_len;j=j+1u){
    let pos=16u+j;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;
    let byte=suffix[suffix_len-1u-j]<<sh;
    ib[wi]=ib[wi]^byte;ob[wi]=ob[wi]^byte;
   }
  }else{ib[4]=ib[4]^candidate;ob[4]=ob[4]^candidate;}
  let ipad=compress(IV,ib);let opad=compress(IV,ob);
  var sm:array<u32,8>;for(var i=0u;i<4u;i=i+1u){sm[i]=P[16+i];}for(var i=4u;i<8u;i=i+1u){sm[i]=0u;}
  var u=hmac(ipad,opad,sm,4u,1u,20u);var acc=u;
  for(var r=1u;r<P[8];r=r+1u){u=hmac(ipad,opad,u,8u,0u,32u);for(var i=0u;i<8u;i=i+1u){acc[i]=acc[i]^u[i];}}out=acc;
 }
 var matches=target_ok(out);
 // PBKDF2 key lengths >32 bytes have multiple SHA-256 output blocks.
 // Reject the first block early for almost every candidate, then evaluate
 // additional blocks only after all preceding exact hex nibbles matched.
 if(matches&&P[0]!=0u&&P[24]==2u&&P[12]>64u){
  var current_block=1u;
  var block_result=out;
  for(var nibble=64u;nibble<P[12];nibble=nibble+1u){
   let block_index=nibble/64u+1u;
   if(block_index!=current_block){
    block_result=pbkdf2_general(candidate,suffix,suffix_len,block_index);
    current_block=block_index;
   }
   let extra=nibble-64u;
   let expected=(P[P[28]+extra/8u]>>((7u-extra%8u)*4u))&15u;
   let position=nibble%64u;
   let actual=(block_result[position/8u]>>((7u-position%8u)*4u))&15u;
   if(actual!=expected){matches=false;break;}
  }
 }
 if(matches&&atomicExchange(&R.flag,1u)==0u){
  R.nonce_lo=candidate;R.nonce_hi=select(0u,candidate_hi,P[11]!=0u);
  for(var i=0u;i<8u;i=i+1u){R.hash[i]=out[i];}
 }
}
