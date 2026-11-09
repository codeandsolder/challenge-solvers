// GPU ALTCHA SHA-384/SHA-512 and PBKDF2-HMAC-SHA-384/SHA-512.
// Includes sha512_core.wgsl before this file. Requires SHADER_INT64.
// Packed BE bytes in P; no fixed salt/nonce length limit beyond GPU storage.
// P: variant,base_low,batch,base_high,cost,data_len,prefix_nibbles,
//    decimal_counter,salt_len,key_len,data_offset,expected_offset,
//    salt_offset,pbkdf_mode,unused,unused,iv_hi_lo[16],variable data...
struct Sha512Result {
 flag: atomic<u32>,
 nonce_low: u32,
 nonce_high: u32,
 spare: u32,
};
@group(0) @binding(0) var<storage, read> P: array<u32>;
@group(0) @binding(1) var<storage, read_write> R: Sha512Result;

fn hash_iv()->array<u64,8> {
 var result:array<u64,8>;
 for(var i=0u;i<8u;i=i+1u){
  result[i]=(u64(P[16u+2u*i])<<32u)|u64(P[17u+2u*i]);
 }
 return result;
}
fn digest_width()->u32 { return select(48u,64u,P[0]!=0u); }
fn data_byte(offset:u32,index:u32)->u32 {
 return (P[offset+index/4u]>>((3u-index%4u)*8u))&255u;
}
fn digest_byte(words:array<u64,8>,index:u32)->u32 {
 return u32((words[index/8u]>>((7u-index%8u)*8u))&u64(255u));
}
fn input_byte(kind:u32,index:u32,previous:array<u64,8>,
              suffix:array<u32,20>,suffix_len:u32,block_index:u32)->u32 {
 if(kind==0u){
  if(index<P[5]){return data_byte(P[10],index);}
  return suffix[suffix_len-1u-(index-P[5])];
 }
 if(kind==1u){return digest_byte(previous,index);}
 // PBKDF2 U1 salt || INT_32_BE(block_index).
 if(index<P[8]){return data_byte(P[12],index);}
 return (block_index>>((3u-(index-P[8]))*8u))&255u;
}
fn hash_source(initial:array<u64,8>,previous_compressed_bytes:u32,
               kind:u32,previous:array<u64,8>,suffix:array<u32,20>,
               suffix_len:u32,block_index:u32)->array<u64,8> {
 var state=initial;
 var len=digest_width();
 if(kind==0u){len=P[5]+suffix_len;}
 if(kind==2u){len=P[8]+4u;}
 let padded_blocks=(len+144u)/128u;
 let total_bits=u64(previous_compressed_bytes+len)<<3u;
 for(var b=0u;b<padded_blocks;b=b+1u){
  var words:array<u64,16>;
  for(var j=0u;j<16u;j=j+1u){words[j]=u64(0u);}
  for(var j=0u;j<128u;j=j+1u){
   let index=b*128u+j;
   var byte=0u;
   if(index<len){
    byte=input_byte(kind,index,previous,suffix,suffix_len,block_index);
   }else if(index==len){
    byte=128u;
   }else if(index>=padded_blocks*128u-8u){
    let shift=(padded_blocks*128u-1u-index)*8u;
    byte=u32((total_bits>>shift)&u64(255u));
   }
   words[j/8u]=words[j/8u]|(u64(byte)<<((7u-j%8u)*8u));
  }
  state=sha512_compress(state,words);
 }
 return state;
}
fn expected_nibble(index:u32)->u32 {
 return (P[P[11]+index/8u]>>((7u-index%8u)*4u))&15u;
}
fn matches_nibbles(digest:array<u64,8>,start_nibble:u32,count:u32)->bool {
 for(var j=0u;j<count;j=j+1u){
  let expected=expected_nibble(start_nibble+j);
  let actual=u32((digest[j/16u]>>((15u-j%16u)*4u))&u64(15u));
  if(actual!=expected){return false;}
 }
 return true;
}
fn make_counter(counter:u64)->array<u32,20>{
 var result:array<u32,20>;
 if(P[7]==0u){
  let binary=u32(counter);
  for(var i=0u;i<4u;i=i+1u){result[i]=(binary>>(i*8u))&255u;}
 }else{
  var value=counter;
  var i=0u;
  loop {
   result[i]=48u+u32(value%u64(10u));
   i=i+1u;
   value=value/u64(10u);
   if(value==u64(0u)){break;}
  }
 }
 return result;
}
fn counter_length(counter:u64)->u32 {
 if(P[7]==0u){return 4u;}
 var value=counter;
 var digits=0u;
 loop{
  digits=digits+1u;
  value=value/u64(10u);
  if(value==u64(0u)){break;}
 }
 return digits;
}
fn hmac_key_state(iv:array<u64,8>,suffix:array<u32,20>,suffix_len:u32,
                  short_hash:array<u64,8>,hash_long:bool,pad:u32)->array<u64,8> {
 var block:array<u64,16>;
 for(var i=0u;i<16u;i=i+1u){block[i]=u64(0u);}
 let password_len=P[5]+suffix_len;
 for(var i=0u;i<128u;i=i+1u){
  var byte=0u;
  if(hash_long){
   if(i<digest_width()){byte=digest_byte(short_hash,i);}
  }else if(i<password_len){
   byte=input_byte(0u,i,short_hash,suffix,suffix_len,1u);
  }
  block[i/8u]=block[i/8u]|(u64(byte^pad)<<((7u-i%8u)*8u));
 }
 return sha512_compress(iv,block);
}
fn hmac(ipad:array<u64,8>,opad:array<u64,8>,source:u32,
        previous:array<u64,8>,suffix:array<u32,20>,suffix_len:u32,
        block_index:u32)->array<u64,8> {
 let inner=hash_source(ipad,128u,source,previous,suffix,suffix_len,block_index);
 return hash_source(opad,128u,1u,inner,suffix,suffix_len,block_index);
}
fn matches_candidate(iv:array<u64,8>,suffix:array<u32,20>,suffix_len:u32)->bool{
 var empty:array<u64,8>;
 for(var i=0u;i<8u;i=i+1u){empty[i]=u64(0u);}
 if(P[6]==0u){return true;}
 if(P[13]==0u){
  var digest=hash_source(iv,0u,0u,empty,suffix,suffix_len,1u);
  for(var iteration=1u;iteration<P[4];iteration=iteration+1u){
   digest=hash_source(iv,0u,1u,digest,suffix,suffix_len,1u);
  }
  return matches_nibbles(digest,0u,P[6]);
 }
 let password_len=P[5]+suffix_len;
 let hash_long=password_len>128u;
 var long_key=empty;
 if(hash_long){long_key=hash_source(iv,0u,0u,empty,suffix,suffix_len,1u);}
 let ipad=hmac_key_state(iv,suffix,suffix_len,long_key,hash_long,0x36u);
 let opad=hmac_key_state(iv,suffix,suffix_len,long_key,hash_long,0x5cu);
 let nibbles_per_block=digest_width()*2u;
 let blocks=(P[6]+nibbles_per_block-1u)/nibbles_per_block;
 for(var block=1u;block<=blocks;block=block+1u){
  var u=hmac(ipad,opad,2u,empty,suffix,suffix_len,block);
  var key=u;
  for(var iteration=1u;iteration<P[4];iteration=iteration+1u){
   u=hmac(ipad,opad,1u,u,suffix,suffix_len,block);
   for(var i=0u;i<8u;i=i+1u){key[i]=key[i]^u[i];}
  }
  let begin=(block-1u)*nibbles_per_block;
  let count=min(nibbles_per_block,P[6]-begin);
  if(!matches_nibbles(key,begin,count)){return false;}
 }
 return true;
}
@compute @workgroup_size(__WG__)
fn solve(@builtin(global_invocation_id) gid:vec3<u32>){
 if(gid.x>=P[2]||atomicLoad(&R.flag)!=0u){return;}
 let counter=((u64(P[3])<<32u)|u64(P[1]))+u64(gid.x);
 let suffix=make_counter(counter);
 let suffix_len=counter_length(counter);
 let iv=hash_iv();
 if(matches_candidate(iv,suffix,suffix_len)){
  if(atomicExchange(&R.flag,1u)==0u){
   R.nonce_low=u32(counter);
   R.nonce_high=u32(counter>>32u);
  }
 }
}
