// Generated directed BlockMix: source/destination and scratch mode are
// compile-time fixed for the workgroup, not tested for each u32 word.
fn __NAME__(lane:u32,v_base:u32) {
 let words=32u*params.r;
 let base=(lane+params.offset)*words;
 let aux=lane*words;
 let last=(2u*params.r-1u)*16u;
 var x:array<u32,16>;
 for(var j=0u;j<16u;j++) {
  x[j]=__SRC__[__SRC_BASE__+last+j];
  __XOR_INITIAL__
 }
 for(var i=0u;i<2u*params.r;i++) {
  for(var j=0u;j<16u;j++) {
   let pos=i*16u+j;
   var word=__SRC__[__SRC_BASE__+pos];
   __XOR_WORD__
   x[j]^=word;
  }
  x=salsa20_8(x);
  let destination=(i/2u)+select(0u,params.r,(i&1u)!=0u);
  for(var j=0u;j<16u;j++) {
   __DEST__[__DEST_BASE__+destination*16u+j]=x[j];
  }
 }
}
