#include <cuda.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <algorithm>
static void check(CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char* msg{};
        cuGetErrorString(status,&msg);
        throw std::runtime_error(msg?msg:"CUDA error");
    }
}
int main(int argc,char**argv) try {
    if(argc!=3)return 2;
    check(cuInit(0));
    CUdevice dev{}; check(cuDeviceGet(&dev,0));
    CUcontext ctx{}; check(cuDevicePrimaryCtxRetain(&ctx,dev)); check(cuCtxSetCurrent(ctx));
    CUmodule mod{}; check(cuModuleLoad(&mod,argv[1]));
    CUfunction fn{}; check(cuModuleGetFunction(&fn,mod,"kern"));
    CUdeviceptr result{}; check(cuMemAlloc(&result,4));
    uint32_t base,batch,difficulty,want;
    std::ifstream cases(argv[2]);
    unsigned n=0;
    while(cases>>base>>batch>>difficulty>>want){
        uint32_t m0=0,m1=0;
        for(unsigned i=0;i<difficulty;++i) {
            unsigned bit=(i/8)*8+7-i%8;
            if(bit<32)m0|=1u<<bit;else m1|=1u<<(bit-32);
        }
        check(cuMemsetD32(result,UINT32_MAX,1));
        void* args[]{&result,&base,&m0,&m1};
        check(cuLaunchKernel(fn,batch/32,1,1,32,1,1,0,nullptr,args,nullptr));
        uint32_t got{};
        check(cuMemcpyDtoH(&got,result,4));
        printf("base=%u difficulty=%u want=%u got=%u\n",base,difficulty,want,got);
        if(got!=want)return 1;
        ++n;
    }
    if(n==0)return 2;
    // 16 large launches per sample, no per-launch host readback.
    base=0;batch=1048576;uint32_t full=UINT32_MAX;
    void* args[]{&result,&base,&full,&full};
    check(cuMemsetD32(result,UINT32_MAX,1));
    check(cuLaunchKernel(fn,batch/32,1,1,32,1,1,0,nullptr,args,nullptr));
    check(cuCtxSynchronize());
    CUevent begin{},end{};check(cuEventCreate(&begin,0));check(cuEventCreate(&end,0));
    std::vector<float> samples;
    for(int trial=0;trial<5;++trial){
        check(cuEventRecord(begin,nullptr));
        for(unsigned i=0;i<16;++i){
            base=i*batch;
            check(cuLaunchKernel(fn,batch/32,1,1,32,1,1,0,nullptr,args,nullptr));
        }
        check(cuEventRecord(end,nullptr));check(cuEventSynchronize(end));
        float ms{};check(cuEventElapsedTime(&ms,begin,end));
        samples.push_back(ms);
    }
    std::sort(samples.begin(),samples.end());
    uint32_t got{};check(cuMemcpyDtoH(&got,result,4));
    if(got!=UINT32_MAX)return 1;
    printf("verified=%u median_ms=%.6f throughput_MHs=%.3f\n",n,double(samples[2]),double(batch)*16/(double(samples[2])*1000));
    check(cuEventDestroy(begin));check(cuEventDestroy(end));check(cuMemFree(result));check(cuModuleUnload(mod));check(cuDevicePrimaryCtxRelease(dev));
    return 0;
} catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}
