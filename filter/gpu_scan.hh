#pragma once
#include "gpu_opencl.hh"
#include <cstdint>
#include <string>
static const char* SCAN_OPENCL_SOURCE=R"OPENCL(
// Four consecutive values per work-item: scan in registers, then scan only
// the 256 thread totals in local memory. Each group handles 1024 values.
__kernel void scan_float_blocks(__global const float* input,__global float* output,
                              __global float* totals,const uint n){
    const uint tid=get_local_id(0),group=get_group_id(0);
    const uint base=group*1024+tid*4;
    float4 v=(float4)(0);
    if(base+3<n)v=vload4(0,input+base);
    else for(uint k=0;k<4;++k)if(base+k<n)v[k]=input[base+k];
    v.y+=v.x;v.z+=v.y;v.w+=v.z;
    __local float shared[256];
    shared[tid]=v.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=1;stride<256;stride*=2){
        const uint idx=(tid+1)*2*stride-1;
        if(idx<256)shared[idx]+=shared[idx-stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(tid==0){totals[group]=shared[255];shared[255]=0;}
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=128;stride>0;stride/=2){
        const uint idx=(tid+1)*2*stride-1;
        if(idx<256){float prev=shared[idx-stride];shared[idx-stride]=shared[idx];shared[idx]+=prev;}
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    v+=(float4)(shared[tid]);
    if(base+3<n)vstore4(v,0,output+base);
    else for(uint k=0;k<4;++k)if(base+k<n)output[base+k]=v[k];
}
__kernel void scan_float_add(__global float* output,__global const float* scanned_totals,const uint n){
    const uint base=get_global_id(0)*4;
    const uint group=base/1024;
    if(group>0 && base<n){
        float offset=scanned_totals[group-1];
        if(base+3<n)vstore4(vload4(0,output+base)+(float4)(offset),0,output+base);
        else for(uint k=0;k<4;++k)if(base+k<n)output[base+k]+=offset;
    }
}
// Four consecutive values per work-item: scan in registers, then scan only
// the 256 thread totals in local memory. Each group handles 1024 values.
__kernel void scan_int_blocks(__global const int* input,__global int* output,
                              __global int* totals,const uint n){
    const uint tid=get_local_id(0),group=get_group_id(0);
    const uint base=group*1024+tid*4;
    int4 v=(int4)(0);
    if(base+3<n)v=vload4(0,input+base);
    else for(uint k=0;k<4;++k)if(base+k<n)v[k]=input[base+k];
    v.y+=v.x;v.z+=v.y;v.w+=v.z;
    __local int shared[256];
    shared[tid]=v.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=1;stride<256;stride*=2){
        const uint idx=(tid+1)*2*stride-1;
        if(idx<256)shared[idx]+=shared[idx-stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(tid==0){totals[group]=shared[255];shared[255]=0;}
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=128;stride>0;stride/=2){
        const uint idx=(tid+1)*2*stride-1;
        if(idx<256){int prev=shared[idx-stride];shared[idx-stride]=shared[idx];shared[idx]+=prev;}
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    v+=(int4)(shared[tid]);
    if(base+3<n)vstore4(v,0,output+base);
    else for(uint k=0;k<4;++k)if(base+k<n)output[base+k]=v[k];
}
__kernel void scan_int_add(__global int* output,__global const int* scanned_totals,const uint n){
    const uint base=get_global_id(0)*4;
    const uint group=base/1024;
    if(group>0 && base<n){
        int offset=scanned_totals[group-1];
        if(base+3<n)vstore4(vload4(0,output+base)+(int4)(offset),0,output+base);
        else for(uint k=0;k<4;++k)if(base+k<n)output[base+k]+=offset;
    }
}
)OPENCL";

// Inclusive scan completed entirely on device: host schedules kernels but never adds offsets.
// Each workgroup scans up to 1024 values; the sums array is scanned recursively.
inline void gpu_scan_impl(OpenCL& gpu,const CLBuffer& input,CLBuffer& output,
                          size_t n,bool integer_type) {
    if(n==0)return;
    const size_t groups=(n+1023)/1024;
    CLBuffer totals=gpu.buffer(groups*(integer_type?sizeof(cl_int):sizeof(float)));
    auto part=gpu.kernel(integer_type?"scan_int_blocks":"scan_float_blocks");
    const cl_uint count=static_cast<cl_uint>(n);
    part.mem(0,input);part.mem(1,output);part.mem(2,totals);part.arg(3,count);
    gpu.run(part,groups*256,256);
    if(groups>1) {
        CLBuffer prefix=gpu.buffer(groups*(integer_type?sizeof(cl_int):sizeof(float)));
        gpu_scan_impl(gpu,totals,prefix,groups,integer_type);
        auto add=gpu.kernel(integer_type?"scan_int_add":"scan_float_add");
        add.mem(0,output);add.mem(1,prefix);add.arg(2,count);
        gpu.run(add,cl_round((n+3)/4,256),256);
    }
}
