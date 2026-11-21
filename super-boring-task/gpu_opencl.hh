#pragma once
#include "minicl.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

inline void cl_check(cl_int rc, const char* where) {
    if (rc != CL_SUCCESS) throw std::runtime_error(std::string(where) + " returned OpenCL error " + std::to_string(rc));
}
inline size_t cl_round(size_t n, size_t block) { return (n+block-1)/block*block; }

struct CLBuffer {
    cl_mem h = nullptr;
    explicit CLBuffer(cl_mem obj=nullptr):h(obj) {}
    ~CLBuffer(){ if(h) clReleaseMemObject(h); }
    CLBuffer(const CLBuffer&)=delete;
    CLBuffer& operator=(const CLBuffer&)=delete;
    CLBuffer(CLBuffer&& x) noexcept : h(x.h) { x.h=nullptr; }
    CLBuffer& operator=(CLBuffer&& x) noexcept {
        if(this!=&x) { if(h) clReleaseMemObject(h); h=x.h; x.h=nullptr; } return *this;
    }
};
struct CLKernel {
    cl_kernel h;
    explicit CLKernel(cl_kernel obj):h(obj) {}
    ~CLKernel(){if(h)clReleaseKernel(h);}
    CLKernel(const CLKernel&)=delete;
    CLKernel& operator=(const CLKernel&)=delete;
    template<class T> void arg(cl_uint i,const T& value) {
        cl_check(clSetKernelArg(h,i,sizeof(T),&value),"clSetKernelArg");
    }
    void mem(cl_uint i,const CLBuffer& buffer){arg(i,buffer.h);}
};
class OpenCL {
    cl_platform_id platform_=nullptr;
    cl_device_id device_=nullptr;
    cl_context context_=nullptr;
    cl_command_queue queue_=nullptr;
    cl_program program_=nullptr;
public:
    explicit OpenCL(const std::string& source) {
        try {
            cl_uint n=0;
            cl_int rc=clGetPlatformIDs(0,nullptr,&n);
            if(rc==-1001 || n==0) throw std::runtime_error("No OpenCL platforms. Install an OpenCL GPU driver/ICD; CPU-only test: make cpu-test");
            cl_check(rc,"clGetPlatformIDs(count)");
            std::vector<cl_platform_id> platforms(n);
            cl_check(clGetPlatformIDs(n,platforms.data(),nullptr),"clGetPlatformIDs(list)");
            for(auto kind:{CL_DEVICE_TYPE_GPU,CL_DEVICE_TYPE_CPU}) {
                for(auto plat:platforms) {
                    if(clGetDeviceIDs(plat,kind,1,&device_,nullptr)==CL_SUCCESS){platform_=plat;break;}
                }
                if(device_)break;
            }
            if(!device_)throw std::runtime_error("No compatible OpenCL devices");
            char platform_name[256]={},device_name[256]={};
            clGetPlatformInfo(platform_,CL_PLATFORM_NAME,sizeof(platform_name),platform_name,nullptr);
            clGetDeviceInfo(device_,CL_DEVICE_NAME,sizeof(device_name),device_name,nullptr);
            std::clog << "Platform name: " << platform_name << "\nDevice name: " << device_name << '\n';
            size_t max_wg=0;
            cl_check(clGetDeviceInfo(device_,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(max_wg),&max_wg,nullptr),"CL_DEVICE_MAX_WORK_GROUP_SIZE");
            if(max_wg<256) throw std::runtime_error("Device must support 256-work-item workgroups");
            cl_int err=CL_SUCCESS;
            context_=clCreateContext(nullptr,1,&device_,nullptr,nullptr,&err);cl_check(err,"clCreateContext");
            queue_=clCreateCommandQueue(context_,device_,0,&err);cl_check(err,"clCreateCommandQueue");
            const char* src=source.c_str(); size_t length=source.size();
            program_=clCreateProgramWithSource(context_,1,&src,&length,&err);cl_check(err,"clCreateProgramWithSource");
            err=clBuildProgram(program_,1,&device_,"-cl-std=CL1.2",nullptr,nullptr);
            if(err!=CL_SUCCESS) {
                size_t len=0;clGetProgramBuildInfo(program_,device_,CL_PROGRAM_BUILD_LOG,0,nullptr,&len);
                std::string log(len,'\0');clGetProgramBuildInfo(program_,device_,CL_PROGRAM_BUILD_LOG,len,log.data(),nullptr);
                std::cerr << "OpenCL build log:\n" << log << '\n';
                cl_check(err,"clBuildProgram");
            }
        } catch(...) { cleanup();throw; }
    }
    ~OpenCL(){cleanup();}
    OpenCL(const OpenCL&)=delete; OpenCL& operator=(const OpenCL&)=delete;
    CLBuffer buffer(size_t bytes){
        cl_int err=CL_SUCCESS;
        cl_mem mem=clCreateBuffer(context_,CL_MEM_READ_WRITE,std::max<size_t>(bytes,1),nullptr,&err);
        cl_check(err,"clCreateBuffer"); return CLBuffer(mem);
    }
    CLKernel kernel(const char* name){
        cl_int err=CL_SUCCESS;cl_kernel k=clCreateKernel(program_,name,&err);cl_check(err,"clCreateKernel");return CLKernel(k);
    }
    void write(const CLBuffer& b, const void* src, size_t bytes){
        if(bytes)cl_check(clEnqueueWriteBuffer(queue_,b.h,CL_TRUE,0,bytes,src,0,nullptr,nullptr),"clEnqueueWriteBuffer");
    }
    void read(const CLBuffer& b, void* dst, size_t bytes, size_t offset=0){
        if(bytes)cl_check(clEnqueueReadBuffer(queue_,b.h,CL_TRUE,offset,bytes,dst,0,nullptr,nullptr),"clEnqueueReadBuffer");
    }
    void run(CLKernel& kernel,size_t x,size_t localx,size_t y=0,size_t localy=0){
        if(y==0){const size_t global[]={x}, local[]={localx};cl_check(clEnqueueNDRangeKernel(queue_,kernel.h,1,nullptr,global,local,0,nullptr,nullptr),"clEnqueueNDRangeKernel(1D)");}
        else {const size_t global[]={x,y},local[]={localx,localy};cl_check(clEnqueueNDRangeKernel(queue_,kernel.h,2,nullptr,global,local,0,nullptr,nullptr),"clEnqueueNDRangeKernel(2D)");}
    }
    void finish(){cl_check(clFinish(queue_),"clFinish");}
private:
    void cleanup(){
        if(program_)clReleaseProgram(program_);
        if(queue_)clReleaseCommandQueue(queue_);
        if(context_)clReleaseContext(context_);
        program_=nullptr;queue_=nullptr;context_=nullptr;
    }
};
