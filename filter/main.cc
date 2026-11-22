#include "gpu_opencl.hh"
#include "gpu_scan.hh"
#include "filter.hh"
#include "linear-algebra.hh"
#include "reduce-scan.hh"

#include <array>
#include <chrono>
#include <limits>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using clock_type=std::chrono::steady_clock;
using time_point=clock_type::time_point;

static const std::string src=std::string(R"OPENCL(
// Stable local compaction: predicate + per-thread scan + workgroup scan.
// Only one count per 1024 inputs needs a global prefix scan.
__kernel void compact_blocks(__global const float* input,
                             __global float* packed,
                             __global int* totals,const uint n){
    const uint lane=get_local_id(0),group=get_group_id(0);
    const uint base=group*1024+lane*4;
    float4 values=(float4)(0);
    if(base+3<n)values=vload4(0,input+base);
    else for(uint k=0;k<4;++k)if(base+k<n)values[k]=input[base+k];
    int4 flags=(int4)(values.x>0,values.y>0,values.z>0,values.w>0);
    const int count=flags.x+flags.y+flags.z+flags.w;
    __local int offsets[256];
    offsets[lane]=count;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=1;stride<256;stride*=2){
        const uint idx=(lane+1)*2*stride-1;
        if(idx<256)offsets[idx]+=offsets[idx-stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0){totals[group]=offsets[255];offsets[255]=0;}
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=128;stride>0;stride/=2){
        const uint idx=(lane+1)*2*stride-1;
        if(idx<256){int prev=offsets[idx-stride];offsets[idx-stride]=offsets[idx];offsets[idx]+=prev;}
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    uint dest=group*1024+offsets[lane];
    for(uint k=0;k<4;++k)if(flags[k])packed[dest++]=values[k];
}

// GPU-scanned group totals determine the final stable destination ranges.
__kernel void scatter_blocks(__global const float* packed,
                             __global const int* prefix,
                             __global float* output){
    const uint group=get_group_id(0),lane=get_local_id(0);
    const uint start=group?prefix[group-1]:0;
    const uint count=prefix[group]-start;
    for(uint i=lane*4;i<count;i+=1024){
        if(i+3<count)vstore4(vload4(0,packed+group*1024+i),0,output+start+i);
        else for(uint k=0;k<4;++k)if(i+k<count)output[start+i+k]=packed[group*1024+i+k];
    }
}
)OPENCL")+SCAN_OPENCL_SOURCE;

void print_columns(){
    std::cout<<std::setw(19)<<"function";
    for(const char* s:{"OpenMP","OpenCL total","OpenCL copy-in","OpenCL kernel","OpenCL copy-out"})
        std::cout<<std::setw(20)<<s;
    std::cout<<'\n';
}
void print(const char* name,time_point t0,time_point t1,time_point t2,time_point t3,time_point t4){
    using namespace std::chrono;
    std::cout<<std::setw(19)<<name;
    for(auto dt:{t1-t0,t4-t1,t2-t1,t3-t2,t4-t3}){
        std::ostringstream os;os<<duration_cast<microseconds>(dt).count()<<"us";
        std::cout<<std::setw(20)<<os.str();
    }
    std::cout<<'\n';
}

using Sample=std::array<time_point,5>;
void print_median(std::vector<Sample> samples){
    std::vector<clock_type::duration> cpu;
    for(const auto& s:samples)cpu.push_back(s[1]-s[0]);
    std::sort(cpu.begin(),cpu.end());
    std::sort(samples.begin(),samples.end(),[](const Sample& a,const Sample& b){
        return a[4]-a[1]<b[4]-b[1];
    });
    auto s=samples[samples.size()/2];
    s[0]=s[1]-cpu[cpu.size()/2];
    print("filter",s[0],s[1],s[2],s[3],s[4]);
}

void profile_filter(const std::vector<float>& input,OpenCL& gpu,int repeats=7){
    const size_t n=input.size(),groups=(n+1023)/1024;
    if(n>static_cast<size_t>(std::numeric_limits<cl_int>::max()))
        throw std::runtime_error("Input exceeds 32-bit prefix capacity");
    std::vector<float> expected_result;
    expected_result.reserve(n);
    std::vector<Sample> samples;
    for(int trial=-1;trial<repeats;++trial){
        expected_result.clear();
        const auto t0=clock_type::now();
        filter(input,expected_result,[](float x){return x>0.0f;});
        const auto t1=clock_type::now();
        // All OpenCL buffer allocations stay inside the end-to-end measurement.
        // input remains alive and unmodified until din is released.
        auto din=gpu.host_input(input.data(),n*sizeof(float));
        auto packed=gpu.buffer(n*sizeof(float));
        auto totals=gpu.buffer(groups*sizeof(cl_int));
        auto prefix=gpu.buffer(groups*sizeof(cl_int));
        auto dout=gpu.buffer(n*sizeof(float));
        const auto t2=clock_type::now();
        if(n>0){
            const cl_uint count=static_cast<cl_uint>(n);
            auto compact=gpu.kernel("compact_blocks");
            compact.mem(0,din);compact.mem(1,packed);compact.mem(2,totals);compact.arg(3,count);
            gpu.run(compact,groups*256,256);
            gpu_scan_impl(gpu,totals,prefix,groups,true);
            auto scatter=gpu.kernel("scatter_blocks");
            scatter.mem(0,packed);scatter.mem(1,prefix);scatter.mem(2,dout);
            gpu.run(scatter,groups*256,256);
            gpu.finish();
        }
        const auto t3=clock_type::now();
        cl_int valid_count=0;
        // Reading the final count does not perform any CPU reduction/scan.
        if(n>0)gpu.read(prefix,&valid_count,sizeof(valid_count),(groups-1)*sizeof(cl_int));
        if(valid_count<0 || size_t(valid_count)>n)throw std::runtime_error("Invalid compacted count");
        std::vector<float> result(static_cast<size_t>(valid_count));
        if(valid_count>0)gpu.read(dout,result.data(),size_t(valid_count)*sizeof(float));
        const auto t4=clock_type::now();
        // Filtering does not change values: exact equality checks count and order,
        // including +infinity. NaNs, negatives and both signed zeros are excluded.
        if(result!=expected_result)throw std::runtime_error("Stable GPU filter mismatch");
        if(trial>=0)samples.push_back({t0,t1,t2,t3,t4});
    }
    print_median(samples);
}

int main(int argc,char** argv){
    try{
        bool quick=false,test=false;
        for(int i=1;i<argc;++i){
            if(std::string(argv[i])=="--quick")quick=true;
            else if(std::string(argv[i])=="--test")test=true;
            else throw std::runtime_error("Usage: filter [--quick | --test]");
        }
        OpenCL gpu(src);
        std::cout<<"Timing: one warm-up, "<<(test?1:7)<<" measured runs; median wall time.\n"
                 <<"OpenCL total includes buffers, input access, kernels, count and output reads.\n"
                 <<"CPU baseline (OpenMP column): original sequential stable filter.\n";
        print_columns();
        if(test){
            for(size_t n:{size_t(0),size_t(1),size_t(2),size_t(17),size_t(255),size_t(256),size_t(257),size_t(511),size_t(512),size_t(513),size_t(1023),size_t(1024),size_t(1025),size_t(262149),size_t(1048577)}){
                std::cout<<"n="<<n<<'\n';
                auto input=random_std_vector<float>(static_cast<int>(n));
                if(n>0)input[n/2]=0.0f;
                profile_filter(input,gpu,1);
            }
            profile_filter(std::vector<float>(1025,-1.0f),gpu,1);
            profile_filter(std::vector<float>(1025,0.0f),gpu,1);
            profile_filter(std::vector<float>(1025,1.0f),gpu,1);
            profile_filter({-2.0f,0.0f,3.0f,-1.0f,5.0f,7.0f,-0.0f},gpu,1);
            const float inf=std::numeric_limits<float>::infinity();
            profile_filter({inf,-inf,std::numeric_limits<float>::quiet_NaN(),1.0f,0.0f},gpu,1);
        }else{
            const size_t n=quick?65536:size_t(1024)*1024;
            profile_filter(random_std_vector<float>(static_cast<int>(n)),gpu);
        }
        std::cout<<"Verification: PASS\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
