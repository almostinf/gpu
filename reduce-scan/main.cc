#include "gpu_opencl.hh"
#include "gpu_scan.hh"
#include "linear-algebra.hh"
#include "reduce-scan.hh"

#include <array>
#include <chrono>
#include <memory>
#include <new>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using clock_type=std::chrono::steady_clock;
using time_point=clock_type::time_point;

// Page-aligned host storage lets integrated GPUs use CL_MEM_USE_HOST_PTR
// without a staging copy. Storage outlives every OpenCL buffer that references it.
class HostVector {
    struct Delete { void operator()(float* p) const {
        ::operator delete[](p,std::align_val_t{16384});
    }};
    std::unique_ptr<float[],Delete> data_;
    size_t size_;
public:
    using value_type=float;
    explicit HostVector(size_t n):data_(static_cast<float*>(
        ::operator new[](std::max<size_t>(n,1)*sizeof(float),std::align_val_t{16384}))),size_(n) {}
    size_t size() const {return size_;}
    float* begin(){return data_.get();}
    const float* begin() const {return data_.get();}
    float* end(){return begin()+size_;}
    float& operator()(size_t i){return data_[i];}
    float operator()(size_t i) const {return data_[i];}
};
HostVector random_host_vector(size_t n){
    HostVector a(n);
    auto prng=make_prng();std::uniform_real_distribution<float> dist(0,1);
    for(size_t i=0;i<n;++i)a(i)=dist(prng);
    return a;
}

static const std::string src=std::string(R"OPENCL(
// Entire reduction is completed on the GPU. A subsequent pass reduces the partials.
__kernel void reduce_pass(__global const float* src,__global float* partial,const uint n) {
    const uint lane=get_local_id(0),group=get_group_id(0);
    float4 acc=(float4)(0);
    // Coalesced grid-stride loads amortize the local-memory reduction.
    for(uint i=group*1024+lane*4;i<n;i+=(uint)get_num_groups(0)*1024){
        if(i+3<n)acc+=vload4(0,src+i);
        else for(uint k=0;k<4;++k)if(i+k<n)acc[k]+=src[i+k];
    }
    const float val=(acc.x+acc.y)+(acc.z+acc.w);
    __local float work[256];
    work[lane]=val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint step=128;step>0;step>>=1){
        if(lane<step)work[lane]+=work[lane+step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0)partial[group]=work[0];
}
)OPENCL")+SCAN_OPENCL_SOURCE;

void print_cols(){
    std::cout<<std::setw(21)<<"function";
    for(const char* s:{"OpenMP","OpenCL total","OpenCL copy-in","OpenCL kernel","OpenCL copy-out"})
        std::cout<<std::setw(20)<<s;
    std::cout<<'\n';
}
void print(const char* s,time_point a,time_point b,time_point c,time_point d,time_point e){
    using namespace std::chrono;
    std::cout<<std::setw(21)<<s;
    for(const auto dt:{b-a,e-b,c-b,d-c,e-d}) {
        std::ostringstream tmp;tmp<<duration_cast<microseconds>(dt).count()<<"us";
        std::cout<<std::setw(20)<<tmp.str();
    }
    std::cout<<'\n';
}

using Sample=std::array<time_point,5>;
void print_median(const char* name,std::vector<Sample> samples){
    std::vector<clock_type::duration> cpu;
    for(const auto& s:samples)cpu.push_back(s[1]-s[0]);
    std::sort(cpu.begin(),cpu.end());
    std::sort(samples.begin(),samples.end(),[](const Sample& a,const Sample& b){
        return a[4]-a[1]<b[4]-b[1];
    });
    auto s=samples[samples.size()/2];
    s[0]=s[1]-cpu[cpu.size()/2];
    print(name,s[0],s[1],s[2],s[3],s[4]);
}

// O(n) validation using a double-precision reference to avoid serial float-scan drift.
void check_scan(const HostVector& input,const HostVector& output){
    double acc=0;
    for(size_t i=0;i<input.size();++i){
        acc+=input(i);
        const double actual=output(i), tolerance=0.015+2e-5*std::abs(acc);
        if(!std::isfinite(actual) || std::abs(acc-actual)>tolerance){
            std::ostringstream ss;ss<<"scan mismatch at "<<i<<": high precision "<<acc<<", GPU "<<actual
                <<", tolerance "<<tolerance;
            throw std::runtime_error(ss.str());
        }
    }
}

void profile_reduce(size_t n,OpenCL& gpu,int repeats=7){
    auto a=random_host_vector(n);
    std::vector<Sample> samples;
    for(int trial=-1;trial<repeats;++trial){
        const auto t0=clock_type::now();
        volatile float cpu_result=reduce(a);
        const auto t1=clock_type::now();
        auto da=gpu.host_buffer(std::begin(a),n*sizeof(float));
        const auto t2=clock_type::now();
        // Only the final scalar is downloaded; partial sums stay on the GPU.
        const CLBuffer* current=&da;
        size_t remaining=n;
        std::vector<CLBuffer> intermediates;
        while(remaining>1){
            const size_t groups=std::min<size_t>(1024,(remaining+4095)/4096);
            auto next=gpu.buffer(groups*sizeof(float));
            auto k=gpu.kernel("reduce_pass");
            const cl_uint count=static_cast<cl_uint>(remaining);
            k.mem(0,*current);k.mem(1,next);k.arg(2,count);
            gpu.run(k,groups*256,256);
            intermediates.push_back(std::move(next));
            current=&intermediates.back();remaining=groups;
        }
        gpu.finish();
        const auto t3=clock_type::now();
        float result=0;gpu.read(*current,&result,sizeof(float));
        const auto t4=clock_type::now();
        double exact=0;for(size_t i=0;i<n;++i)exact+=a(i);
        const double tolerance=0.025+2e-5*std::abs(exact);
        if(!std::isfinite(result) || std::abs(double(result)-exact)>tolerance)
            throw std::runtime_error("reduce mismatch vs double-precision reference");
        (void)cpu_result;
        if(trial>=0)samples.push_back({t0,t1,t2,t3,t4});
    }
    print_median("reduce",samples);
}

void profile_scan_inclusive(size_t n,OpenCL& gpu,int repeats=7){
    auto a=random_host_vector(n);
    HostVector result(n);
    Vector<float> expected(n);
    std::vector<Sample> samples;
    for(int trial=-1;trial<repeats;++trial){
        std::copy(a.begin(),a.end(),std::begin(expected));
        const auto t0=clock_type::now();scan_inclusive(expected);
        const auto t1=clock_type::now();
        // Consume the CPU result so the timed scan cannot be optimized away.
        volatile float cpu_last=expected(n-1);(void)cpu_last;
        auto da=gpu.host_buffer(std::begin(a),n*sizeof(float));
        auto db=gpu.host_buffer(std::begin(result),n*sizeof(float));
        const auto t2=clock_type::now();
        gpu_scan_impl(gpu,da,db,n,false);gpu.finish();
        const auto t3=clock_type::now();
        gpu.read(db,std::begin(result),n*sizeof(float));
        const auto t4=clock_type::now();
        check_scan(a,result);
        if(trial>=0)samples.push_back({t0,t1,t2,t3,t4});
    }
    print_median("scan-inclusive",samples);
}

// Exercise the integer variant as well: negative values expose offset mistakes
// that an all-positive float input can hide behind its relative tolerance.
void check_integer_scan(size_t n,OpenCL& gpu){
    Vector<cl_int> input(n),output(n);
    for(size_t i=0;i<n;++i)input(i)=static_cast<cl_int>(i%11)-5;
    auto a=gpu.buffer(n*sizeof(cl_int)),b=gpu.buffer(n*sizeof(cl_int));
    gpu.write(a,std::begin(input),n*sizeof(cl_int));
    gpu_scan_impl(gpu,a,b,n,true);
    gpu.read(b,std::begin(output),n*sizeof(cl_int));
    cl_int sum=0;
    for(size_t i=0;i<n;++i){
        sum+=input(i);
        if(output(i)!=sum)throw std::runtime_error("integer scan mismatch at "+std::to_string(i));
    }
}

int main(int argc,char** argv){
    try {
        bool quick=false,test=false;
        for(int i=1;i<argc;++i){
            if(std::string(argv[i])=="--quick")quick=true;
            else if(std::string(argv[i])=="--test")test=true;
            else throw std::runtime_error("Usage: reduce-scan [--quick | --test]");
        }
        OpenCL gpu(src);
        std::cout<<"Timing: one warm-up, "<<(test?1:7)<<" measured runs; median wall time.\n"
                 <<"OpenCL total includes buffer creation, host-memory access, kernels and result synchronization.\n"
                 <<"CPU baseline: OpenMP SIMD reduction; sequential inclusive scan.\n";
        print_cols();
        if(test){
            for(size_t n:{size_t(1),size_t(2),size_t(17),size_t(255),size_t(256),size_t(257),size_t(511),size_t(512),size_t(513),size_t(1023),size_t(1024),size_t(1025),size_t(4095),size_t(4096),size_t(4097),size_t(262149),size_t(1048581)}){
                std::cout<<"n="<<n<<'\n';profile_reduce(n,gpu,1);profile_scan_inclusive(n,gpu,1);check_integer_scan(n,gpu);
            }
        }else {
            const size_t n=quick?65536:size_t(1024)*1024*10;
            profile_reduce(n,gpu);profile_scan_inclusive(n,gpu);
        }
        std::cout<<"Verification: PASS\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
