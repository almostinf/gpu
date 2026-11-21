#include "gpu_opencl.hh"
#include "linear-algebra.hh"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using clock_type=std::chrono::steady_clock;
using duration=clock_type::duration;
using time_point=clock_type::time_point;

static const std::string src=R"OPENCL(
__kernel void vector_times_vector(__global const float* a,
                                  __global const float* b,
                                  __global float* output,const uint n) {
    const uint i=get_global_id(0);
    if(i<n)output[i]=a[i]*b[i];
}

// One workgroup per row, contiguous loads across its 256 work-items.
// Reduction is performed with shared/local memory without CPU participation.
__kernel void matrix_times_vector(__global const float* a,
                                  __global const float* b,
                                  __global float* output,const uint n) {
    const uint row=get_group_id(0);
    const uint lane=get_local_id(0);
    __local float partial[256];
    float sum=0.0f;
    for(uint col=lane;col<n;col+=256)sum+=a[row*n+col]*b[col];
    partial[lane]=sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint step=128;step>0;step>>=1){
        if(lane<step)partial[lane]+=partial[lane+step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0)output[row]=partial[0];
}

// C=A*B, 16x16 output tile. Cooperatively stage A/B tiles in local memory.
__kernel void matrix_times_matrix(__global const float* a,
                                  __global const float* b,
                                  __global float* output,const uint n) {
    const uint x=get_global_id(0),y=get_global_id(1);
    const uint tx=get_local_id(0),ty=get_local_id(1);
    __local float tile_a[16][16];
    __local float tile_b[16][16];
    float acc=0.0f;
    for(uint base=0;base<n;base+=16){
        const uint acol=base+tx;
        const uint brow=base+ty;
        tile_a[ty][tx]=(y<n && acol<n)?a[y*n+acol]:0.0f;
        tile_b[ty][tx]=(brow<n && x<n)?b[brow*n+x]:0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(uint k=0;k<16;++k)acc+=tile_a[ty][k]*tile_b[k][tx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(x<n && y<n)output[y*n+x]=acc;
}
)OPENCL";

void print_columns(){
    std::cout<<std::setw(21)<<"Function";
    for(const char* s:{"OpenMP","OpenCL total","OpenCL copy-in","OpenCL kernel","OpenCL copy-out"})
        std::cout<<std::setw(19)<<s;
    std::cout<<'\n';
}
void print_result(const char* name,time_point t0,time_point t1,time_point t2,time_point t3,time_point t4){
    using namespace std::chrono;
    std::cout<<std::setw(21)<<name;
    for(auto delta:{t1-t0,t4-t1,t2-t1,t3-t2,t4-t3}) {
        std::ostringstream s;s<<duration_cast<microseconds>(delta).count()<<"us";
        std::cout<<std::setw(19)<<s.str();
    }
    std::cout<<'\n';
}

void profile_vector_times_vector(int n,OpenCL& gpu){
    auto a=random_vector<float>(n),b=random_vector<float>(n);
    Vector<float> result(n),expected(n);
    const auto t0=clock_type::now();vector_times_vector(a,b,expected);
    const auto t1=clock_type::now();
    auto da=gpu.buffer(size_t(n)*sizeof(float)),db=gpu.buffer(size_t(n)*sizeof(float));
    auto dc=gpu.buffer(size_t(n)*sizeof(float));
    gpu.write(da,std::begin(a),size_t(n)*sizeof(float));
    gpu.write(db,std::begin(b),size_t(n)*sizeof(float));
    const auto t2=clock_type::now();
    auto k=gpu.kernel("vector_times_vector");
    const cl_uint cn=n;k.mem(0,da);k.mem(1,db);k.mem(2,dc);k.arg(3,cn);
    gpu.run(k,cl_round(n,256),256);gpu.finish();
    const auto t3=clock_type::now();
    gpu.read(dc,std::begin(result),size_t(n)*sizeof(float));
    const auto t4=clock_type::now();
    verify_vector(expected,result,1e-6f);
    print_result("vector-times-vector",t0,t1,t2,t3,t4);
}

void profile_matrix_times_vector(int n,OpenCL& gpu){
    auto a=random_matrix<float>(n,n);auto b=random_vector<float>(n);
    Vector<float> result(n),expected(n);
    const auto t0=clock_type::now();matrix_times_vector(a,b,expected);
    const auto t1=clock_type::now();
    const size_t cells=size_t(n)*n;
    auto da=gpu.buffer(cells*sizeof(float)),db=gpu.buffer(size_t(n)*sizeof(float));
    auto dc=gpu.buffer(size_t(n)*sizeof(float));
    gpu.write(da,std::begin(a),cells*sizeof(float));
    gpu.write(db,std::begin(b),size_t(n)*sizeof(float));
    const auto t2=clock_type::now();
    auto k=gpu.kernel("matrix_times_vector");
    const cl_uint cn=n;k.mem(0,da);k.mem(1,db);k.mem(2,dc);k.arg(3,cn);
    gpu.run(k,size_t(n)*256,256);gpu.finish();
    const auto t3=clock_type::now();
    gpu.read(dc,std::begin(result),size_t(n)*sizeof(float));
    const auto t4=clock_type::now();
    for(int i=0;i<n;++i)if(!std::isfinite(result(i)))throw std::runtime_error("GPU matvec non-finite result");
    verify_vector(expected,result,0.1f);
    print_result("matrix-times-vector",t0,t1,t2,t3,t4);
}

void profile_matrix_times_matrix(int n,OpenCL& gpu){
    auto a=random_matrix<float>(n,n),b=random_matrix<float>(n,n);
    Matrix<float> result(n,n),expected(n,n);
    const auto t0=clock_type::now();matrix_times_matrix(a,b,expected);
    const auto t1=clock_type::now();
    const size_t bytes=size_t(n)*n*sizeof(float);
    auto da=gpu.buffer(bytes),db=gpu.buffer(bytes),dc=gpu.buffer(bytes);
    gpu.write(da,std::begin(a),bytes);gpu.write(db,std::begin(b),bytes);
    const auto t2=clock_type::now();
    auto k=gpu.kernel("matrix_times_matrix");
    const cl_uint cn=n;k.mem(0,da);k.mem(1,db);k.mem(2,dc);k.arg(3,cn);
    gpu.run(k,cl_round(n,16),16,cl_round(n,16),16);gpu.finish();
    const auto t3=clock_type::now();
    gpu.read(dc,std::begin(result),bytes);
    const auto t4=clock_type::now();
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)if(!std::isfinite(result(i,j)))throw std::runtime_error("GPU matmul non-finite result");
    verify_matrix(expected,result,0.1f);
    print_result("matrix-times-matrix",t0,t1,t2,t3,t4);
}

int main(int argc,char** argv){
    try{
        bool test=false,quick=false;
        for(int i=1;i<argc;++i){
            if(std::string(argv[i])=="--test")test=true;
            else if(std::string(argv[i])=="--quick")quick=true;
            else throw std::runtime_error("Usage: super-boring-task [--quick | --test]");
        }
        OpenCL gpu(src);
        print_columns();
        if(test){
            for(int n:{1,3,17,64,129,257}) {
                std::cout<<"n="<<n<<'\n';
                profile_vector_times_vector(n,gpu);
                profile_matrix_times_vector(n,gpu);
                profile_matrix_times_matrix(n,gpu);
            }
        }else if(quick){
            profile_vector_times_vector(65536,gpu);
            profile_matrix_times_vector(512,gpu);
            profile_matrix_times_matrix(128,gpu);
        }else{
            profile_vector_times_vector(1024*1024*10,gpu);
            profile_matrix_times_vector(1024*10,gpu);
            profile_matrix_times_matrix(1024,gpu);
        }
        std::cout<<"Verification: PASS\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
