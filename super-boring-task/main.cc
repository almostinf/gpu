#include "gpu_opencl.hh"
#include "linear-algebra.hh"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <new>
#include <random>
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

// Vectorized matrix-vector product: 128 work-items per row, four adjacent
// matrix elements processed per work-item at a time. Works for any n,
// including n not divisible by 4. Each workgroup reduces in local memory.
__kernel void matrix_times_vector_vec4(__global const float* a,
                                       __global const float* b,
                                       __global float* output,const uint n) {
    const uint row=get_group_id(0);
    const uint lane=get_local_id(0);
    const uint nvec=n/4;
    __global const float* matrix_row=a+row*n;
    float sum=0.0f;
    for(uint vec=lane;vec<nvec;vec+=128){
        const float4 av=vload4(vec,matrix_row);
        const float4 bv=vload4(vec,b);
        sum+=dot(av,bv);
    }
    for(uint col=4*nvec+lane;col<n;col+=128)
        sum+=matrix_row[col]*b[col];
    __local float partial[128];
    partial[lane]=sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint step=64;step>0;step>>=1){
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

// Host memory backed by a page-aligned allocation. On some unified-memory
// OpenCL implementations the alignment of the host pointer makes a large
// difference to whether CL_MEM_USE_HOST_PTR can be accessed efficiently.
// Alignment is an optimization experiment, not a zero-copy guarantee.
struct AlignedFloats {
    float* data=nullptr;
    explicit AlignedFloats(size_t count) {
        const size_t bytes=std::max<size_t>(count*sizeof(float),16384);
        void* ptr=nullptr;
        if(posix_memalign(&ptr,16384,bytes)!=0)throw std::bad_alloc();
        data=static_cast<float*>(ptr);
    }
    ~AlignedFloats(){std::free(data);}
    AlignedFloats(const AlignedFloats&)=delete;
    AlignedFloats& operator=(const AlignedFloats&)=delete;
};

void profile_matrix_times_vector_aligned(int n,OpenCL& gpu,
                                         const std::string& kernel_mode){
    const size_t matrix_cells=size_t(n)*n;
    const size_t matrix_bytes=matrix_cells*sizeof(float);
    const size_t vector_bytes=size_t(n)*sizeof(float);
    // Produce the inputs directly in page-aligned memory. Importantly, the
    // OpenMP and OpenCL computations read exactly the SAME original storage;
    // there is no unmeasured copy from another array to a GPU-only array.
    AlignedFloats a(matrix_cells), b(n);
    auto prng=make_prng();
    std::uniform_real_distribution<float> dist(0.0f,1.0f);
    for(size_t i=0;i<matrix_cells;++i)a.data[i]=dist(prng);
    for(int i=0;i<n;++i)b.data[i]=dist(prng);
    Vector<float> expected(n),result(n);
    const auto t0=clock_type::now();
    #pragma omp parallel for
    for(int row=0;row<n;++row){
        float sum=0.0f;
        #pragma omp simd reduction(+:sum)
        for(int col=0;col<n;++col)
            sum+=a.data[size_t(row)*n+col]*b.data[col];
        expected(row)=sum;
    }
    const auto t1=clock_type::now();
    auto da=gpu.host_read_buffer(a.data,matrix_bytes);
    auto db=gpu.host_read_buffer(b.data,vector_bytes);
    auto dc=gpu.buffer(vector_bytes);
    const auto t2=clock_type::now();
    const bool vectorized=kernel_mode=="vec4";
    auto k=gpu.kernel(vectorized?"matrix_times_vector_vec4":"matrix_times_vector");
    const cl_uint cn=n;
    k.mem(0,da);k.mem(1,db);k.mem(2,dc);k.arg(3,cn);
    const size_t local=vectorized?128:256;
    gpu.run(k,size_t(n)*local,local);
    gpu.finish();
    const auto t3=clock_type::now();
    gpu.read(dc,std::begin(result),vector_bytes);
    const auto t4=clock_type::now();
    for(int i=0;i<n;++i)
        if(!std::isfinite(result(i)))
            throw std::runtime_error("GPU aligned matvec non-finite result");
    verify_vector(expected,result,0.1f);
    print_result("matrix-times-vector",t0,t1,t2,t3,t4);
    using namespace std::chrono;
    std::clog<<"  matvec transfer=aligned, kernel="<<kernel_mode
             <<", buffer creation="<<duration_cast<microseconds>(t2-t1).count()<<"us"
             <<", explicit upload=0us, matrix="<<(matrix_bytes/(1024*1024))
             <<" MiB, input alignment=16384 bytes\n";
}

void profile_matrix_times_vector(int n,OpenCL& gpu,
                                 const std::string& transfer_mode,
                                 const std::string& kernel_mode){
    if(transfer_mode=="aligned") {
        profile_matrix_times_vector_aligned(n,gpu,kernel_mode);
        return;
    }
    auto a=random_matrix<float>(n,n);
    auto b=random_vector<float>(n);
    Vector<float> result(n),expected(n);
    const auto t0=clock_type::now();
    matrix_times_vector(a,b,expected);
    const auto t1=clock_type::now();
    const size_t cells=size_t(n)*n;
    const size_t matrix_bytes=cells*sizeof(float);
    const size_t vector_bytes=size_t(n)*sizeof(float);
    CLBuffer da, db, dc;
    if(transfer_mode=="hostptr"){
        // Note: USE_HOST_PTR may still trigger an implicit device-side copy.
        // Host pointers stay alive and unchanged until clFinish completes.
        da=gpu.host_read_buffer(std::begin(a),matrix_bytes);
        db=gpu.host_read_buffer(std::begin(b),vector_bytes);
        dc=gpu.buffer(vector_bytes);
    }else{
        da=gpu.buffer(matrix_bytes);
        db=gpu.buffer(vector_bytes);
        dc=gpu.buffer(vector_bytes);
    }
    const auto t_alloc=clock_type::now();
    if(transfer_mode=="explicit"){
        gpu.write(da,std::begin(a),matrix_bytes);
        gpu.write(db,std::begin(b),vector_bytes);
    }
    const auto t2=clock_type::now();
    const bool vectorized=kernel_mode=="vec4";
    auto k=gpu.kernel(vectorized?"matrix_times_vector_vec4":"matrix_times_vector");
    const cl_uint cn=n;
    k.mem(0,da);k.mem(1,db);k.mem(2,dc);k.arg(3,cn);
    const size_t local=vectorized?128:256;
    gpu.run(k,size_t(n)*local,local);
    gpu.finish();
    const auto t3=clock_type::now();
    gpu.read(dc,std::begin(result),vector_bytes);
    const auto t4=clock_type::now();
    for(int i=0;i<n;++i)
        if(!std::isfinite(result(i)))throw std::runtime_error("GPU matvec non-finite result");
    verify_vector(expected,result,0.1f);
    print_result("matrix-times-vector",t0,t1,t2,t3,t4);
    using namespace std::chrono;
    std::clog<<"  matvec transfer="<<transfer_mode
             <<", kernel="<<kernel_mode
             <<", buffer creation="<<duration_cast<microseconds>(t_alloc-t1).count()<<"us"
             <<", explicit upload="<<duration_cast<microseconds>(t2-t_alloc).count()<<"us"
             <<", matrix="<<(matrix_bytes/(1024*1024))<<" MiB\n";
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
        bool test=false,quick=false,matvec_only=false;
        std::string transfer="auto",kernel="vec4";
        for(int i=1;i<argc;++i){
            const std::string arg=argv[i];
            if(arg=="--test")test=true;
            else if(arg=="--quick")quick=true;
            else if(arg=="--matvec-only")matvec_only=true;
            else if(arg=="--matvec-transfer" && i+1<argc)transfer=argv[++i];
            else if(arg=="--matvec-kernel" && i+1<argc)kernel=argv[++i];
            else throw std::runtime_error("Usage: super-boring-task [--quick | --test] "
                "[--matvec-only] [--matvec-transfer auto|explicit|hostptr|aligned] "
                "[--matvec-kernel vec4|scalar]");
        }
        if(transfer!="auto" && transfer!="explicit" && transfer!="hostptr" && transfer!="aligned")
            throw std::runtime_error("Invalid transfer mode");
        if(kernel!="vec4" && kernel!="scalar")
            throw std::runtime_error("Invalid matvec kernel mode");
        OpenCL gpu(src);
        if(transfer=="auto")transfer=gpu.is_apple()?"hostptr":"explicit";
        print_columns();
        if(test){
            for(int n:{1,3,17,64,129,257,513}){
                std::cout<<"n="<<n<<'\n';
                if(!matvec_only)profile_vector_times_vector(n,gpu);
                profile_matrix_times_vector(n,gpu,transfer,kernel);
                if(!matvec_only)profile_matrix_times_matrix(n,gpu);
            }
        }else if(quick){
            if(!matvec_only)profile_vector_times_vector(65536,gpu);
            profile_matrix_times_vector(512,gpu,transfer,kernel);
            if(!matvec_only)profile_matrix_times_matrix(128,gpu);
        }else{
            if(!matvec_only)profile_vector_times_vector(1024*1024*10,gpu);
            profile_matrix_times_vector(1024*10,gpu,transfer,kernel);
            if(!matvec_only)profile_matrix_times_matrix(1024,gpu);
        }
        std::cout<<"Verification: PASS\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
