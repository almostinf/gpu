#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <omp.h>

#include "color.hh"
#include "scene.hh"
#include "random.hh"
#include "gpu_opencl.hh"
#include "trace_kernel.hh"
#ifdef HAVE_THEORA
#include "theora.hh"
#endif

using Image=Pixel_matrix<float>;
using Clock=std::chrono::steady_clock;
using Milliseconds=std::chrono::duration<double,std::milli>;
struct Options {
    std::string mode="cpu",output="out";
    int width=600,height=400,samples=100,frames=60,repeats=3;
};

vec trace(ray r,const Object_group& objects,Random& rng){
    float factor=1;
    for(int depth=0;depth<50;++depth){
        if(Hit hit=objects.hit(r,1e-3f,std::numeric_limits<float>::max())){
            r=ray(hit.point,hit.normal+random_in_unit_sphere(rng));
            factor*=0.5f;
        }else break;
    }
    float t=0.5f*(unit(r.direction())[1]+1.0f);
    return factor*((1-t)*vec(1.f,1.f,1.f)+t*vec(0.5f,0.7f,1.f));
}

Object_group make_scene(){
    Object_group objects;
    objects.add(std::make_unique<Sphere>(vec(0.f,0.f,-1.f),0.5f));
    objects.add(std::make_unique<Sphere>(vec(0.f,-1000.5f,-1.f),1000.f));
    return objects;
}

double ray_tracing_cpu(Image& pixels,int samples,int frame,const Object_group& scene){
    const int width=pixels.ncols(),height=pixels.nrows();
    Camera camera;camera.move(vec(0.f,0.f,0.1f*frame));
    const auto start=Clock::now();
    #pragma omp parallel for collapse(2) schedule(static)
    for(int j=0;j<height;++j){
        for(int i=0;i<width;++i){
            Random rng(static_cast<uint32_t>(j*width+i),static_cast<uint32_t>(frame));
            vec sum;
            for(int k=0;k<samples;++k){
                const float u=(i+rng.uniform())/width;
                const float v=(j+rng.uniform())/height;
                sum+=trace(camera.make_ray(u,v),scene,rng);
            }
            sum/=float(samples);
            for(float& value:sum)value=std::sqrt(value);
            pixels(i,j)=to_color(sum);
        }
    }
    return Milliseconds(Clock::now()-start).count();
}

class GpuRenderer {
    OpenCL gpu_{TRACE_SOURCE};
    CLKernel kernel_=gpu_.kernel("ray_tracing_gpu");
    CLBuffer output_;
    size_t capacity_=0;
public:
    void prepare(size_t count){
        if(count!=capacity_){output_=gpu_.buffer(count*3*sizeof(float));capacity_=count;}
    }
    double ray_tracing_gpu(Image& pixels,int samples,int frame){
        const size_t count=size_t(pixels.ncols())*pixels.nrows();
        prepare(count);
        static_assert(sizeof(Color<float>)==3*sizeof(float),"RGB must be packed");
        const auto start=Clock::now();
        const cl_uint w=pixels.ncols(),h=pixels.nrows(),s=samples,f=frame;
        const float camera_z=3.0f+0.1f*frame;
        kernel_.mem(0,output_);kernel_.arg(1,w);kernel_.arg(2,h);
        kernel_.arg(3,s);kernel_.arg(4,f);kernel_.arg(5,camera_z);
        gpu_.run(kernel_,cl_round(count,256),256);
        // Blocking read waits for the kernel and includes the whole frame download.
        gpu_.read(output_,pixels.pixels().data(),count*3*sizeof(float));
        return Milliseconds(Clock::now()-start).count();
    }
};

struct Difference {double rmse=0,mae=0,max_error=0;};
Difference compare(const Image& cpu,const Image& gpu){
    if(cpu.nrows()!=gpu.nrows() || cpu.ncols()!=gpu.ncols())throw std::runtime_error("Image size mismatch");
    Difference d;size_t channels=0;
    for(size_t i=0;i<cpu.pixels().size();++i){
        const auto a=cpu.pixels()[i],b=gpu.pixels()[i];
        const float ca[]={a.r,a.g,a.b},cb[]={b.r,b.g,b.b};
        for(int c=0;c<3;++c){
            if(!std::isfinite(ca[c]) || !std::isfinite(cb[c]) || ca[c]<0 || cb[c]<0 || ca[c]>1 || cb[c]>1)
                throw std::runtime_error("Invalid pixel value");
            const double error=std::abs(double(ca[c])-cb[c]);
            d.rmse+=error*error;d.mae+=error;d.max_error=std::max(d.max_error,error);++channels;
        }
    }
    d.rmse=std::sqrt(d.rmse/channels);d.mae/=channels;
    // Stochastic paths can diverge after numerically borderline intersections.
    // Compare gamma-corrected RGB in [0,1], not exact floating-point equality.
    if(d.rmse>0.03 || d.mae>0.015)throw std::runtime_error("CPU/GPU images differ: RMSE="+std::to_string(d.rmse));
    return d;
}
void save(const std::string& filename,const Image& image){
    std::ofstream out(filename);out<<image;
    if(!out)throw std::runtime_error("Cannot write "+filename);
}
double median(std::vector<double> values){
    std::sort(values.begin(),values.end());
    return values[values.size()/2];
}

void benchmark(const Options& o){
    GpuRenderer renderer;
    auto scene=make_scene();
    std::ofstream csv("benchmark.csv"),report("benchmark.md");
    csv<<"width,height,pixels,samples,repeats,openmp_ms,opencl_ms,speedup,rmse,mae,max_error\n";
    report<<"# Ray tracing: CPU / GPU\n\nSamples/pixel: "<<o.samples<<". OpenMP threads (maximum): "<<omp_get_max_threads()
          <<". One warm-up and "<<o.repeats<<" measured runs per size; median wall time.\n\n"
          <<"OpenCL includes kernel submission, all ray tracing, synchronization and RGB download. "
          <<"Context/program/buffer setup and PPM/video encoding are excluded from both timings.\n\n"
          <<"| Image | OpenMP time, ms | OpenCL time, ms | CPU/GPU | RMSE |\n|---|---:|---:|---:|---:|\n";
    std::cout<<"Image       OpenMP time (ms)  OpenCL time (ms)  Speedup  RMSE\n";
    bool faster=true;
    for(auto dims:{std::pair<int,int>{150,100},{300,200},{600,400},{900,600}}){
        const int w=dims.first,h=dims.second;
        Image cpu(w,h),gpu(w,h);renderer.prepare(size_t(w)*h);
        ray_tracing_cpu(cpu,o.samples,0,scene);renderer.ray_tracing_gpu(gpu,o.samples,0);
        compare(cpu,gpu);
        std::vector<double> ct,gt;
        Difference d;
        for(int run=0;run<o.repeats;++run){
            // Alternate execution order to reduce systematic thermal/order bias.
            if(run%2){gt.push_back(renderer.ray_tracing_gpu(gpu,o.samples,0));ct.push_back(ray_tracing_cpu(cpu,o.samples,0,scene));}
            else {ct.push_back(ray_tracing_cpu(cpu,o.samples,0,scene));gt.push_back(renderer.ray_tracing_gpu(gpu,o.samples,0));}
            d=compare(cpu,gpu);
        }
        const double c=median(ct),g=median(gt);faster&=g<c;
        const std::string size=std::to_string(w)+"x"+std::to_string(h);
        std::cout<<std::left<<std::setw(12)<<size<<std::right<<std::setw(15)<<c<<std::setw(18)<<g<<std::setw(10)<<c/g<<"  "<<d.rmse<<'\n';
        csv<<w<<','<<h<<','<<w*h<<','<<o.samples<<','<<o.repeats<<','<<c<<','<<g<<','<<c/g<<','<<d.rmse<<','<<d.mae<<','<<d.max_error<<'\n';
        report<<"| "<<size<<" | "<<c<<" | "<<g<<" | "<<c/g<<" | "<<d.rmse<<" |\n";
        if(w==600){save("cpu.ppm",cpu);save("gpu.ppm",gpu);}
    }
    if(!csv || !report)throw std::runtime_error("Cannot write benchmark results");
    std::cout<<"Verification: PASS\nOpenCL faster at every measured size: "<<(faster?"YES":"NO")<<'\n';
}
void tests(){
    GpuRenderer renderer;auto scene=make_scene();
    for(auto dims:{std::pair<int,int>{1,1},{17,13},{64,48},{127,91}}){
        for(int frame:{0,7,59}){
            Image cpu(dims.first,dims.second),gpu(dims.first,dims.second);
            ray_tracing_cpu(cpu,100,frame,scene);renderer.ray_tracing_gpu(gpu,100,frame);
            const auto d=compare(cpu,gpu);
            std::cout<<dims.first<<'x'<<dims.second<<" frame="<<frame<<" RMSE="<<d.rmse<<" MAE="<<d.mae<<'\n';
        }
    }
    std::cout<<"Verification: PASS\n";
}
void render(const Options& o){
    auto scene=make_scene();Image pixels(o.width,o.height);
    std::unique_ptr<GpuRenderer> gpu;
    if(o.mode=="gpu"){gpu=std::make_unique<GpuRenderer>();gpu->prepare(size_t(o.width)*o.height);}
#ifdef HAVE_THEORA
    thx::screen_recorder recorder(o.output+".ogv",o.width,o.height);
#else
    std::clog<<"Theora unavailable: saving PPM only.\n";
#endif
    std::cout<<"Time step   No. of steps   "<<(gpu?"OpenCL":"OpenMP")<<" time (ms)\n";
    double total=0;
    for(int frame=0;frame<o.frames;++frame){
        const double ms=gpu?gpu->ray_tracing_gpu(pixels,o.samples,frame):ray_tracing_cpu(pixels,o.samples,frame,scene);
        total+=ms;std::cout<<frame+1<<"           "<<o.frames<<"             "<<ms<<'\n';
#ifdef HAVE_THEORA
        recorder.record_frame(pixels);
#endif
    }
    save(o.output+".ppm",pixels);
    std::cout<<"Ray-tracing time: "<<total/1000<<"s\n";
}
int main(int argc,char** argv){
    try {
        Options o;
        for(int i=1;i<argc;++i){
            const std::string arg=argv[i];
            if(arg=="gpu" || arg=="cpu")o.mode=arg;
            else if(arg=="--benchmark" || arg=="--test")o.mode=arg;
            else if(arg=="--output" && i+1<argc)o.output=argv[++i];
            else if((arg=="--width" || arg=="--height" || arg=="--samples" || arg=="--frames" || arg=="--repeats") && i+1<argc){
                const std::string text=argv[++i];size_t end=0;const int v=std::stoi(text,&end);
                if(end!=text.size() || v<=0)throw std::runtime_error("Expected positive integer for "+arg);
                if(arg=="--width")o.width=v;else if(arg=="--height")o.height=v;
                else if(arg=="--samples")o.samples=v;else if(arg=="--frames")o.frames=v;else o.repeats=v;
            }else throw std::runtime_error("Usage: ray-tracing [cpu|gpu|--benchmark|--test] [--width N --height N --samples N --frames N --repeats N --output PREFIX]");
        }
        if(int64_t(o.width)*o.height>std::numeric_limits<int>::max())throw std::runtime_error("Image too large");
        std::cout<<"OpenMP max threads: "<<omp_get_max_threads()<<'\n';
        if(o.mode=="--benchmark")benchmark(o);else if(o.mode=="--test")tests();else render(o);
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
