#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <omp.h>
#include "sph.hh"
#include "gpu_sph.hh"
#ifdef HAVE_GUI
#include "opengl.hh"
#ifdef HAVE_THEORA
#include "theora.hh"
#endif
#endif

using Clock=std::chrono::steady_clock;
using Seconds=std::chrono::duration<double>;
struct Options {
    std::string mode="cpu",output="out";
    size_t count=340;
    int frames=120;
    bool headless=false,limited=false;
};
struct Errors {double position_rms=0,position_max=0,relative_max=0;};
void check_state(const std::vector<Particle>& p){
    for(const auto& a:p){
        for(int k=0;k<2;++k){
            if(!std::isfinite(a.position[k]) || !std::isfinite(a.velocity[k]) || !std::isfinite(a.force[k]))
                throw std::runtime_error("Non-finite particle state");
            if(a.position[k]<16 || a.position[k]>(k==0?784:584))
                throw std::runtime_error("Particle outside boundaries");
        }
        if(!std::isfinite(a.density) || a.density<=0 || !std::isfinite(a.pressure))
            throw std::runtime_error("Invalid density/pressure");
    }
}
Errors compare(const std::vector<Particle>& a,const std::vector<Particle>& b){
    if(a.size()!=b.size())throw std::runtime_error("Particle count mismatch");
    check_state(a);check_state(b);Errors e;
    for(size_t i=0;i<a.size();++i){
        for(int k=0;k<2;++k){
            const double d=std::abs(double(a[i].position[k])-b[i].position[k]);
            e.position_rms+=d*d;e.position_max=std::max(e.position_max,d);
        }
        const float x[]={a[i].velocity[0],a[i].velocity[1],a[i].force[0],a[i].force[1],a[i].density,a[i].pressure};
        const float y[]={b[i].velocity[0],b[i].velocity[1],b[i].force[0],b[i].force[1],b[i].density,b[i].pressure};
        for(int k=0;k<6;++k)e.relative_max=std::max(e.relative_max,
            std::abs(double(x[k])-y[k])/std::max(1.0,std::abs(double(x[k]))));
    }
    if(!a.empty())e.position_rms=std::sqrt(e.position_rms/(2*a.size()));
    if(e.position_max>0.1 || e.relative_max>0.005)
        throw std::runtime_error("CPU/GPU mismatch: position max="+std::to_string(e.position_max)+", relative max="+std::to_string(e.relative_max));
    return e;
}
void snapshot(const std::vector<Particle>& p,const std::string& filename){
    std::vector<unsigned char> rgb(size_t(domain_width)*domain_height*3,230);
    for(const auto& a:p){
        if(!std::isfinite(a.position[0]) || !std::isfinite(a.position[1]))throw std::runtime_error("Cannot render non-finite particle");
        const int cx=static_cast<int>(std::lround(a.position[0]));
        const int cy=domain_height-1-static_cast<int>(std::lround(a.position[1]));
        for(int y=std::max(0,cy-8);y<=std::min(domain_height-1,cy+8);++y)
            for(int x=std::max(0,cx-8);x<=std::min(domain_width-1,cx+8);++x){
                if((x-cx)*(x-cx)+(y-cy)*(y-cy)>64)continue;
                size_t index=(size_t(y)*domain_width+x)*3;
                rgb[index]=51;rgb[index+1]=153;rgb[index+2]=255;
            }
    }
    std::ofstream out(filename,std::ios::binary);
    out<<"P6\n"<<domain_width<<' '<<domain_height<<"\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()),rgb.size());
    if(!out)throw std::runtime_error("Cannot write "+filename);
}
template<class F> double timed(F&& f){const auto t=Clock::now();f();return Seconds(Clock::now()-t).count();}
double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
void tests(){
    GpuSph gpu;
    // Independent one-particle oracle: self-density and gravity-only motion.
    std::vector<Particle> one(1);one[0].position=vec2(400.f,300.f);
    gpu.reset(one);gpu.step(one);
    const float self_density=particle_mass*poly6*(256.f*256.f*256.f);
    if(std::abs(one[0].density-self_density)>1e-6f ||
       std::abs(one[0].velocity[1]-time_step*gravity)>1e-4f ||
       std::abs(one[0].position[1]-(300.f+time_step*time_step*gravity))>1e-4f)
        throw std::runtime_error("Single-particle gravity/density oracle failed");
    for(size_t n:{size_t(0),size_t(1),size_t(17),size_t(127),size_t(128),size_t(129),size_t(340),size_t(1024)}){
        auto cpu=generate_particles(n),device=cpu;gpu.reset(device);Errors worst;
        for(int frame=0;frame<120;++frame){
            step_cpu(cpu);gpu.step(device);const auto e=compare(cpu,device);
            worst.position_max=std::max(worst.position_max,e.position_max);
            worst.relative_max=std::max(worst.relative_max,e.relative_max);
        }
        std::cout<<"n="<<n<<" steps=120 max position error="<<worst.position_max<<" relative="<<worst.relative_max<<'\n';
    }
    // Coincident particles, nonzero velocities and wall collisions.
    auto cpu=generate_particles(4);
    cpu[0].position=vec2(16.f,16.f);cpu[1].position=cpu[0].position;
    cpu[2].position=vec2(784.f,584.f);cpu[3].position=vec2(400.f,300.f);
    cpu[0].velocity=vec2(-500.f,-500.f);cpu[2].velocity=vec2(500.f,500.f);
    auto device=cpu;gpu.reset(device);
    for(int f=0;f<10;++f){step_cpu(cpu);gpu.step(device);compare(cpu,device);}
    std::cout<<"Verification: PASS\n";
}
void benchmark(int frames){
    GpuSph gpu;
    std::ofstream csv("benchmark.csv"),md("benchmark.md");
    csv<<"particles,frames,openmp_seconds,opencl_seconds,speedup,position_rms,position_max\n";
    md<<"# SPH benchmark\n\nOpenMP max threads: "<<omp_get_max_threads()<<". Three warm-up frames, then "<<frames
      <<" frames; median wall-clock duration. GPU includes all three kernels and blocking particle download. "
      <<"Initial upload, allocation, compilation and rendering excluded.\n\n"
      <<"| Particles | OpenMP Frame duration, ms | OpenCL Frame duration, ms | CPU/GPU | Max position error |\n|---:|---:|---:|---:|---:|\n";
    std::cout<<"Particles   OpenMP Frame duration (s)   OpenCL Frame duration (s)   Speedup\n";
    for(size_t n:{size_t(340),size_t(1024),size_t(2048),size_t(4096),size_t(8192)}){
        auto cpu=generate_particles(n),device=cpu;gpu.reset(device);
        std::vector<double> ct,gt;Errors worst;
        for(int frame=-3;frame<frames;++frame){
            double c,g;
            if(frame%2){g=timed([&]{gpu.step(device);});c=timed([&]{step_cpu(cpu);});}
            else {c=timed([&]{step_cpu(cpu);});g=timed([&]{gpu.step(device);});}
            auto e=compare(cpu,device);
            worst.position_rms=std::max(worst.position_rms,e.position_rms);worst.position_max=std::max(worst.position_max,e.position_max);
            if(frame>=0){ct.push_back(c);gt.push_back(g);}
        }
        const double c=median(ct),g=median(gt);
        std::cout<<std::setw(9)<<n<<std::setw(28)<<c<<std::setw(28)<<g<<std::setw(12)<<c/g<<'\n';
        csv<<n<<','<<frames<<','<<c<<','<<g<<','<<c/g<<','<<worst.position_rms<<','<<worst.position_max<<'\n';
        md<<"| "<<n<<" | "<<1000*c<<" | "<<1000*g<<" | "<<c/g<<" | "<<worst.position_max<<" |\n";
    }
    if(!csv || !md)throw std::runtime_error("Cannot write benchmark results");
    // Compare a visible evolved state using the original 340-particle scene.
    auto cpu=generate_particles(340),device=cpu;gpu.reset(device);
    for(int frame=0;frame<120;++frame){step_cpu(cpu);gpu.step(device);compare(cpu,device);}
    snapshot(cpu,"cpu.ppm");snapshot(device,"gpu.ppm");
    std::cout<<"Verification: PASS\n";
}
#ifdef HAVE_GUI
std::vector<Particle> interactive_particles;
std::unique_ptr<GpuSph> interactive_gpu;
int interactive_frames=-1;
std::string interactive_output;
#ifdef HAVE_THEORA
std::unique_ptr<thx::screen_recorder> interactive_recorder;
#endif
void display(){
    draw_particles(interactive_particles);
#ifdef HAVE_THEORA
    glReadBuffer(GL_FRONT);interactive_recorder->record_frame();
#endif
    if(interactive_frames==0){snapshot(interactive_particles,interactive_output+".ppm");std::exit(0);}
}
void idle(){
    const double dt=timed([]{if(interactive_gpu)interactive_gpu->step(interactive_particles);else step_cpu(interactive_particles);});
    if(interactive_frames>0)--interactive_frames;
    std::cout<<std::setw(20)<<dt<<std::setw(20)<<1/dt<<'\n';glutPostRedisplay();
}
void keyboard(unsigned char key,int,int){
    if(key==27)std::exit(0);
    if(key=='r' || key=='R' || key==' '){
        if(key==' '){auto added=generate_particles(340,43);interactive_particles.insert(interactive_particles.end(),added.begin(),added.end());}
        else interactive_particles=generate_particles(340);
        if(interactive_gpu)interactive_gpu->reset(interactive_particles);
    }
}
void reshape(int w,int h){glViewport(0,0,w,h);}
void gui(const Options& o,int argc,char** argv){
    interactive_frames=o.limited?o.frames:-1;interactive_output=o.output;
    interactive_particles=generate_particles(o.count);
    if(o.mode=="gpu"){interactive_gpu=std::make_unique<GpuSph>();interactive_gpu->reset(interactive_particles);}
    // Pass only the executable name: application options are not GLUT options.
    argc=1;glutInit(&argc,argv);glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA);
    glutInitWindowSize(domain_width,domain_height);glutCreateWindow("SPH: CPU / OpenCL");
#ifdef HAVE_THEORA
    interactive_recorder=std::make_unique<thx::screen_recorder>(o.output+".ogv",domain_width,domain_height);
#endif
    glutDisplayFunc(display);glutIdleFunc(idle);glutKeyboardFunc(keyboard);glutReshapeFunc(reshape);
    std::cout<<"No. of particles: "<<o.count<<"\n      Frame duration   Frames per second\n";
    glutMainLoop();
}
#endif
int main(int argc,char** argv){
    try {
        Options o;
        for(int i=1;i<argc;++i){
            std::string arg=argv[i];
            if(arg=="cpu" || arg=="gpu" || arg=="--test" || arg=="--benchmark")o.mode=arg;
            else if(arg=="--headless")o.headless=true;
            else if(arg=="--output" && i+1<argc)o.output=argv[++i];
            else if((arg=="--particles" || arg=="--frames") && i+1<argc){
                std::string value=argv[++i];size_t end;long long n=std::stoll(value,&end);
                if(end!=value.size() || n<=0 || n>std::numeric_limits<int>::max())throw std::runtime_error("Invalid positive integer for "+arg);
                if(arg=="--particles")o.count=size_t(n);else {o.frames=int(n);o.limited=true;}
            }else throw std::runtime_error("Usage: sph [cpu|gpu|--test|--benchmark] [--headless --particles N --frames N --output PREFIX]");
        }
        std::cout<<"OpenMP max threads: "<<omp_get_max_threads()<<'\n';
        if(o.mode=="--test"){tests();return 0;}
        if(o.mode=="--benchmark"){benchmark(o.frames);return 0;}
#ifdef HAVE_GUI
        if(!o.headless){gui(o,argc,argv);return 0;}
#endif
        auto particles=generate_particles(o.count);std::unique_ptr<GpuSph> gpu;
        if(o.mode=="gpu"){gpu=std::make_unique<GpuSph>();gpu->reset(particles);}
        std::cout<<"No. of particles: "<<o.count<<"\n      Frame duration   Frames per second\n";
        for(int frame=0;frame<o.frames;++frame){
            const double dt=timed([&]{if(gpu)gpu->step(particles);else step_cpu(particles);});
            check_state(particles);std::cout<<std::setw(20)<<dt<<std::setw(20)<<(dt?1/dt:0)<<'\n';
        }
        snapshot(particles,o.output+".ppm");
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
