#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <type_traits>
#include <vector>
#include "vector.hh"

using vec2=Vector<float,2>;
constexpr float kernel_radius=16,particle_mass=65,pi=3.14159265358979323846f;
constexpr float poly6=315.f/(65.f*pi*68719476736.f); // h^9
constexpr float spiky_grad=-45.f/(pi*16777216.f); // h^6
constexpr float visc_laplacian=45.f/(pi*16777216.f);
constexpr float gas_const=2000,rest_density=1000,visc_const=250;
constexpr float gravity=-117600,time_step=0.0008f;
constexpr int domain_width=800,domain_height=600;

struct Particle {
    vec2 position,velocity,force;
    float density=0,pressure=0;
};
static_assert(std::is_standard_layout<Particle>::value && sizeof(Particle)==32,
              "Particle must match eight packed OpenCL floats");
static_assert(offsetof(Particle,velocity)==8 && offsetof(Particle,force)==16 &&
              offsetof(Particle,density)==24 && offsetof(Particle,pressure)==28,
              "Particle OpenCL offsets must match");

inline std::vector<Particle> generate_particles(size_t n,unsigned seed=42){
    std::vector<Particle> p(n);
    std::mt19937 rng(seed);std::uniform_real_distribution<float> jitter(-1,1);
    const size_t cols=n?static_cast<size_t>(std::ceil(std::sqrt(double(n)*400/480))):1;
    const size_t rows=n?(n+cols-1)/cols:1;
    const float dx=n==340?24.f:400.f/cols,dy=n==340?24.f:480.f/rows;
    for(size_t i=0;i<n;++i){
        p[i].position=vec2(200.f+float(i/rows)*dx+jitter(rng),120.f+float(i%rows)*dy+jitter(rng));
    }
    return p;
}
inline void compute_density_and_pressure(std::vector<Particle>& p){
    const int n=static_cast<int>(p.size());
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n;++i){
        float sum=0;
        for(int j=0;j<n;++j){
            const float sd=square(p[j].position-p[i].position);
            if(sd<256.f){const float q=256.f-sd;sum+=particle_mass*poly6*(q*q*q);}
        }
        p[i].density=sum;p[i].pressure=gas_const*(sum-rest_density);
    }
}
inline void compute_forces(std::vector<Particle>& p){
    const int n=static_cast<int>(p.size());
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n;++i){
        vec2 pressure(0.f,0.f),viscosity(0.f,0.f);
        for(int j=0;j<n;++j){
            if(i==j)continue;
            const vec2 delta=p[j].position-p[i].position;
            const float sd=square(delta);
            if(sd<256.f){
                const float r=std::sqrt(sd),q=kernel_radius-r;
                const vec2 direction=r==0?delta:delta/r;
                pressure+=-direction*particle_mass*(p[i].pressure+p[j].pressure)
                    /(2.f*p[j].density)*spiky_grad*(q*q);
                viscosity+=visc_const*particle_mass*(p[j].velocity-p[i].velocity)
                    /p[j].density*visc_laplacian*q;
            }
        }
        p[i].force=pressure+viscosity+vec2(0.f,gravity)*p[i].density;
    }
}
inline void compute_positions(std::vector<Particle>& p){
    const int n=static_cast<int>(p.size());
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n;++i){
        auto& a=p[i];
        a.velocity+=time_step*a.force/a.density;
        a.position+=time_step*a.velocity;
        for(int axis=0;axis<2;++axis){
            const float bound=axis==0?domain_width:domain_height;
            if(a.position[axis]-kernel_radius<0){a.velocity[axis]*=-0.5f;a.position[axis]=kernel_radius;}
            if(a.position[axis]+kernel_radius>bound){a.velocity[axis]*=-0.5f;a.position[axis]=bound-kernel_radius;}
        }
    }
}
inline void step_cpu(std::vector<Particle>& p){
    compute_density_and_pressure(p);compute_forces(p);compute_positions(p);
}
