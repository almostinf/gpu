#pragma once
#include "gpu_opencl.hh"
#include "sph_kernel.hh"

class GpuSph {
    OpenCL cl_{sph_source()};
    CLKernel density_=cl_.kernel("compute_density_and_pressure");
    CLKernel forces_=cl_.kernel("compute_forces");
    CLKernel positions_=cl_.kernel("compute_positions");
    CLBuffer particles_;
    size_t size_=0;
public:
    void reset(const std::vector<Particle>& p){
        size_=p.size();particles_=cl_.buffer(size_*sizeof(Particle));
        cl_.write(particles_,p.data(),size_*sizeof(Particle));
        const cl_uint count=static_cast<cl_uint>(size_);
        for(CLKernel* k:{&density_,&forces_,&positions_}){k->mem(0,particles_);k->arg(1,count);}
    }
    // In-order queue gives a global dependency between the three stages.
    // Particles stay on the GPU between frames; no CPU force/density updates.
    void step(std::vector<Particle>& host){
        if(host.size()!=size_)throw std::runtime_error("Particle count changed without reset");
        if(size_==0)return;
        cl_.run(density_,cl_round(size_,128),128);
        cl_.run(forces_,cl_round(size_,128),128);
        cl_.run(positions_,cl_round(size_,128),128);
        // Blocking download is included in Frame duration and supplies display data.
        cl_.read(particles_,host.data(),size_*sizeof(Particle));
    }
};
