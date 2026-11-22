#pragma once
#include <iomanip>
#include <sstream>
#include "sph.hh"

inline std::string sph_source(){
    std::ostringstream s;
    s<<std::scientific<<std::setprecision(9);
    s<<"#define POLY6 "<<poly6<<"f\n#define SPIKY "<<spiky_grad
     <<"f\n#define VISC_LAP "<<visc_laplacian<<"f\n";
    s<<R"CL(
#pragma OPENCL FP_CONTRACT OFF
typedef struct {float2 position,velocity,force;float density,pressure;} Particle;
float square2(float2 x){float r=0;r+=x.x*x.x;r+=x.y*x.y;return r;}

// All lanes, including padding, participate in every tile barrier.
__kernel void compute_density_and_pressure(__global Particle* p,uint n){
    uint i=get_global_id(0),lane=get_local_id(0);
    float2 pos=i<n?p[i].position:(float2)(0);
    float sum=0;
    __local float2 tile[128];
    for(uint base=0;base<n;base+=128){
        tile[lane]=base+lane<n?p[base+lane].position:(float2)(0);
        barrier(CLK_LOCAL_MEM_FENCE);
        if(i<n)for(uint j=0;j<min((uint)128,n-base);++j){
            float sd=square2(tile[j]-pos);
            if(sd<256){float q=256-sd;sum+=65.0f*POLY6*(q*q*q);}
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(i<n){p[i].density=sum;p[i].pressure=2000.0f*(sum-1000.0f);}
}
__kernel void compute_forces(__global Particle* p,uint n){
    uint i=get_global_id(0),lane=get_local_id(0);
    float2 pos=i<n?p[i].position:(float2)(0),vel=i<n?p[i].velocity:(float2)(0);
    float ap=i<n?p[i].pressure:0;
    float2 pressure=(float2)(0),viscosity=(float2)(0);
    __local float2 positions[128];
    __local float4 state[128];
    for(uint base=0;base<n;base+=128){
        uint j=base+lane;
        positions[lane]=j<n?p[j].position:(float2)(0);
        state[lane]=j<n?(float4)(p[j].velocity,p[j].density,p[j].pressure):(float4)(0);
        barrier(CLK_LOCAL_MEM_FENCE);
        if(i<n)for(uint k=0;k<min((uint)128,n-base);++k){
            if(base+k==i)continue;
            float2 delta=positions[k]-pos;
            float sd=square2(delta);
            if(sd<256){
                float r=sqrt(sd),q=16-r;
                float2 direction=r==0?delta:delta/r;
                pressure+=-direction*65.0f*(ap+state[k].w)/(2.0f*state[k].z)*SPIKY*(q*q);
                viscosity+=250.0f*65.0f*(state[k].xy-vel)/state[k].z*VISC_LAP*q;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(i<n)p[i].force=pressure+viscosity+(float2)(0,-117600.0f)*p[i].density;
}
__kernel void compute_positions(__global Particle* p,uint n){
    uint i=get_global_id(0);if(i>=n)return;
    float2 velocity=p[i].velocity+0.0008f*p[i].force/p[i].density;
    float2 position=p[i].position+0.0008f*velocity;
    for(uint axis=0;axis<2;++axis){
        float bound=axis==0?800:600;
        if(position[axis]-16<0){velocity[axis]*=-0.5f;position[axis]=16;}
        if(position[axis]+16>bound){velocity[axis]*=-0.5f;position[axis]=bound-16;}
    }
    p[i].position=position;p[i].velocity=velocity;
}

)CL";
    return s.str();
}
