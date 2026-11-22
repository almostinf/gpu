#pragma once
static const char* TRACE_SOURCE=R"CL(
#pragma OPENCL FP_CONTRACT OFF
uint seed_pixel(uint pixel,uint frame){
    uint s=pixel*747796405u+frame*2891336453u+277803737u;
    s=(s^(s>>16))*2246822519u;s^=s>>13;
    return s?s:1u;
}
float uniform_random(uint* s){
    *s^=*s<<13;*s^=*s>>17;*s^=*s<<5;
    return (float)(*s>>8)*(1.0f/16777216.0f);
}
float dot3(float3 a,float3 b){
    float r=0;r+=a.x*b.x;r+=a.y*b.y;r+=a.z*b.z;return r;
}
float3 random_sphere(uint* state){
    float4 x;float q;
    do {
        x.x=2*uniform_random(state)-1;x.y=2*uniform_random(state)-1;
        x.z=2*uniform_random(state)-1;x.w=2*uniform_random(state)-1;
        q=0;q+=x.x*x.x;q+=x.y*x.y;q+=x.z*x.z;q+=x.w*x.w;
    }while(q>=1 || q==0);
    return (float3)(2*(x.y*x.w+x.x*x.z),2*(x.z*x.w-x.x*x.y),
                    x.x*x.x+x.w*x.w-x.y*x.y-x.z*x.z)/q;
}
float sphere_hit(float3 origin,float3 direction,float3 center,float radius,float closest){
    float3 oc=origin-center;
    float a=dot3(direction,direction),b=dot3(oc,direction);
    float c=dot3(oc,oc)-radius*radius;
    float discriminant=b*b-a*c;
    if(discriminant>0){
        float d=sqrt(discriminant),t=(-b-d)/a;
        if(t>1e-3f && t<closest)return t;
        t=(-b+d)/a;
        if(t>1e-3f && t<closest)return t;
    }
    return closest;
}
float3 trace_path(float3 origin,float3 direction,uint* state){
    float factor=1;
    for(int depth=0;depth<50;++depth){
        float closest=MAXFLOAT;
        float t=sphere_hit(origin,direction,(float3)(0,0,-1),0.5f,closest);
        int object=-1;
        if(t<closest){closest=t;object=0;}
        t=sphere_hit(origin,direction,(float3)(0,-1000.5f,-1),1000.0f,closest);
        if(t<closest){closest=t;object=1;}
        if(object<0)break;
        float3 center=object?(float3)(0,-1000.5f,-1):(float3)(0,0,-1);
        float radius=object?1000.0f:0.5f;
        origin=origin+closest*direction;
        direction=(origin-center)/radius+random_sphere(state);
        factor*=0.5f;
    }
    float len=sqrt(dot3(direction,direction));
    float t=0.5f*((len==0?direction.y:direction.y/len)+1.0f);
    return factor*((1-t)*(float3)(1,1,1)+t*(float3)(0.5f,0.7f,1));
}
// One work-item per pixel; every sample and bounce is traced on the GPU.
// Packed RGB uses vstore3 rather than the padded host/device float3 ABI.
__kernel void ray_tracing_gpu(__global float* pixels,uint width,uint height,
                             uint samples,uint frame,float camera_z){
    uint id=get_global_id(0);
    if(id>=width*height)return;
    uint x=id%width,y=id/width,state=seed_pixel(id,frame);
    float3 sum=(float3)(0);
    float3 origin=(float3)(13,2,camera_z);
    for(uint k=0;k<samples;++k){
        float u=((float)x+uniform_random(&state))/(float)width;
        float v=((float)y+uniform_random(&state))/(float)height;
        float3 direction=(float3)(3.02374f,-1.22628f,3.4122f)
            +u*(float3)(1.18946f,0,-5.15434f)
            +v*(float3)(-0.509421f,3.48757f,-0.117559f)-origin;
        sum+=trace_path(origin,direction,&state);
    }
    vstore3(sqrt(sum/(float)samples),id,pixels);
}
)CL";
