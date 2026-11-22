#ifndef RANDOM_HH
#define RANDOM_HH
#include <cstdint>
#include "scene.hh"

// Independent deterministic stream per pixel and frame, identical on CPU/GPU.
struct Random {
    uint32_t state;
    Random(uint32_t pixel,uint32_t frame) {
        state=pixel*747796405u+frame*2891336453u+277803737u;
        state=(state^(state>>16))*2246822519u;
        state^=state>>13;
        if(!state)state=1;
    }
    float uniform() {
        state^=state<<13;state^=state>>17;state^=state<<5;
        return float(state>>8)*(1.0f/16777216.0f);
    }
};
inline vec random_in_unit_sphere(Random& rng) {
    Vector<float,4> x;
    float q;
    do {
        for(int i=0;i<4;++i)x[i]=2.0f*rng.uniform()-1.0f;
        q=square(x);
    } while(q>=1.0f || q==0.0f);
    return vec{2*(x[1]*x[3]+x[0]*x[2]),
               2*(x[2]*x[3]-x[0]*x[1]),
               x[0]*x[0]+x[3]*x[3]-x[1]*x[1]-x[2]*x[2]}/q;
}
#endif
