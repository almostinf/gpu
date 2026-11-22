#include "../filter.hh"
#include <iostream>
#include <stdexcept>
int main(){
    std::vector<float> in={-2,0,3,-1,5,7,0},out;
    filter(in,out,[](float x){return x>0;});
    if(out!=std::vector<float>({3,5,7}))throw std::runtime_error("stable filter");
    std::vector<float> empty,result;
    filter(empty,result,[](float x){return x>0;});
    if(!result.empty())throw std::runtime_error("empty filter");
    std::cout<<"CPU reference PASS (stable positive filter, empty)\n";
}
