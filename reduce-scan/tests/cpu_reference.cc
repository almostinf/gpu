#include "../reduce-scan.hh"
#include <iostream>
#include <stdexcept>
int main(){
    Vector<int> v(6);const int values[]={3,-2,7,0,-5,4};
    for(int i=0;i<6;++i)v(i)=values[i];
    if(reduce(v)!=7)throw std::runtime_error("reduce");
    Vector<int> incl(v);scan_inclusive(incl);
    const int expected[]={3,1,8,8,3,7};
    for(int i=0;i<6;++i)if(incl(i)!=expected[i])throw std::runtime_error("inclusive scan");
    Vector<int> excl(v);scan_exclusive(excl);
    const int exclusive[]={0,3,1,8,8,3};
    for(int i=0;i<6;++i)if(excl(i)!=exclusive[i])throw std::runtime_error("exclusive scan");
    std::cout<<"CPU reference PASS (reduce, inclusive/exclusive scan)\n";
}
