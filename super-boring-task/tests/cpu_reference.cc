#include "../linear-algebra.hh"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main(){
    Matrix<float> a(2,3),b(3,2),c(2,2);Vector<float> v(3),out(2);
    const float aa[]={1,2,3,4,5,6},bb[]={7,8,9,10,11,12};
    for(int i=0;i<2;++i)for(int j=0;j<3;++j)a(i,j)=aa[i*3+j];
    for(int i=0;i<3;++i)for(int j=0;j<2;++j)b(i,j)=bb[i*2+j];
    for(int i=0;i<3;++i)v(i)=1;
    matrix_times_vector(a,v,out);
    if(out(0)!=6 || out(1)!=15)throw std::runtime_error("matvec CPU reference");
    matrix_times_matrix(a,b,c);
    if(c(0,0)!=58 || c(0,1)!=64 || c(1,0)!=139 || c(1,1)!=154)throw std::runtime_error("matmul CPU reference");
    auto rect=random_matrix<float>(3,5);
    for(int i=0;i<3;++i)for(int j=0;j<5;++j)
        if(!(rect(i,j)>=0 && rect(i,j)<=1))throw std::runtime_error("random_matrix rectangular");
    std::cout<<"CPU reference PASS (matvec, matmul, rectangular dimensions)\n";
}
