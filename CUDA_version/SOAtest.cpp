#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
struct soa{
    vector<float> a;
    vector<float> b;
};

struct soa2{
    float* c;
    float* d;
};
int main(){
    soa2 s;
    cout<<sizeof(s)<<endl;
    for (int i = 0; i < 10; i++)
    {
        s.c[i]=i;
    }
    cout<<s.c[1]<<endl;
    cout<<sizeof(s)<<endl;
    return 0;
}