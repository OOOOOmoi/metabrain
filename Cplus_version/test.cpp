#include <iostream>
#include <time.h>
#include <pthread.h>
using namespace std;
void *func1(void *var1){
  pthread_exit(NULL);
}
int main(){
  pthread_t tid;
  for (int i = 0; i < 10000; i++)
  {
    pthread_create(&tid,NULL,func1,NULL);
    pthread_join(tid);
  }
  return 0;
  
}