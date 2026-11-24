#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void entry(const float in[1][3][640][640], float c8[1][6400][1], float c16[1][1600][1], float c32[1][400][1], float o8[1][6400][1], float o16[1][1600][1], float o32[1][400][1], float b8[1][6400][4], float b16[1][1600][4], float b32[1][400][4], float k8[1][6400][10], float k16[1][1600][10], float k32[1][400][10]);
static float in[1][3][640][640];
static float c8[6400],c16[1600],c32[400],o8[6400],o16[1600],o32[400],b8[6400*4],b16[1600*4],b32[400*4],k8[6400*10],k16[1600*10],k32[400*10];
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb"); fread(in,4,3*640*640,f); fclose(f);
  int n = argc>3?atoi(argv[3]):1;
  struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
  for(int i=0;i<n;i++) entry(in,(void*)c8,(void*)c16,(void*)c32,(void*)o8,(void*)o16,(void*)o32,(void*)b8,(void*)b16,(void*)b32,(void*)k8,(void*)k16,(void*)k32);
  clock_gettime(CLOCK_MONOTONIC,&b);
  fprintf(stderr,"ms/iter %.2f\n",((b.tv_sec-a.tv_sec)*1e3+(b.tv_nsec-a.tv_nsec)/1e6)/n);
  f=fopen(argv[2],"wb");
  fwrite(c8,4,6400,f);fwrite(c16,4,1600,f);fwrite(c32,4,400,f);fwrite(o8,4,6400,f);fwrite(o16,4,1600,f);fwrite(o32,4,400,f);
  fwrite(b8,4,25600,f);fwrite(b16,4,6400,f);fwrite(b32,4,1600,f);fwrite(k8,4,64000,f);fwrite(k16,4,16000,f);fwrite(k32,4,4000,f);fclose(f);
}
