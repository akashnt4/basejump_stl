#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>

#ifdef WARM_CACHE
__attribute__((noinline))
static void warmup(float *A, float *B, float *C, int N)
{
  for (int i = __bsg_id*CACHE_LINE_WORDS; i < N; i += bsg_tiles_X*bsg_tiles_Y*CACHE_LINE_WORDS) {
      asm volatile ("lw x0, %[p]" :: [p] "m" (A[i]));
      asm volatile ("lw x0, %[p]" :: [p] "m" (B[i]));
      asm volatile ("sw x0, %[p]" :: [p] "m" (C[i]));
  }
  bsg_fence();
}
#endif


// Vector-Add: C = A + B
// N = vector size
extern "C" __attribute__ ((noinline))
int
kernel_vector_add(float * A, float * B, float *C, int N) {

  bsg_barrier_hw_tile_group_init();
#ifdef WARM_CACHE
  warmup(A, B, C, N);
#endif
  bsg_barrier_hw_tile_group_sync();
  bsg_cuda_print_stat_kernel_start();

  // Each tile does a portion of vector_add
  int len = N / (bsg_tiles_X*bsg_tiles_Y);
  float *myA = &A[__bsg_id*len];
  float *myB = &B[__bsg_id*len];
  float *myC = &C[__bsg_id*len];
 
  //float tempA[8], tempB[8], tempC[8];
  const int unroll = 8;

  for (int i = 0; i < len; i+=unroll) {
    //float tempA[unroll], tempB[unroll], tempC[unroll];

    //float tempA, tempB, tempC;

    //tempA = myA[i];
    //bsg_unroll(unroll)    
    //for(int j=0 ; j<unroll; j++) {
    //    tempA[j] = myA[i+j];
    //}
    float tempAi   = myA[i];
    float tempAi_1 = myA[i+1];
    float tempAi_2 = myA[i+2];
    float tempAi_3 = myA[i+3];
    float tempAi_4 = myA[i+4];
    float tempAi_5 = myA[i+5];
    float tempAi_6 = myA[i+6];
    float tempAi_7 = myA[i+7];

    //tempB = myB[i];
    //bsg_unroll(unroll)
    //for(int j=0; j<unroll; j++) {
    //    tempB[j] = myB[i+j];
    //}
    float tempBi   = myB[i];
    float tempBi_1 = myB[i+1];
    float tempBi_2 = myB[i+2];
    float tempBi_3 = myB[i+3];
    float tempBi_4 = myB[i+4];
    float tempBi_5 = myB[i+5];
    float tempBi_6 = myB[i+6];
    float tempBi_7 = myB[i+7];

    //tempC = tempA + tempB;
    
    //bsg_unroll(unroll)
    //for(int j=0; j<unroll; j++) {
    //    tempC[j] = tempA[i+j] + tempB[i+j];
    //}
    float tempCi   = tempAi   + tempBi;
    float tempCi_1 = tempAi_1 + tempBi_1;
    float tempCi_2 = tempAi_2 + tempBi_2;
    float tempCi_3 = tempAi_3 + tempBi_3;
    float tempCi_4 = tempAi_4 + tempBi_4;
    float tempCi_5 = tempAi_5 + tempBi_5;
    float tempCi_6 = tempAi_6 + tempBi_6;
    float tempCi_7 = tempAi_7 + tempBi_7;

    //myC[i] = tempC;

    //bsg_unroll(unroll)
    //for(int j=0; j<unroll; j++) {
    //    myC[i+j] = tempC[j];
    //}
    myC[i]   = tempCi;
    myC[i+1] = tempCi_1;
    myC[i+2] = tempCi_2;
    myC[i+3] = tempCi_3;
    myC[i+4] = tempCi_4;
    myC[i+5] = tempCi_5;
    myC[i+6] = tempCi_6;
    myC[i+7] = tempCi_7;


//myC[i] = myA[i] + myB[i];
  }

  bsg_fence();
  bsg_cuda_print_stat_kernel_end();
  bsg_fence();
  bsg_barrier_hw_tile_group_sync();

  return 0;
}
