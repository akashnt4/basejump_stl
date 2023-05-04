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
 
  const int unroll = 16;

  for (int i = 0; i < len; i+=unroll) {

    float tempAi   = myA[i];
    float tempAi_1 = myA[i+1];
    float tempAi_2 = myA[i+2];
    float tempAi_3 = myA[i+3];
    float tempAi_4 = myA[i+4];
    float tempAi_5 = myA[i+5];
    float tempAi_6 = myA[i+6];
    float tempAi_7 = myA[i+7];
    float tempAi_8 = myA[i+8];
    float tempAi_9 = myA[i+9];
    float tempAi_10 = myA[i+10];
    float tempAi_11 = myA[i+11];
    float tempAi_12 = myA[i+12];
    float tempAi_13 = myA[i+13];
    float tempAi_14 = myA[i+14];
    float tempAi_15 = myA[i+15];

    float tempBi   = myB[i];
    float tempBi_1 = myB[i+1];
    float tempBi_2 = myB[i+2];
    float tempBi_3 = myB[i+3];
    float tempBi_4 = myB[i+4];
    float tempBi_5 = myB[i+5];
    float tempBi_6 = myB[i+6];
    float tempBi_7 = myB[i+7];
    float tempBi_8 = myB[i+8];
    float tempBi_9 = myB[i+9];
    float tempBi_10 = myB[i+10];
    float tempBi_11 = myB[i+11];
    float tempBi_12 = myB[i+12];
    float tempBi_13 = myB[i+13];
    float tempBi_14 = myB[i+14];
    float tempBi_15 = myB[i+15];

    float tempCi   = tempAi   + tempBi;
    float tempCi_1 = tempAi_1 + tempBi_1;
    float tempCi_2 = tempAi_2 + tempBi_2;
    float tempCi_3 = tempAi_3 + tempBi_3;
    float tempCi_4 = tempAi_4 + tempBi_4;
    float tempCi_5 = tempAi_5 + tempBi_5;
    float tempCi_6 = tempAi_6 + tempBi_6;
    float tempCi_7 = tempAi_7 + tempBi_7;
    float tempCi_8 = tempAi_8 + tempBi_8;
    float tempCi_9 = tempAi_9 + tempBi_9;
    float tempCi_10 = tempAi_10 + tempBi_10;
    float tempCi_11 = tempAi_11 + tempBi_11;
    float tempCi_12 = tempAi_12 + tempBi_12;
    float tempCi_13 = tempAi_13 + tempBi_13;
    float tempCi_14 = tempAi_14 + tempBi_14;
    float tempCi_15 = tempAi_15 + tempBi_15;

    myC[i]   = tempCi;
    myC[i+1] = tempCi_1;
    myC[i+2] = tempCi_2;
    myC[i+3] = tempCi_3;
    myC[i+4] = tempCi_4;
    myC[i+5] = tempCi_5;
    myC[i+6] = tempCi_6;
    myC[i+7] = tempCi_7;
    myC[i+8] = tempCi_8;
    myC[i+9] = tempCi_9;
    myC[i+10] = tempCi_10;
    myC[i+11] = tempCi_11;
    myC[i+12] = tempCi_12;
    myC[i+13] = tempCi_13;
    myC[i+14] = tempCi_14;
    myC[i+15] = tempCi_15;


  }

  bsg_fence();
  bsg_cuda_print_stat_kernel_end();
  bsg_fence();
  bsg_barrier_hw_tile_group_sync();

  return 0;
}
