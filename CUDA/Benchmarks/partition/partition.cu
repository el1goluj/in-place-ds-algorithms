/***************************************************************************
 *cr
 *cr            (C) Copyright 2015 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/*
  In-Place Data Sliding Algorithms for Many-Core Architectures, presented in ICPP’15

  Copyright (c) 2015 University of Illinois at Urbana-Champaign. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Authors: Juan Gómez-Luna (el1goluj@uco.es, gomezlun@illinois.edu), Li-Wen Chang (lchang20@illinois.edu)
*/

#include "../../DS/ds.h"

// Sample predicate for partition (only for INT)
struct is_even{
  __host__ __device__
  bool operator()(const T &x){
    return (x % 2) == 0;
  }
};

#include "kernel.cu"

// Sequential CPU version
void cpu_partition(T* input, int elements, struct is_even pred){
  T* aux = (T*)malloc(sizeof(T)*elements);
  int pos1 = 0;
  int pos2 = 0;
  for (int i = 0; i < elements; i++){
    if(pred(input[i])){
	  input[pos1] = input[i];
      pos1++;
    }
    else{
      aux[pos2] = input[i];
      pos2++;
    }
  }
  for (int i = 0; i < pos2; i++){
	input[pos1 + i] = aux[i];
  }
}

int main(int argc, char **argv){

  // Syntax verification
  if (argc != 4) {
      printf("Wrong format\n");
      printf("Syntax: %s <Device Input (%% elements) numElements>\n",argv[0]);
      exit(1);
  }
  int device = atoi(argv[1]);
  int input = atoi(argv[2]);
  int numElements = atoi(argv[3]);
  size_t size = numElements * sizeof(T);

  // Set device
  cudaDeviceProp device_properties;
  cudaGetDeviceProperties(&device_properties,device);
  cudaSetDevice(device);

  printf("DS Partition on %s\n", device_properties.name);
  printf("Thread block size = %d\n", L_DIM);
  printf("Coarsening factor = %d\n", REGS);
#ifdef FLOAT
  printf("Single precision array: %d elements\n", numElements);
#elif INT
  printf("Integer array: %d elements\n", numElements);
#else
  printf("Double precision array: %d elements\n", numElements);
#endif

  // Event creation
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float time1 = 0;
  float time2 = 0;

  // Allocate the host input vector A
  T *h_A = (T*)malloc(size);

  // Allocate the host output vector
  T *h_B = (T*)malloc(size);

  // Allocate the device input vector A and auxiliary vector B
  T *d_A = NULL;
  cudaMalloc((void **)&d_A, size);
  T *d_B = NULL;
  cudaMalloc((void **)&d_B, size);

#define WARMUP 2
#define REP 10
  for(int iteration = 0; iteration < REP+WARMUP; iteration++){
    // Initialize the host input vectors
    srand(2014);
    for(int i = 0; i < numElements; i++)
    	h_A[i] = i % 2 != 0 ? i:i+1;
    int M = (numElements * input)/100;
    int m = M;
    while(m>0){
        int x = (int)(numElements*(((float)rand()/(float)RAND_MAX)));
        if(h_A[x] % 2 != 0){
    	    h_A[x] = x * 2;
            m--;
        }
    }

#if PRINT
    printf("\n");
    for(int i = 0; i < numElements; ++i){
        printf("%d ",*(h_A+i));
    }
    printf("\n");
#endif

    // Copy the host input vector A in host memory to the device input vector in device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    int ldim = L_DIM;
    // Atomic flags
    unsigned int* d_flags1 = NULL;
    unsigned int* d_flags2 = NULL;
    const int num_flags = numElements % (ldim * REGS) == 0 ? numElements / (ldim * REGS) : numElements / (ldim * REGS) + 1;
    unsigned int *flags = (unsigned int *)calloc(sizeof(unsigned int), num_flags + 2);
    flags[0] = 1;
    flags[num_flags + 1] = 0;
    cudaMalloc((void **)&d_flags1, (num_flags + 2) * sizeof(unsigned int));
    cudaMemcpy(d_flags1, flags, (num_flags + 2) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_flags2, (num_flags + 2) * sizeof(unsigned int));
    cudaMemcpy(d_flags2, flags, (num_flags + 2) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    free(flags);
    // Number of work-groups/thread blocks
    int num_wg = num_flags;

    // Start timer
    cudaEventRecord( start, 0 );

    // Kernel launch
    partition<<<num_wg, ldim>>>(d_A, d_B, d_A, numElements, d_flags1, d_flags2, is_even());

    unsigned int flagM = 0;
    cudaMemcpy(&flagM, d_flags1 + num_flags, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    num_wg = (numElements - flagM + 1) == 0 ? 1 : (numElements - flagM + 1) % ldim == 0 ? (numElements - flagM + 1) / ldim : (numElements - flagM + 1) / ldim + 1;
    cudaDeviceSynchronize();
    if((numElements - flagM + 1) != 0)
      move_part<<<num_wg, ldim>>>(d_A, d_B, flagM - 1, numElements);
      //cudaMemcpy(&d_A[flagM - 1], d_B, (numElements - flagM + 1)*sizeof(int), cudaMemcpyDeviceToDevice);

    // End timer
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time1, start, stop );
    if(iteration >= WARMUP) time2 += time1;

    if(iteration == REP+WARMUP-1){
      float timer = time2 / REP;
      double bw = (double)((2 * numElements) * sizeof(T)) / (double)(timer * 1000000.0);
      printf("Execution time = %f ms, Throughput = %f GB/s\n", timer, bw);
    }

    // Free flags
    cudaFree(d_flags1);
    cudaFree(d_flags2);
  }
  // Copy to host memory
  cudaMemcpy(h_B, d_A, size, cudaMemcpyDeviceToHost);

  // CPU execution for comparison
  cpu_partition(h_A, numElements, is_even());

  // Verify that the result vector is correct
#if PRINT
  for(int i = 0; i < numElements; ++i){
     printf("%d ",*(h_B+i));
  }
  printf("\n");
  for(int i = 0; i < numElements; ++i){
      printf("%d ",*(h_A+i));
  }
  printf("\n");
#endif
  for (int i = 0; i < numElements; ++i){
      if (h_B[i] != h_A[i]){
          fprintf(stderr, "Result verification failed at element %d!\n", i);
          exit(EXIT_FAILURE);
      }
  }
  printf("Test PASSED\n");

  // Free device global memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // Free host memory
  free(h_A);
  free(h_B);

  return 0;
}

