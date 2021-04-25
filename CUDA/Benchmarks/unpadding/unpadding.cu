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
#include "kernel.cu"

// Sequential CPU version
double cpu_unpadding(T* matrix, int x_size, int y_size, int pad_size){
  struct timeval t1, t2;
  // start timer
  gettimeofday(&t1, NULL);

  for (int my_y = 1; my_y < y_size; my_y++){
    for (int my_x = 0; my_x < x_size; my_x++){
      matrix[my_y * x_size + my_x] = matrix[my_y * pad_size + my_x];
    }
  }
  for (int i = y_size * x_size; i < y_size * pad_size; i++)
    matrix[i] = 0.0f;

  // end timer
  gettimeofday(&t2, NULL);
  double timer = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_usec - t1.tv_usec);
  // compute bandwidth
  double bandwidth = 2*x_size * y_size * sizeof(T) / (timer * 1000.0);
  return bandwidth;
}

// GPU version
double gpu_unpadding(T* matrix, int x_size, int y_size, int pad_size){
  if (pad_size > y_size || x_size > pad_size){
    printf("dim error\n");
    exit(1);
  }

  float time_kernel;
  // Create CUDA event handles
  cudaEvent_t start_event,stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);

  // Allocate matrix
  T* d_matrix = NULL;
  cudaMalloc((void **)&d_matrix, pad_size * y_size * sizeof(T));
  cudaMemcpy(d_matrix, matrix, pad_size * y_size * sizeof(T), cudaMemcpyHostToDevice);

  int ldim = L_DIM;
  // Atomic flags
  const int num_flags = (y_size * pad_size) / (ldim * REGS);
  unsigned int *flags = (unsigned int *)calloc(sizeof(unsigned int), num_flags + 2);
  flags[0] = 1;
  flags[num_flags + 1] = 0;
  unsigned int* d_flags = NULL;
  cudaMalloc((void **)&d_flags, (num_flags + 2) * sizeof(unsigned int));
  cudaMemcpy(d_flags, flags, (num_flags + 2) * sizeof(unsigned int), cudaMemcpyHostToDevice);
  free(flags);

  // Number of work-groups/thread blocks
  int num_wg = num_flags + 1;

  // Start timer
  cudaEventRecord(start_event,0);

  // Kernel launch 
  unpadding<<<num_wg, ldim>>>(d_matrix, x_size, pad_size, y_size, d_flags);

  // End timer
  cudaEventRecord(stop_event,0);
  cudaEventSynchronize(stop_event);
  cudaEventElapsedTime(&time_kernel,start_event,stop_event);

  // Compute bandwidth
  double bandwidth = 2*x_size * y_size * sizeof(float) / (time_kernel * 1000000.0);

  cudaMemcpy(matrix, d_matrix, pad_size * y_size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_matrix);
  cudaFree(d_flags);
  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);

  return bandwidth;
}

#define TEST_CPU 0
int bench(int m, int n, int pad) {
  T* M = (T*) malloc (m * (n+pad) * sizeof(T) );
  T* M2 = (T*) malloc (m * (n+pad) * sizeof(T) );

  // Generate input matrix
  srand(time(NULL));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n+pad; j++) {
      if (j < n){
        M[i * (n+pad) + j] = (T)rand() / RAND_MAX;
        M2[i * (n+pad) + j] = M[i * (n+pad) + j];
      }
      else{
        M[i * (n+pad) + j] = 0.0f;
        M2[i * (n+pad) + j] = M[i * (n+pad) + j];
      }
    }
  }

  T* N = (T*) malloc (m * (n+pad) * sizeof(T));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (j < n) {
        N[i * n + j] = M[i * (n+pad) + j];
      }
    }
  }
  for (int i = m * n; i < m * (n+pad); i++)
    N[i] = 0.0f;

#if PRINT
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n+pad; j++){
      printf("%.0f ", M[i * (n+pad) + j]);
      if (j == n+pad-1) printf("\n");
    }
  printf("\n");

  for (int i = 0; i < m+((m*(n+pad)-m*n)/n); i++)
    for (int j = 0; j < n; j++){
      printf("%.0f ", N[i * n + j]);
      if (j == n-1) printf("\n");
    }
#endif

#if TEST_CPU
  // CPU execution
  double bw_cpu = cpu_unpadding(M2, n, m, n+pad);

#if PRINT
  for (int i = 0; i < m+((m*(n+pad)-m*n)/n); i++)
    for (int j = 0; j < n; j++){
      printf("%.0f ", M2[i * n + j]);
      if (j == n-1) printf("\n");
    }
#endif
#endif

  // GPU execution
  double bw = gpu_unpadding(M, n, m, n+pad);

#if PRINT
  printf("\n");
  for (int i = 0; i < m+((m*(n+pad)-m*n)/n); i++)
    for (int j = 0; j < n; j++){
      printf("%.0f ", M[i * n + j]);
      if (j == n-1) printf("\n");
    }
#endif

  // Check results
  for (int i = 0; i < m*(n+pad); i++){
    if( N[i] != M[i] ) {
        std::cerr << "Error. i = " << i/n << " j = " << i%n << std::endl;
        return 1;
    }
#if TEST_CPU
    if( N[i] != M2[i] ) {
      std::cerr << "Error-CPU. i = " << i << "\t" << N[i] << "\t" << M2[i] <<std::endl;
      return 1;
    }
#endif
  }
#if TEST_CPU
  printf("%d\t%d\t%d\t%.2f\t\t%.2f\n", m, n+pad, n, bw, bw_cpu);
#else
  printf("%d\t%d\t%d\t%.2f\n", m, n+pad, n, bw);
#endif
  free(M);
  free(M2);
  free(N);
  return 0;
}

int main(const int argc, const char* argv[]) {
  // Check the compute capability of the device
  int num_devices=0;
  cudaGetDeviceCount(&num_devices);
  if(0==num_devices){
    printf("Your system does not have a CUDA capable device\n");
    return 1;
  }
  cudaDeviceProp device_properties;
  cudaGetDeviceProperties(&device_properties,0);
  if((1==device_properties.major)&&(device_properties.minor<1))
  printf("%s does not have compute capability 1.1 or later\n\n",device_properties.name);

  // Syntax verification
  if (argc != 2) {
    printf("Wrong format\n");
    printf("Syntax: %s <Device>\n",argv[0]);
    exit(1);
  }

  int device = atoi(argv[1]);
  // Set device
  cudaGetDeviceProperties(&device_properties,device);
  cudaSetDevice(device);
  printf("DS Unpadding on %s\n", device_properties.name);
  printf("Thread block size = %d\n", L_DIM);
  printf("Coarsening factor = %d\n", REGS);
#ifdef FLOAT
  printf("Single precision array\n");
#elif INT
  printf("Integer array\n");
#else
  printf("Double precision array\n");
#endif

#if TEST_CPU
  printf("m\tn+pad\tn\tGPU (GB/s)\tCPU (GB/s)\n");
#else
  printf("m\tn+pad\tn\tGPU (GB/s)\n");
#endif

  int pad = 2;
  for(int m=1000;m<13000; m+=1000)
    for(int j=m-pad;j<m; j+=1){
      bench(m, j, m-j);
    }
}
 
