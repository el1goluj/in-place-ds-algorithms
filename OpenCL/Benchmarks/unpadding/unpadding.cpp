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

#include "../../DS/ds_host.h"

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
double gpu_unpadding(cl_device_id device, cl_context clContext, T* matrix, int x_size, int y_size, int pad_size){
  if (pad_size > y_size || x_size > pad_size){
    printf("dim error\n");
    exit(1);
  }

  struct timeval t1, t2;

  cl_int clStatus;
  // Create command queue
  cl_command_queue clCommandQueue = clCreateCommandQueue(clContext, device, 0, &clStatus);
  CL_ERR();

  std::filebuf clFile;
  clFile.open("kernel.cl",std::ios::in);
  std::istream in(&clFile);
  std::string clCode(std::istreambuf_iterator<char>(in), (std::istreambuf_iterator<char>()));

  // Create program
  const char* clSource[] = {clCode.c_str()};
  cl_program clProgram = clCreateProgramWithSource(clContext, 1, clSource, NULL, &clStatus);
  CL_ERR();

  char clOptions[100];
#ifdef FLOAT
  sprintf(clOptions,"-I. -D FLOAT -D COARSENING=%d -D THREADS=%d -D ATOMIC=%d", REGS, L_DIM, ATOM);
#elif INT
  sprintf(clOptions,"-I. -D INT -D COARSENING=%d -D THREADS=%d -D ATOMIC=%d", REGS, L_DIM, ATOM);
#else
  sprintf(clOptions,"-I. -D DOUBLE -D COARSENING=%d -D THREADS=%d -D ATOMIC=%d", REGS, L_DIM, ATOM);
#endif

  // Build program
  clStatus = clBuildProgram(clProgram, 0, NULL, clOptions, NULL, NULL);
  CL_ERR();

  // Create kernel
  cl_kernel clKernel = clCreateKernel(clProgram,"unpadding", &clStatus);
  CL_ERR();

  // Allocate matrix
  cl_mem d_matrix = clCreateBuffer(clContext, CL_MEM_READ_WRITE, pad_size * y_size * sizeof(T), NULL, &clStatus);
  clEnqueueWriteBuffer(clCommandQueue, d_matrix, CL_TRUE, 0, sizeof(T) * pad_size * y_size, matrix, 0, NULL, NULL);
  clFinish(clCommandQueue);
  CL_ERR();

  int ldim = L_DIM;
  // Atomic flags
  const int num_flags = (y_size * pad_size) / (ldim * REGS);
  unsigned int *flags = (unsigned int *)calloc(sizeof(unsigned int), num_flags + 2);
  flags[0] = 1;
  flags[num_flags + 1] = 0;
  cl_mem d_flags = clCreateBuffer(clContext, CL_MEM_READ_WRITE, (num_flags + 2) * sizeof(unsigned int), NULL, &clStatus);
  clEnqueueWriteBuffer(clCommandQueue, d_flags, CL_TRUE, 0, sizeof(unsigned int) * (num_flags + 2), flags, 0, NULL, NULL);
  clFinish(clCommandQueue);
  free(flags);
  CL_ERR();

  // Number of work-groups/thread blocks
  int num_wg = num_flags + 1;

  // Set kernel arguments
  clSetKernelArg(clKernel, 0, sizeof(cl_mem), &d_matrix);
  clSetKernelArg(clKernel, 1, sizeof(int), &x_size);
  clSetKernelArg(clKernel, 2, sizeof(int), &pad_size);
  clSetKernelArg(clKernel, 3, sizeof(int), &y_size);
  clSetKernelArg(clKernel, 4, sizeof(cl_mem), &d_flags);

  // Start timer
  gettimeofday(&t1, NULL);

  // Kernel launch 
  size_t ls[1] = {ldim};
  size_t gs[1] = {ldim * num_wg};
  clStatus = clEnqueueNDRangeKernel(clCommandQueue, clKernel, 1, NULL, gs, ls, 0, NULL, NULL);
  CL_ERR();
  clFinish(clCommandQueue);

  // End timer
  gettimeofday(&t2, NULL);
  double time_kernel = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_usec - t1.tv_usec);

  // Compute bandwidth
  double bandwidth = 2*x_size * y_size * sizeof(T) / (time_kernel * 1000.0);

  clStatus = clEnqueueReadBuffer(clCommandQueue, d_matrix, CL_TRUE, 0, pad_size * y_size * sizeof(T), matrix, 0, NULL, NULL);
  CL_ERR();
  clFinish(clCommandQueue);
  CL_ERR();
  clReleaseMemObject(d_flags);
  clReleaseMemObject(d_matrix);
  clReleaseKernel(clKernel);
  clReleaseProgram(clProgram);
  clReleaseCommandQueue(clCommandQueue);

  return bandwidth;
}

#define TEST_CPU 1
int bench(cl_device_id device, cl_context clContext, int m, int n, int pad) {
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
  double bw = gpu_unpadding(device, clContext, M, n, m, n+pad);

#if PRINT
  printf("\n");
  for (int i = 0; i < m+((m*(n+pad)-m*n)/n); i++)
    for (int j = 0; j < n; j++){
      printf("%.0f ", M[i * n + j]);
      if (j == n-1) printf("\n");
    }
#endif

  // Check results
  for (int i = 0; i < m*n; i++){
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
  // Syntax verification
  if (argc != 3) {
    printf("Wrong format\n");
    printf("Syntax: %s <Platform Device>\n",argv[0]);
    exit(1);
  }

  int platform = atoi(argv[1]);
  int device = atoi(argv[2]);

  // Select device and create clContext
  cl_int clStatus;
  cl_uint clNumPlatforms;
  clStatus = clGetPlatformIDs(0, NULL, &clNumPlatforms);
  CL_ERR();
  cl_platform_id* clPlatforms = new cl_platform_id[clNumPlatforms];
  clStatus = clGetPlatformIDs(clNumPlatforms, clPlatforms, NULL);
  CL_ERR();
  char clPlatformVendor[128];
  char clPlatformVersion[128];
  cl_platform_id clPlatform;
  char clVendorName[128];
  for(int i=0; i<clNumPlatforms; i++) {
    clStatus = clGetPlatformInfo(clPlatforms[i], CL_PLATFORM_VENDOR, 128*sizeof(char), clPlatformVendor, NULL);
    CL_ERR();
    std::string clVendorName(clPlatformVendor);
    if(clVendorName.find(clVendorName) != std::string::npos) {
      clPlatform = clPlatforms[i];
      if(platform == i) break;
    }
  }
  delete[] clPlatforms;			
  cl_uint clNumDevices;
  clStatus = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_ALL, 0, NULL, &clNumDevices);
  CL_ERR();
  cl_device_id* clDevices = new cl_device_id[clNumDevices];
  clStatus = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_ALL, clNumDevices, clDevices, NULL);
  CL_ERR();
  cl_context clContext = clCreateContext(NULL, clNumDevices, clDevices, NULL, NULL, &clStatus);
  CL_ERR();
  char device_name_[100];
  clGetDeviceInfo(clDevices[device], CL_DEVICE_NAME, 100, &device_name_, NULL);

  printf("DS Unpadding on %s\n", device_name_);
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
  printf("m\tn\tn+pad\tGPU (GB/s)\tCPU (GB/s)\n");
#else
  printf("m\tn\tn+pad\tGPU (GB/s)\n");
#endif

  int pad = 2;
  for(int m=1000;m<13000; m+=1000)
    for(int j=m-pad;j<m; j+=1){
      bench(clDevices[device], clContext, m, j, m-j);
    }

  // Release clContext
  clReleaseContext(clContext);
}    
