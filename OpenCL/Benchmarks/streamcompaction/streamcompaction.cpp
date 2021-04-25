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
void cpu_streamcompaction(T* input, int elements, T value){
  int j = 0;
  for (int i = 0; i < elements; i++){
    if (input[i] != value){
      input[j] = input[i];
      j++;		
    }
  }
}

// GPU version
double gpu_streamcompaction(cl_device_id device, cl_context clContext, T* matrix, int numElements, int value){

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
  char* opt = "";
#ifndef SEQUENTIAL
#if defined(SHFL) && defined(NVIDIA)
  opt = "-D NVIDIA -D SHFL";
#elif defined(NVIDIA)
  opt = "-D NVIDIA";
#elif defined(SHFL)
  opt = "-D SHFL";
#endif
#else
  opt = "-D SEQUENTIAL";
#endif
#ifdef FLOAT
  sprintf(clOptions,"-I. -D FLOAT -D COARSENING=%d -D THREADS=%d -D ATOMIC=%d %s", REGS, L_DIM, ATOM, opt);
#elif INT
  sprintf(clOptions,"-I. -D INT -D COARSENING=%d -D THREADS=%d -D ATOMIC=%d %s", REGS, L_DIM, ATOM, opt);
#else
  sprintf(clOptions,"-I. -D DOUBLE -D COARSENING=%d -D THREADS=%d -D ATOMIC=%d %s", REGS, L_DIM, ATOM, opt);
#endif
	//puts(clOptions);
	
  // Build program
  clStatus = clBuildProgram(clProgram, 0, NULL, clOptions, NULL, NULL);
  CL_ERR();

  // Create kernel
  cl_kernel clKernel = clCreateKernel(clProgram,"streamcompaction", &clStatus);
  CL_ERR();

  // Allocate matrix
  cl_mem d_matrix = clCreateBuffer(clContext, CL_MEM_READ_WRITE, numElements * sizeof(T), NULL, &clStatus);
  clEnqueueWriteBuffer(clCommandQueue, d_matrix, CL_TRUE, 0, sizeof(T) * numElements, matrix, 0, NULL, NULL);
  clFinish(clCommandQueue);
  CL_ERR();

  int ldim = L_DIM;
  // Atomic flags
  const int num_flags = numElements % (ldim * REGS) == 0 ? numElements / (ldim * REGS) : numElements / (ldim * REGS) + 1;
  unsigned int *flags = (unsigned int *)calloc(sizeof(unsigned int), num_flags + 2);
  flags[0] = 1;
  flags[num_flags + 1] = 0;
  cl_mem d_flags = clCreateBuffer(clContext, CL_MEM_READ_WRITE, (num_flags + 2) * sizeof(unsigned int), NULL, &clStatus);
  clEnqueueWriteBuffer(clCommandQueue, d_flags, CL_TRUE, 0, sizeof(unsigned int) * (num_flags + 2), flags, 0, NULL, NULL);
  clFinish(clCommandQueue);
  free(flags);
  CL_ERR();

  // Number of work-groups/thread blocks
  int num_wg = num_flags;
	
  // Set kernel arguments
  clSetKernelArg(clKernel, 0, sizeof(cl_mem), &d_matrix);
  clSetKernelArg(clKernel, 1, sizeof(cl_mem), &d_matrix);
  clSetKernelArg(clKernel, 2, L_DIM*sizeof(int), NULL);
  clSetKernelArg(clKernel, 3, L_DIM*sizeof(int), NULL);
  clSetKernelArg(clKernel, 4, sizeof(int), &numElements);
  clSetKernelArg(clKernel, 5, sizeof(cl_mem), &d_flags);
  clSetKernelArg(clKernel, 6, sizeof(int), &value);

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
  double bandwidth = 2 * numElements * sizeof(T) / (time_kernel * 1000.0);

  clStatus = clEnqueueReadBuffer(clCommandQueue, d_matrix, CL_TRUE, 0, numElements * sizeof(T), matrix, 0, NULL, NULL);
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

int bench(cl_device_id device, cl_context clContext, int numElements, int input, int value) {
  T* M = (T*) malloc (numElements * sizeof(T));
  T* M2 = (T*) malloc (numElements * sizeof(T));

  // Initialize the host input vectors
  srand(2014);
  for(int i = 0; i < numElements; i++)
   	M[i] = value;
  int Ma = (numElements * input)/100;
  int ma = Ma;
  while(ma>0){
    int x = (int)(numElements*(((float)rand()/(float)RAND_MAX)));
    if(M[x]==value){
      M[x] = x+2;
      ma--;
    }
  }

  for (int i = 0; i < numElements; i++) {
    M2[i] = M[i];
  }

#if PRINT
  for(int i = 0; i < numElements; ++i){
      printf("%d ",*(M+i));
  }
  printf("\n");
#endif

  // CPU execution
  cpu_streamcompaction(M2, numElements, value);

#if PRINT
  for (int i = 0; i < numElements; i++)
    printf("%d ", M2[i]);
  printf("\n");
#endif

  // GPU execution
  double bw = gpu_streamcompaction(device, clContext, M, numElements, value);

#if PRINT
  printf("\n");
  for (int i = 0; i < numElements; i++)
    printf("%d ", M[i]);
  printf("\n");
#endif

  // Check results
  for (int i = 0; i < Ma; ++i){
    if (M[i] != M2[i]){
      printf("Error i=%d\n", i);
      exit(1);
    }
  }

  printf("numElements = %d\tThroughput = %f GB/s\n", numElements, bw);
  printf("TEST PASSED\n");

  free(M);
  free(M2);
  return 0;
}

int main(const int argc, const char* argv[]) {

  // Syntax verification
  if (argc != 6) {
      printf("Wrong format\n");
      printf("Syntax: %s <Platform Device Input (%% elements) numElements value>\n",argv[0]);
      exit(1);
  }
  int platform = atoi(argv[1]);
  int device = atoi(argv[2]);
  int input = atoi(argv[3]);
  int numElements = atoi(argv[4]);
  size_t size = numElements * sizeof(T);
  int value = atoi(argv[5]);

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

  printf("DS Stream Compaction on %s\n", device_name_);
  printf("Thread block size = %d\n", L_DIM);
  printf("Coarsening factor = %d\n", REGS);
#ifdef FLOAT
  printf("Single precision array\n");
#elif INT
  printf("Integer array\n");
#else
  printf("Double precision array\n");
#endif

  bench(clDevices[device], clContext, numElements, input, value);

  // Release clContext
  clReleaseContext(clContext);
}
