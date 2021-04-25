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

// Sample predicate for partition
struct is_even{
  bool operator()(const int &x){
    return (x % 2) == 0;
  }
};

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

// GPU version
double gpu_partition(cl_device_id device, cl_context clContext, T* matrix, int numElements, int Ma){

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
  cl_kernel clKernel = clCreateKernel(clProgram,"partition", &clStatus);
  CL_ERR();
  cl_kernel clKernel_move = clCreateKernel(clProgram,"move_part", &clStatus);
  CL_ERR();

  // Allocate matrix
  cl_mem d_matrix = clCreateBuffer(clContext, CL_MEM_READ_WRITE, numElements * sizeof(T), NULL, &clStatus);
  cl_mem d_matrix_aux = clCreateBuffer(clContext, CL_MEM_READ_WRITE, numElements * sizeof(T), NULL, &clStatus);
  clEnqueueWriteBuffer(clCommandQueue, d_matrix, CL_TRUE, 0, sizeof(T) * numElements, matrix, 0, NULL, NULL);
  clEnqueueWriteBuffer(clCommandQueue, d_matrix_aux, CL_TRUE, 0, sizeof(T) * numElements, matrix, 0, NULL, NULL);
  clFinish(clCommandQueue);
  CL_ERR();

  int ldim = L_DIM;
  // Atomic flags
  const int num_flags = numElements % (ldim * REGS) == 0 ? numElements / (ldim * REGS) : numElements / (ldim * REGS) + 1;
  unsigned int *flags = (unsigned int *)calloc(sizeof(unsigned int), num_flags + 2);
  flags[0] = 1;
  flags[num_flags + 1] = 0;
  cl_mem d_flags1 = clCreateBuffer(clContext, CL_MEM_READ_WRITE, (num_flags + 2) * sizeof(unsigned int), NULL, &clStatus);
  clEnqueueWriteBuffer(clCommandQueue, d_flags1, CL_TRUE, 0, sizeof(unsigned int) * (num_flags + 2), flags, 0, NULL, NULL);
  cl_mem d_flags2 = clCreateBuffer(clContext, CL_MEM_READ_WRITE, (num_flags + 2) * sizeof(unsigned int), NULL, &clStatus);
  clEnqueueWriteBuffer(clCommandQueue, d_flags2, CL_TRUE, 0, sizeof(unsigned int) * (num_flags + 2), flags, 0, NULL, NULL);
  clFinish(clCommandQueue);
  free(flags);
  CL_ERR();

  // Number of work-groups/thread blocks
  int num_wg = num_flags;
  struct is_even pred;
	
  // Set kernel arguments
  clSetKernelArg(clKernel, 0, sizeof(cl_mem), &d_matrix);
  clSetKernelArg(clKernel, 1, sizeof(cl_mem), &d_matrix_aux);
  clSetKernelArg(clKernel, 2, sizeof(cl_mem), &d_matrix);
  clSetKernelArg(clKernel, 3, L_DIM*sizeof(int), NULL);
  clSetKernelArg(clKernel, 4, L_DIM*sizeof(int), NULL);
  clSetKernelArg(clKernel, 5, sizeof(int), &numElements);
  clSetKernelArg(clKernel, 6, sizeof(cl_mem), &d_flags1);
  clSetKernelArg(clKernel, 7, sizeof(cl_mem), &d_flags2);
  //clSetKernelArg(clKernel, 8, sizeof(struct is_even), &pred);

  clSetKernelArg(clKernel_move, 0, sizeof(cl_mem), &d_matrix);
  clSetKernelArg(clKernel_move, 1, sizeof(cl_mem), &d_matrix_aux);
  clSetKernelArg(clKernel_move, 2, sizeof(int), &Ma);
  clSetKernelArg(clKernel_move, 3, sizeof(int), &numElements);

  // Start timer
  gettimeofday(&t1, NULL);

  // Kernel launch 
  size_t ls[1] = {ldim};
  size_t gs[1] = {ldim * num_wg};
  clStatus = clEnqueueNDRangeKernel(clCommandQueue, clKernel, 1, NULL, gs, ls, 0, NULL, NULL);
  CL_ERR();
  num_wg = (numElements - Ma) % ldim == 0 ? (numElements - Ma) / ldim : (numElements - Ma) / ldim + 1;
  size_t gs2[1] = {ldim * num_wg};
  if(numElements != Ma) clStatus = clEnqueueNDRangeKernel(clCommandQueue, clKernel_move, 1, NULL, gs2, ls, 0, NULL, NULL);
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
  clReleaseMemObject(d_flags1);
  clReleaseMemObject(d_flags2);
  clReleaseMemObject(d_matrix);
  clReleaseMemObject(d_matrix_aux);
  clReleaseKernel(clKernel);
  clReleaseProgram(clProgram);
  clReleaseCommandQueue(clCommandQueue);

  return bandwidth;
}

int bench(cl_device_id device, cl_context clContext, int numElements, int input) {
  T* M = (T*) malloc (numElements * sizeof(T));
  T* M2 = (T*) malloc (numElements * sizeof(T));

  // Initialize the host input vectors
  srand(2014);
  for(int i = 0; i < numElements; i++)
    M[i] = i % 2 != 0 ? i:i+1;
  int Ma = (numElements * input)/100;
  int ma = Ma;
  while(ma>0){
    int x = (int)(numElements*(((float)rand()/(float)RAND_MAX)));
    if(M[x] % 2 != 0){
      M[x] = x * 2;
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
  cpu_partition(M2, numElements, is_even());

#if PRINT
  for (int i = 0; i < numElements; i++)
    printf("%d ", M2[i]);
  printf("\n");
#endif

  // GPU execution
  double bw = gpu_partition(device, clContext, M, numElements, Ma);

#if PRINT
  printf("\n");
  for (int i = 0; i < numElements; i++)
    printf("%d ", M[i]);
  printf("\n");
#endif

  // Check results
  for (int i = 0; i < numElements; ++i){
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
  if (argc != 5) {
      printf("Wrong format\n");
      printf("Syntax: %s <Platform Device Input (%% elements) numElements>\n",argv[0]);
      exit(1);
  }
  int platform = atoi(argv[1]);
  int device = atoi(argv[2]);
  int input = atoi(argv[3]);
  int numElements = atoi(argv[4]);
  size_t size = numElements * sizeof(T);

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

  printf("DS Partition on %s\n", device_name_);
  printf("Thread block size = %d\n", L_DIM);
  printf("Coarsening factor = %d\n", REGS);
#ifdef FLOAT
  printf("Single precision array\n");
#elif INT
  printf("Integer array\n");
#else
  printf("Double precision array\n");
#endif

  bench(clDevices[device], clContext, numElements, input);

  // Release clContext
  clReleaseContext(clContext);
}    
