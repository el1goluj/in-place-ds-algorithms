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

#ifdef FLOAT
#define T float
#elif INT
#define T int
#else
#define T double
#endif

#ifdef THREADS
#define L_DIM THREADS
#else 
#define L_DIM 256
#endif

#ifdef COARSENING
#define REGS COARSENING
#else
#ifdef FLOAT
#define REGS 16
#elif INT
#define REGS 16
#else
#define REGS 8 
#endif
#endif

#ifdef ATOMIC
#define ATOM 1
#else
#define ATOM 0
#endif

#ifdef NVIDIA
#define WARP_SIZE 32
#else
#define WARP_SIZE 64
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

// Dynamic allocation of runtime workgroup id
int dynamic_wg_id(__local int* gid_, volatile __global unsigned int *flags, const int num_flags){
  if (get_local_id(0) == 0) *gid_ = atom_add(&flags[num_flags + 1], 1);
  barrier(CLK_LOCAL_MEM_FENCE);
  int my_s = *gid_;
  return my_s;
}

// Set global synchronization (regular DS)
void ds_sync(volatile __global unsigned int *flags, const int my_s){
#if ATOM
  if (get_local_id(0) == 0){
    while (atom_or(&flags[my_s], 0) == 0){}
    atom_or(&flags[my_s + 1], 1);
  }
#else
  if (get_local_id(0) == 0){
    while (flags[my_s] == 0){}
    flags[my_s + 1] = 1;
  }
#endif
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

// Set global synchronization (irregular DS)
void ds_sync_irregular(volatile __global unsigned int *flags, const int my_s, __local int *count){
#if ATOM
  if (get_local_id(0) == 0){
    while (atom_or(&flags[my_s], 0) == 0){}
    int flag = flags[my_s];
    atom_add(&flags[my_s + 1], flag + *count);
    *count = flag - 1;
  }
#else
  if (get_local_id(0) == 0){
    while (flags[my_s] == 0){}
    int flag = flags[my_s];
    flags[my_s + 1] = flag + *count;
    *count = flag - 1;
  }
#endif
  barrier(CLK_GLOBAL_MEM_FENCE);
}

// Set global synchronization (irregular DS Partition)
void ds_sync_irregular_partition(volatile __global unsigned int *flags1, volatile __global unsigned int *flags2, const int my_s, __local int *count1, __local int *count2){
#if ATOM
  if (get_local_id(0) == 0){
    while (atom_or(&flags1[my_s], 0) == 0){}
    int flag2 = flags2[my_s];
    atom_add(&flags2[my_s + 1], flag2 + *count2);
    int flag1 = flags1[my_s];
    atom_add(&flags1[my_s + 1], flag1 + *count1);
    *count2 = flag2 - 1;
    *count1 = flag1 - 1;
  }
#else
  if (get_local_id(0) == 0){
    while (flags1[my_s] == 0){}
    int flag2 = flags2[my_s];
    flags2[my_s + 1] = flag2 + *count2;
    int flag1 = flags1[my_s];
    flags1[my_s + 1] = flag1 + *count1;
    *count2 = flag2 - 1;
    *count1 = flag1 - 1;
  }
#endif
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

// Reduction kernel
void reduce(__local int* count, int local_cnt, __local int* sdata){
#ifndef SEQUENTIAL
  unsigned int tid = get_local_id(0);
  unsigned int bid = get_group_id(0);

  // Load local mem
  unsigned int localSize = get_local_size(0);
  sdata[tid] = local_cnt;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Do reduction in local mem
  for(unsigned int s = localSize >> 1; s > 0; s >>= 1){
    if(tid < s){
      sdata[tid] += sdata[tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write result for this block to global mem
  if(tid == 0) *count = sdata[0];
#else
  unsigned int tid = get_local_id(0);

  // Load local mem
  sdata[tid] = local_cnt;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Do reduction in local mem
  if(tid==0){
    int sum=0;
    for(int i=0;i<get_local_size(0);i++){
      sum += sdata[i];
    }
    *count = sum;
  }
#endif
}

// Binary prefix-sum
inline int lane_id(void) { return get_local_id(0) % WARP_SIZE; }
inline int warp_id(void) { return get_local_id(0) / WARP_SIZE; }

int warp_up(int reg, int delta, volatile __local int* R){
  R[get_local_id(0)] = reg;
  return (lane_id() - delta >= 0 ? R[get_local_id(0) - delta] : 0);
}

int warp_scan(int val, volatile __local int *s_data, volatile __local int* R){
#ifndef SHFL
  int idx = 2 * get_local_id(0) - (get_local_id(0) & (WARP_SIZE - 1));
  s_data[idx] = 0;
  idx += WARP_SIZE;
  int t = s_data[idx] = val;
  s_data[idx] = t = t + s_data[idx - 1];
  s_data[idx] = t = t + s_data[idx - 2];
  s_data[idx] = t = t + s_data[idx - 4];
  s_data[idx] = t = t + s_data[idx - 8];
  s_data[idx] = t = t + s_data[idx - 16];
#ifndef NVIDIA
  s_data[idx] = t = t + s_data[idx - 32];
#endif
  return s_data[idx - 1];
#else
  int x = val;
  #pragma unroll
  for(int offset = 1; offset < WARP_SIZE; offset <<= 1){
  // From GTC: Kepler shuffle tips and tricks:
#ifndef NVIDIA
    int y = warp_up(x, offset, &R[0]);
    if(lane_id() >= offset)
      x += y;
#else
    asm volatile("{"
        " .reg .s32 r0;"
        " .reg .pred p;"
        " shfl.up.b32 r0|p, %0, %1, 0x0;"
        " @p add.s32 r0, r0, %0;"
        " mov.s32 %0, r0;"
        "}" : "+r"(x) : "r"(offset));
#endif
  }
  return x - val;
#endif
}

int block_binary_prefix_sums(__local int* count, int x, volatile __local int* sdata, volatile __local int* R){
#ifndef SEQUENTIAL
#if defined(NVIDIA) || defined(SHFL)
  // A. Exclusive scan within each warp
  int warpPrefix = warp_scan(x, &sdata[0], &R[0]);

  // B. Store in shared memory
  if(lane_id() == WARP_SIZE - 1)
    sdata[warp_id()] = warpPrefix + x;
  barrier(CLK_LOCAL_MEM_FENCE); 

  // C. One warp scans in shared memory
  if(get_local_id(0) < WARP_SIZE)
    sdata[get_local_id(0)] = warp_scan(sdata[get_local_id(0)], &sdata[0], &R[0]);
  barrier(CLK_LOCAL_MEM_FENCE); 

  // D. Each thread calculates it final value
  int thread_out_element = warpPrefix + sdata[warp_id()];
  int output = thread_out_element + *count;
  barrier(CLK_LOCAL_MEM_FENCE); 
  if(get_local_id(0) == get_local_size(0) - 1)
    *count += (thread_out_element + x);
#else
  sdata[get_local_id(0)] = x;
  unsigned int length = get_local_size(0);
  // Build up tree 
  int offset = 1;
  for(int l = length>>1; l > 0; l >>= 1){
    barrier(CLK_LOCAL_MEM_FENCE);
    if(get_local_id(0) < l) {
      int ai = offset*(2*get_local_id(0) + 1) - 1;
      int bi = offset*(2*get_local_id(0) + 2) - 1;
      sdata[bi] += sdata[ai];
    }
    offset <<= 1;
  }

  if(offset < length) { offset <<= 1; }

  // Build down tree
  int maxThread = offset>>1;
  for(int d = 0; d < maxThread; d<<=1){
    d += 1;
    offset >>=1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_local_id(0) < d) {
      int ai = offset*(get_local_id(0) + 1) - 1;
      int bi = ai + (offset>>1);
      sdata[bi] += sdata[ai];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  int output = sdata[get_local_id(0)] + *count - x;
  barrier(CLK_LOCAL_MEM_FENCE);
  if(get_local_id(0) == get_local_size(0) - 1)
    *count += sdata[get_local_id(0)];
#endif
#else
  // sequential one
  int idx = get_local_id(0);
  sdata[idx] = x;
  barrier(CLK_LOCAL_MEM_FENCE);
  int output=0;

  if(idx==0){
    for(int ii = 1; ii<get_local_size(0);ii++){
       sdata[ii] =sdata[ii]+sdata[ii-1];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE); 

  output = sdata[idx] + *count-x;
  if(idx == get_local_size(0) - 1)
    *count =*count+ sdata[idx];

  barrier(CLK_LOCAL_MEM_FENCE); 
#endif
  return output;
}
