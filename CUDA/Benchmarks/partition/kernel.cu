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

__global__ void partition(T *matrix_out1, T *matrix_out2, T *matrix,
    int size,
    volatile unsigned int *flags1,
    volatile unsigned int *flags2,
    struct is_even pred)
{
  __shared__ int count1; // Counter for "true" elements
  __shared__ int count2; // Counter for "false" elements
  const int num_flags = size % (blockDim.x * REGS) == 0 ? size / (blockDim.x * REGS) : size / (blockDim.x * REGS) + 1;

  // Dynamic allocation of runtime workgroup id
  if (threadIdx.x == 0){
    count1 = 0;
    count2 = 0;
  }
  const int my_s = dynamic_wg_id(flags1, num_flags);

  int local_cnt1 = 0;
  int local_cnt2 = 0;
  // Declare on-chip memory
  T reg[REGS];
  int pos = my_s * REGS * blockDim.x + threadIdx.x;
  // Load in on-chip memory
  #pragma unroll
  for (int j = 0; j < REGS; j++){
    if (pos < size){
      reg[j] = matrix[pos];
      if(pred(reg[j]))
        local_cnt1++;
      else
        local_cnt2++;
    }
    else
      reg[j] = -1;
    pos += blockDim.x;
  }
  reduction<int>(&count1, local_cnt1);
  __syncthreads();
  reduction<int>(&count2, local_cnt2);  

  // Set global synch
  ds_sync_irregular_partition(flags1, flags2, my_s, &count1, &count2);

  // Store to global memory 
  #pragma unroll
  for (int j = 0; j < REGS; j++){
    pos = block_binary_prefix_sums(&count1, pred(reg[j]) && reg[j] >= 0);
    if (pred(reg[j]) && reg[j] >= 0){
      matrix_out1[pos] = reg[j];
    }
    pos = block_binary_prefix_sums(&count2, (!pred(reg[j])) && reg[j] >= 0);
    if (!pred(reg[j]) && reg[j] >= 0){
      matrix_out2[pos] = reg[j];
    }
  }
}

__global__ void move_part(T* trues, T* falses, unsigned int _M, unsigned int size){
  const unsigned int gtid = blockDim.x * blockIdx.x + threadIdx.x;
  if(_M + gtid < size)
    trues[_M + gtid] = falses[gtid];
}
