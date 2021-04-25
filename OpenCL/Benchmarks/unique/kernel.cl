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

T __shuffle_up(__global T* matrix, int my_s, int pos, T regi, int i, volatile __local int* R){
#if 0
#ifdef NVIDIA
  int offset = 1;
  T p;// = __shfl_up(regi, 1);
  asm volatile("{"
      " .reg .s32 r0;"
      " .reg .pred pp;"
      " shfl.up.b32 r0|pp, %1, %2, 0x0;"
      " @pp mov.s32 %0, r0;"
      "}" : "=r"(p): "r"(regi), "r"(offset));
#else
  T p = warp_up(regi, 1, &R[0]);
#endif
  if(lane_id() == 0 && i > 0)
    p = matrix[pos - 1]; 
  if(lane_id() == 0 && get_local_id(0) != 0 && i == 0)
    p = matrix[pos - 1];
  if(my_s > 0 && get_local_id(0) == 0 && i == 0)
    p = matrix[pos - 1];
  if(my_s == 0 && get_local_id(0) == 0 && i == 0)
    p = -1;
#else
  T p;
  if(pos != 0) p = matrix[pos - 1];
  else p = -1;
  return p;
#endif
}

__kernel void unique(__global T *matrix_out, __global T *matrix,
    volatile __local int* sdata,
    volatile __local int* R,
    int size,
    volatile __global unsigned int *flags)
{
  __local int count; // Counter for number of non-zero elements per block
  const int num_flags = size % (get_local_size(0) * REGS) == 0 ? size / (get_local_size(0) * REGS) : size / (get_local_size(0) * REGS) + 1;
  // Dynamic allocation of runtime workgroup id
  __local int gid_;
  int my_s = dynamic_wg_id(&gid_, flags, num_flags);

  int local_cnt = 0;
  // Declare on-chip memory
  T reg[REGS];
  int pos = my_s * REGS * get_local_size(0) + get_local_id(0);
  // Load in on-chip memory
  #pragma unroll
  for (int j = 0; j < REGS; j++){
    if (pos < size){
      reg[j] = matrix[pos];
      if(reg[j] != __shuffle_up(matrix, my_s, pos, reg[j], j, &R[0]))
        local_cnt++;
      else
        reg[j] = -1;
    }
    else
      reg[j] = -1;
    pos += get_local_size(0);
  }
  reduce(&count, local_cnt, &sdata[0]);

  // Set global synch
  ds_sync_irregular(flags, my_s, &count);

  // Store to global memory 
  #pragma unroll
  for (int j = 0; j < REGS; j++){
    pos = block_binary_prefix_sums(&count, reg[j] >= 0, &sdata[0], &R[0]);
    if (reg[j] >= 0){
      matrix_out[pos] = reg[j];
    }
  }
}
