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

__kernel void streamcompaction(__global T *matrix_out, __global T *matrix,
    volatile __local int* sdata,
    volatile __local int* R,
    int size,
    volatile __global unsigned int *flags,
    T value)
{
  __local int count; // Per block counter
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
      if(reg[j] != value)
        local_cnt++;
    }
    else
      reg[j] = value;
    pos += get_local_size(0);
  }
  reduce(&count, local_cnt, &sdata[0]);

  // Set global synch
  ds_sync_irregular(flags, my_s, &count);

  // Store to global memory 
  #pragma unroll
  for (int j = 0; j < REGS; j++){
    pos = block_binary_prefix_sums(&count, (unsigned int)((reg[j] != value)?1:0), &sdata[0], &R[0]);
    if (reg[j] != value){
      matrix_out[pos] = reg[j];
    }
  }
}
