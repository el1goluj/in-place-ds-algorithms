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

// DS Unpadding kernel
__kernel void unpadding( __global T *matrix,
    int x_size,
    int pad_size,
    int y_size,
    volatile __global unsigned int *flags)
{
  const int num_flags = (y_size * pad_size) / (get_local_size(0) * REGS);
  // Dynamic allocation of runtime workgroup id
  __local int gid_;
  int my_s = dynamic_wg_id(&gid_, flags, num_flags);

  // Declare on-chip memory
  T reg[REGS];
  int pos = my_s * REGS * get_local_size(0) + get_local_id(0);
  // Load in on-chip memory
  #pragma unroll
  for (int j = 0; j < REGS; j++){
    if (pos < y_size * pad_size) reg[j] = matrix[pos];
    pos += get_local_size(0);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Set global synch
  ds_sync(flags, my_s);

  pos = my_s * REGS * get_local_size(0) + get_local_id(0);
  int my_s_row = pos / pad_size;
  int my_x = pos % pad_size;
  // Store to global memory 
  #pragma unroll
  for (int j = 0; j < REGS; j++){
    if (my_x < x_size && pos < y_size * pad_size) matrix[my_s_row * x_size + my_x] = reg[j];
    pos += get_local_size(0);
    my_s_row = pos / pad_size;
    my_x = pos % pad_size;
  }
  // Zeros at the end (Optionally)
  /*if ((my_s + 1) * REGS * get_local_size(0) >= y_size * x_size)
    for (int j = y_size * x_size + get_local_id(0); j < y_size * pad_size; j += get_local_size(0)){
      matrix[j] = 0.0f;
    }*/
}
