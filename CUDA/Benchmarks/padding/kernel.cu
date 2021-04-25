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

// DS padding kernel
__global__ void padding(T *matrix,
    int x_size,
    int pad_size,
    int y_size,
    volatile unsigned int *flags)
{
  const int matrix_size = y_size * pad_size;
  const int matrix_size_align = (matrix_size + blockDim.x - 1) / blockDim.x * blockDim.x; 
  const int num_flags = matrix_size / (blockDim.x * REGS);

  // Dynamic allocation of runtime workgroup id
  int my_s = dynamic_wg_id(flags, num_flags);

  // Declare on-chip memory
  T reg[REGS];
  int pos = matrix_size_align - 1 - (my_s * REGS * blockDim.x + threadIdx.x);
  int my_s_row = pos / pad_size;
  int my_x = pos % pad_size;
  int pos2 = my_s_row * x_size + my_x;
  // Load in on-chip memory
  #pragma unroll
  for (int j = 0; j < REGS; j++){
    if (pos2 >= 0 && my_x < x_size) reg[j] = matrix[pos2];
    else reg[j] = 0;
    pos -= blockDim.x;
    my_s_row = pos / pad_size;
    my_x = pos % pad_size;
    pos2 = my_s_row * x_size + my_x;
  }

  __syncthreads();

  // Set global synch
  ds_sync(flags, my_s);

  pos = matrix_size_align - 1 - (my_s * REGS * blockDim.x + threadIdx.x);
  // Store to global memory 
  #pragma unroll
  for (int j = 0; j < REGS; j++){
    if (pos >= 0 && pos < matrix_size) matrix[pos] = reg[j];
    pos -= blockDim.x;
  }
}
