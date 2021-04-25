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

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <math.h>
#include <sys/time.h>
#include <vector>
#include <CL/cl.h>
//#include <OpenCL/cl.h> // Mac OS X

#define PRINT 0

#define CL_ERR()					\
  if(clStatus != CL_SUCCESS) {					\
    fprintf(stderr, "OpenCL error: %d\n at %s, %d\n",	\
        clStatus, __FILE__, __LINE__);			\
    exit(-1);						\
  }

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
