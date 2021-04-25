Compilation flags

FLOAT - For single precision arrays (Default: Double precision)
INT - For integer arrays (Note: Sample predicate is only for INT)
THREADS - Work-group size (Default: 256)
COARSENING - Coarsening factor (Default: 16 (SP and INT); 8 (DP))
ATOMIC - Global atomics for synchronization (Default: No atomics)
NVIDIA - For Nvidia GPUs specific code
SHFL - Use of shuffle instructions (Nvidia Kepler or later), or pseudo-shuffle through local memory
SEQUENTIAL - Use of sequential versions of reduce and prefix-sum (for CPUs)
