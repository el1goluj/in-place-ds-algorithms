all:
	nvcc -O3 -arch=sm_52 partition.cu -I/usr/local/cuda/include -lm -o partition -D COARSENING=32 -D THREADS=512 -D INT  
	#nvcc -O3 -arch=sm_52 partition.cu -I/usr/local/cuda/include -lm -o partition 

clean:
	rm partition
