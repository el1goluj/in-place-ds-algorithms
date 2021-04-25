all:
	nvcc -O3 -arch=sm_52 unique.cu -I/usr/local/cuda/include -lm -o unique -D COARSENING=32 -D THREADS=512 -D INT  
	#nvcc -O3 -arch=sm_52 unique.cu -I/usr/local/cuda/include -lm -o unique 

clean:
	rm unique
