all:
	nvcc -O3 -arch=sm_52 select.cu -I/usr/local/cuda/include -lm -o select -D COARSENING=32 -D THREADS=512 -D INT  
	#nvcc -O3 -arch=sm_52 select.cu -I/usr/local/cuda/include -lm -o select 

clean:
	rm select
