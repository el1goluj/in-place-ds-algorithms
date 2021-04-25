all:
	nvcc -O3 -arch=sm_52 streamcompaction.cu -I/usr/local/cuda/include -lm -o streamcompaction -D COARSENING=32 -D THREADS=512 -D INT  
	#nvcc -O3 -arch=sm_52 streamcompaction.cu -I/usr/local/cuda/include -lm -o streamcompaction 

clean:
	rm streamcompaction
