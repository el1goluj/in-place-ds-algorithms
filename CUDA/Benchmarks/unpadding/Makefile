all:
	nvcc -arch=sm_52 unpadding.cu -I/usr/local/cuda/include -lm -o unpadding
	#nvcc -arch=sm_52 unpadding.cu -I/usr/local/cuda/include -lm -o unpadding -D FLOAT -D COARSENING=32 -D THREADS=256

clean:
	rm unpadding 
