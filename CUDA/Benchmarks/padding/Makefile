all:
	nvcc -arch=sm_52 padding.cu -I/usr/local/cuda/include -lm -o padding
	#nvcc -arch=sm_52 padding.cu -I/usr/local/cuda/include -lm -o padding -D FLOAT -D COARSENING=32 -D THREADS=256

clean:
	rm padding
