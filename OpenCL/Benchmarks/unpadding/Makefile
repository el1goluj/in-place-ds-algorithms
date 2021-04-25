all:
	g++ -g unpadding.cpp -I/usr/local/cuda/include -I/opt/AMDAPPSDK-3.0-0-Beta/include/ -lOpenCL -lm -o unpadding
	#g++ -g unpadding.cpp -I/usr/local/cuda/include -I/opt/AMDAPPSDK-3.0-0-Beta/include/ -lOpenCL -lm -o unpadding -D FLOAT -D THREADS=128 -D COARSENING=32 -D ATOMIC
	#Mac OS X : -framework OpenCL

clean:
	rm unpadding
