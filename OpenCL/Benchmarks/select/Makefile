all:
	g++ -g select.cpp -I/usr/local/cuda/include -I/opt/AMDAPPSDK-3.0-0-Beta/include/ -lOpenCL -lm -o select -D INT
	#g++ -g select.cpp -I/usr/local/cuda/include -I/opt/AMDAPPSDK-3.0-0-Beta/include/ -lOpenCL -lm -o select -D INT -D THREADS=128 -D COARSENING=32 -D ATOMIC -D SEQUENTIAL
	#g++ -g select.cpp -I/usr/local/cuda/include -I/opt/AMDAPPSDK-3.0-0-Beta/include/ -lOpenCL -lm -o select -D INT -D THREADS=128 -D COARSENING=32 -D ATOMIC -D NVIDIA -D SHFL
	#Mac OS X : -framework OpenCL

clean:
	rm select
