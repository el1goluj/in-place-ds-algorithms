all:
	g++ -g unique.cpp -I/usr/local/cuda/include -I/opt/AMDAPPSDK-3.0-0-Beta/include/ -lOpenCL -lm -o unique -D INT
	#g++ -g unique.cpp -I/usr/local/cuda/include -I/opt/AMDAPPSDK-3.0-0-Beta/include/ -lOpenCL -lm -o unique -D INT -D THREADS=128 -D COARSENING=32 -D ATOMIC -D SEQUENTIAL
	#g++ -g unique.cpp -I/usr/local/cuda/include -I/opt/AMDAPPSDK-3.0-0-Beta/include/ -lOpenCL -lm -o unique -D INT -D THREADS=128 -D COARSENING=32 -D ATOMIC -D NVIDIA -D SHFL
	#Mac OS X : -framework OpenCL

clean:
	rm unique
