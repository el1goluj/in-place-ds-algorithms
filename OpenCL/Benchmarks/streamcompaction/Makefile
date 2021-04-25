all:
	g++ -g streamcompaction.cpp -I/usr/local/cuda/include -I/opt/AMDAPPSDK-3.0-0-Beta/include/ -lOpenCL -lm -o streamcompaction -D INT
	#g++ -g streamcompaction.cpp -I/usr/local/cuda/include -I/opt/AMDAPPSDK-3.0-0-Beta/include/ -lOpenCL -lm -o streamcompaction -D INT -D THREADS=128 -D COARSENING=32 -D ATOMIC -D SEQUENTIAL
	#g++ -g streamcompaction.cpp -I/usr/local/cuda/include -I/opt/AMDAPPSDK-3.0-0-Beta/include/ -lOpenCL -lm -o streamcompaction -D INT -D THREADS=128 -D COARSENING=32 -D ATOMIC -D NVIDIA -D SHFL
	#Mac OS X : -framework OpenCL

clean:
	rm streamcompaction
