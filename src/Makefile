# LiveStreamDarknet Makefile.

CXX = clang++
NVCC = nvcc
CPPFLAGS += -g -O3 -std=c++11 -I/usr/local/cuda/include/
LDFLAGS = -ldl -lpthread -L/usr/local/cuda/lib64 -lcuda -lcudart 
#-lnvToolsExt -lnvidia-ml

all: vtask-bench

vtask-bench: benchmark.cu
#vtask-bench: vtask-bench.o
	nvcc -ccbin g++ $(CPPFLAGS) -m64 $(LDFLAGS) -arch=sm_52 -o $@ $+

#vtask-bench.o: benchmark.cu
#	nvcc -ccbin g++ -m64 $(CPPFLAGS) $(LDFLAGS) -arch=sm_52 -o $@ -c $^

clean:
	rm -f *.o vtask-bench 
