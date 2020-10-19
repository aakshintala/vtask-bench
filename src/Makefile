# LiveStreamDarknet Makefile.

CXX = clang++
NVCC = nvcc
CPPFLAGS += -g -O3 -std=c++11 -I/usr/local/cuda/include/
LDFLAGS = -ldl -lpthread -L/usr/local/cuda/lib64 -lcuda -lcudart -L/usr/local/lib64 -lqatzip
#-lnvToolsExt -lnvidia-ml

all: vtask-bench

vtask-bench: benchmark.cu
	nvcc -ccbin g++ $(CPPFLAGS) -m64 $(LDFLAGS) -arch=sm_52 -o $@ $+

clean:
	rm -f *.o vtask-bench 
