#include <assert.h>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

using namespace std;

inline cudaError_t cudaCheckError(cudaError_t retval, const char *txt,
                                  const char *file, int line) {
#ifdef DEBUG
  std::cout << "[cuda] " << txt << std::endl;

  if (retval != cudaSuccess) {
    std::cout << "[cuda] error" << retval << " " << cudaGetErrorString(retval)
              << std::endl;
    std::cout << "[cuda] " << file << " " << line << std::endl;
  }
#endif

  return retval;
}

#define CUDACER(x) cudaCheckError((x), #x, __FILE__, __LINE__)
#define CUDA_ASSERT(x) assert(CUDACER(x) == cudaSuccess)

// Wait until the application notifies us that it has completed queuing up the
// experiment, or timeout and exit, allowing the application to make progress
__global__ void delay(volatile int *flag, uint64_t cyclesToSpin = 10000000) {
  uint64_t startClock = clock64();
  while (*flag == 0 && clock64() - startClock <= cyclesToSpin) {
    continue;
  }
}

// This kernel just occupies the GPU for the specified number of cycles
__global__ void incrementBufferAndSpin(int *buffer, uint64_t num_elems,
                                       uint64_t cyclesToSpin = 10000000) {
  uint64_t startClock = clock64();

  while (clock64() - startClock <= cyclesToSpin) {
    continue;
  }
}

auto getGpuClockRate() -> int {
  cudaDeviceProp prop;
  CUDA_ASSERT(cudaGetDeviceProperties(&prop, 0));
  std::cout << "Device properties:\n"
            << "Name: " << prop.name << std::endl
            << "pciBusID:" << std::hex << prop.pciBusID << std::endl
            << "pciDeviceID:" << std::hex << prop.pciDeviceID << std::endl
            << "pciDomainID:" << std::hex << prop.pciDomainID << std::endl
            << std::dec << std::endl;
  return prop.clockRate;
}

void copyAndSpin(uint64_t numElems, size_t objectSize,
                 uint64_t computeTimeMicroseconds) {
  // We use the first GPU for our experiments
  cudaSetDevice(0);

  uint64_t bufferSize = numElems * objectSize;

  // Allocate a pinned buffer on the host to copy from
  int *hostBuffer;
  CUDA_ASSERT(cudaMallocHost(&hostBuffer, bufferSize));
  for (int i = 0; i < bufferSize / sizeof(int); ++i)
    hostBuffer[i] = i;

  // Device side buffer to copy to.
  int *deviceBuffer;
  CUDA_ASSERT(cudaMalloc(&deviceBuffer, bufferSize));

  // We use this flag as a sort of release gate for Async events queued on a
  // stream. Primarily used so we can control when we set start and stop events
  // to make time measurements. See the 'delay' kernel.
  volatile int *flag = NULL;
  CUDA_ASSERT(
      cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable));

  // Stream that we enqueue our Async operations to.
  // Using a stream theoretically enables us to avoid measuring CPU enqueue time
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // Start and Stop events allow us to time the time it took to do the work
  cudaEvent_t start, stop;
  CUDA_ASSERT(cudaEventCreate(&start));
  CUDA_ASSERT(cudaEventCreate(&stop));

#ifdef PROFILE
  pid_t pid = fork();
  if (pid == 0) {
    int childOutFD = open("./cpu-utilization", O_WRONLY | O_CREAT,
                          S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    dup2(childOutFD, STDOUT_FILENO);
    dup2(childOutFD, STDERR_FILENO);
    close(childOutFD);
    execl("./cpu-stat", "cpu-stat");
    exit(EXIT_SUCCESS);
  } else {
#endif // PROFILE
    CUDA_ASSERT(cudaStreamSynchronize(stream));

    // Block the stream until all the work is queued up
    // DANGER! - cudaMemcpyAsync may infinitely block waiting for
    // room to push the operation, so keep the number of repetitions
    // relatively low.  Higher repetitions will cause the delay kernel
    // to timeout and lead to unstable results.
    *flag = 0;
    delay<<<1, 1, 0, stream>>>(flag);

    // Figure out what the max numbers for blockSize and numBlocks can be on
    // this GPU.
    int blockSize = 128;
    int numBlocks = 1024;
    CUDA_ASSERT(cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize,
                                                   incrementBufferAndSpin));

    uint64_t computeTimeCycles =
        (computeTimeMicroseconds / 1e6) * getGpuClockRate() * 1e3;
    std::cout << "Compute Time Cycles per element = " << computeTimeCycles
              << std::endl;

    // Enqueue GPU Commands to the stream; won't be executed until we set flag.
    CUDA_ASSERT(cudaEventRecord(start, stream));
    for (int e = 0; e < numElems; e++) {
      uint64_t offset = e * objectSize / sizeof(int);
      CUDA_ASSERT(cudaMemcpyAsync((void *)(hostBuffer + offset),
                                  (const void *)(deviceBuffer + offset),
                                  objectSize, cudaMemcpyDeviceToHost, stream));
      // This kernel increments each element in the buffer and then spins until
      // the desired number of cycles have elapsed.
      incrementBufferAndSpin<<<numBlocks, blockSize, 0, stream>>>(
          (int *)deviceBuffer + offset, objectSize / sizeof(int),
          computeTimeCycles);

      CUDA_ASSERT(cudaMemcpyAsync((void *)(deviceBuffer + offset),
                                  (const void *)(hostBuffer + offset),
                                  objectSize, cudaMemcpyHostToDevice, stream));
    }
    CUDA_ASSERT(cudaEventRecord(stop, stream));

    // Release the queued events and wait on stream synchronization event.
    *flag = 1;
    CUDA_ASSERT(cudaStreamSynchronize(stream));

    // Measure elapsed time.
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

#ifdef PROFILE
    kill(pid, SIGTERM);
    int status;
    waitpid(pid, &status, 0);
#endif // PROFILE

    std::cout << "Transferred " << bufferSize / (double)1e9 << " GB of data in "
              << time_ms << " ms." << std::endl
              << "Number of chunks = " << numElems << "." << std::endl
              << "Chunk size = " << objectSize << " B" << std::endl
              << "Per chunk compute time =" << computeTimeMicroseconds << " us."
              << std::endl;

    // Free buffers and destroy stream & events
    CUDA_ASSERT(cudaEventDestroy(stop));
    CUDA_ASSERT(cudaEventDestroy(start));
    CUDA_ASSERT(cudaStreamDestroy(stream));
    CUDA_ASSERT(cudaFree(deviceBuffer));
    CUDA_ASSERT(cudaFreeHost(hostBuffer));
    CUDA_ASSERT(cudaFreeHost((void *)flag));

#ifdef PROFILE
  }
#endif // PROFILE
}

void panicIfNoGPU() {
  std::cout << "Making sure there is at least one GPU on this system. Will use "
               "GPU 0 if there are multiple."
            << std::endl;
  int numGPUs = 0;
  CUDA_ASSERT(cudaGetDeviceCount(&numGPUs));
  assert(numGPUs != 0);
}

int main(int argc, char **argv) {
  uint64_t queueDepth = 500000; // 1 million
  size_t objectSize = 1024 * sizeof(int);
  int computeTimeMicroseconds = 10;

  // process command line args
  for (int i = 1; i < argc; i++) {
    if (0 == strcmp(argv[i], "-h")) {
      std::cerr << "Usage:" << argv[0] << " [OPTION]..." << std::endl
                << "Options:" << std::endl
                << "\t-h\tDisplay this Help menu" << std::endl
                << "\t-q\tNumber of Chunks ()" << std::endl
                << "\t-s\tChunk size (in increments of sizeof(int))"
                << std::endl
                << "\t-t\tPer chunk compute time to simulate (us)" << std::endl
                << std::endl
                << std::endl;
      return EXIT_FAILURE;
    } else if (0 == strcmp(argv[i], "-q")) {
      queueDepth = atoi(argv[i + 1]);
    } else if (0 == strcmp(argv[i], "-s")) {
      objectSize = atoi(argv[i + 1]) * sizeof(int);
    } else if (0 == strcmp(argv[i], "-t")) {
      computeTimeMicroseconds = atoi(argv[i + 1]);
    }
  }

  panicIfNoGPU();

  std::cout << "Synthetic GPU data movement and compute benchmark.\n";
  std::cout << "Moving " << queueDepth * objectSize / 1000 / 1000
            << " MB of data." << std::endl;
  std::cout << "Simulated compute time: " << computeTimeMicroseconds << " us."
            << std::endl;

  copyAndSpin(queueDepth, objectSize, computeTimeMicroseconds);

  exit(EXIT_SUCCESS);
}
