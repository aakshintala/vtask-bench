#include <assert.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

#include "qatzip.h" // Fix this to point to the right header.

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
__global__ void spinForNCycles(int *buffer, uint64_t num_elems,
                               uint64_t cyclesToSpin = 10000000) {
  uint64_t startClock = clock64();

  while (clock64() - startClock <= cyclesToSpin) {
    continue;
  }
}

auto getGpuClockRate(bool minimal) -> int {
  cudaDeviceProp prop;
  CUDA_ASSERT(cudaGetDeviceProperties(&prop, 0));

  if (!minimal) {
    std::cout << "Device properties:\n"
              << "Name: " << prop.name << std::endl
              << "pciBusID:" << std::hex << prop.pciBusID << std::endl
              << "pciDeviceID:" << std::hex << prop.pciDeviceID << std::endl
              << "pciDomainID:" << std::hex << prop.pciDomainID << std::endl
              << std::dec << std::endl;
  }

  return prop.clockRate;
}

static int processBuffer(QzSession_T *sess, unsigned char *src,
                         unsigned int *src_len, unsigned char *dst,
                         unsigned int dst_len, unsigned int *compressed_size,
                         bool isCompress) {
  int ret = QZ_FAIL;
  unsigned int done = 0;
  unsigned int buf_processed = 0;
  unsigned int buf_remaining = *src_len;
  unsigned int valid_dst_buf_len = dst_len;

  *compressed_size = 0;

  while (!done) {
    /* Do actual work */
    if (isCompress) {
      ret = qzCompress(sess, src, src_len, dst, &dst_len, 1);
    } else {
      ret = qzDecompress(sess, src, src_len, dst, &dst_len);

      if (QZ_DATA_ERROR == ret || (QZ_BUF_ERROR == ret && 0 == *src_len)) {
        done = 1;
      }
    }

    if (QZ_OK != ret && QZ_BUF_ERROR != ret && QZ_DATA_ERROR != ret) {
      std::cerr << "doProcessBuffer failed with error: " << ret << std::endl;
      break;
    }

    buf_processed += *src_len;
    buf_remaining -= *src_len;
    if (0 == buf_remaining) {
      done = 1;
    }
    src += *src_len;
    *src_len = buf_remaining;
    *compressed_size += dst_len;
    dst_len = valid_dst_buf_len;
  }

  *src_len = buf_processed;
  return ret;
}

double measureDecompressionTime(unsigned char *buffer, uint32_t bufferSize,
                                uint32_t numObjects, uint32_t objectSize,
                                QzSession_T *session) {
  unsigned char *destBuffer[numObjects] = {nullptr};
  for (int i = 0; i < numObjects; ++i)
    CUDA_ASSERT(cudaMallocHost(&destBuffer[i], objectSize));
  unsigned int destBufferSize = objectSize;

  unsigned int srcLen = objectSize;
  unsigned char *srcBuf = buffer;

  unsigned int compressedSize[numObjects] = {0};

  // Compress buffer
  for (int i = 0; i < numObjects; ++i) {
    int ret =
        processBuffer(session, (srcBuf + i * objectSize), &srcLen,
                      destBuffer[i], destBufferSize, &compressedSize[i], true);
    if (ret != QZ_OK) {
      std::cerr << "QATZip compression failed. ret = " << ret << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Start timer
  using namespace std::chrono;
  high_resolution_clock::time_point timer = high_resolution_clock::now();

  for (int i = 0; i < numObjects; ++i) {
    // Decompress buffer
    unsigned int compressedLen = compressedSize[i];
    int ret = processBuffer(session, destBuffer[i], &compressedLen,
                            (srcBuf + i * objectSize), srcLen,
                            &compressedSize[i], false);
    // Check for errors
    if (ret != QZ_OK) {
      std::cerr << "QATZip decompression failed. ret = " << ret << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // Stop Timer
  double usElapsed =
      duration_cast<microseconds>(high_resolution_clock::now() - timer).count();

  // Free the allocated DestBuffers
  for (int i = 0; i < numObjects; ++i)
    cudaFreeHost(destBuffer[i]);

  return usElapsed;
}

void decompressAndCopyAndSpin(uint32_t numElems, uint32_t objectSize,
                              bool minimal, uint32_t computeTimeMicroseconds,
                              QzSession_T *session) {
  // We use the first GPU for our experiments
  cudaSetDevice(0);

  uint64_t bufferSize = numElems * objectSize;

  // Allocate and initialize a pinned buffer on the host to copy from
  unsigned char *hostBuffer;
  CUDA_ASSERT(cudaMallocHost(&hostBuffer, bufferSize));
  for (int i = 0; i < bufferSize; ++i) {
    hostBuffer[i] = i % CHAR_MAX;
  }

  // Device side buffer to copy to.
  unsigned char *deviceBuffer;
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

    // measure the time spent 'decompressing' the data on QATZip.
    double QAT_time_us = measureDecompressionTime(
        hostBuffer, bufferSize, numElems, objectSize, session);

    // Do the GPU work.
    CUDA_ASSERT(cudaStreamSynchronize(stream));

    // Block the stream until all the work is queued up
    // DANGER! - cudaMemcpyAsync may infinitely block waiting for
    // room to push the operation, so keep the number of repetitions
    // relatively low.  Higher repetitions will cause the delay kernel
    // to timeout and lead to unstable results.
    *flag = 0;
    delay<<<1, 1, 0, stream>>>(flag);

    uint64_t computeTimeCycles =
        (computeTimeMicroseconds / 1e6) * getGpuClockRate(minimal) * 1e3;

    if (!minimal) {
      std::cout << "Compute Time Cycles per element = " << computeTimeCycles
                << std::endl;
    }

    // Enqueue GPU Commands to the stream; won't be executed until we set flag.
    CUDA_ASSERT(cudaEventRecord(start, stream));
    for (int e = 0; e < numElems; e++) {
      uint64_t offset = e * objectSize;
      CUDA_ASSERT(cudaMemcpyAsync((void *)(hostBuffer + offset),
                                  (const void *)(deviceBuffer + offset),
                                  objectSize, cudaMemcpyDeviceToHost, stream));
      // This kernel until spins desired the number of cycles have elapsed.
      spinForNCycles<<<1024, 26, 0, stream>>>(
          (int *)deviceBuffer + offset, objectSize, computeTimeCycles);

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

    if (minimal) {
      std::cout << numElems << " " << objectSize << " "
                << computeTimeMicroseconds << " " << time_ms << " | "
                << QAT_time_us / 1e3 << std::endl;
    } else {
      std::cout << "Transferred " << bufferSize / (double)1e9
                << " GB of data in " << time_ms << " ms." << std::endl
                << "Number of chunks = " << numElems << "." << std::endl
                << "Chunk size = " << objectSize << " B" << std::endl
                << "Per chunk compute time =" << computeTimeMicroseconds
                << " us." << std::endl;
      std::cout << "Time spent decompressing on QAT = " << QAT_time_us / 1e3
                << " ms." << std::endl;
    }
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

void panicIfNoGPU(bool minimal) {
  if (!minimal) {
    std::cout << "Making sure there is at least one GPU on this system. "
              << "Will use GPU 0 if there are multiple." << std::endl;
  }
  int numGPUs = 0;
  CUDA_ASSERT(cudaGetDeviceCount(&numGPUs));
  assert(numGPUs != 0);
}

int main(int argc, char **argv) {
  uint32_t queueDepth = 256 * 1024;
  uint32_t objectSize = 4096;
  uint32_t computeTimeMicroseconds = 1;
  bool minimalOutput = false;

  // process command line args
  for (int i = 1; i < argc; ++i) {
    if (0 == strcmp(argv[i], "-h")) {
      std::cerr << "Usage:" << argv[0] << " [OPTION]..." << std::endl
                << "Options:" << std::endl
                << "\t-m\tMinimal output for processing." << std::endl
                << "\t-h\tDisplay this Help menu" << std::endl
                << "\t-q\tNumber of Chunks" << std::endl
                << "\t-s\tChunk size (in increments of pages (4 KiB))" << std::endl
                << "\t-t\tPer chunk compute time to simulate (us)" << std::endl
                << std::endl
                << std::endl;
      return EXIT_FAILURE;
    } else if (0 == strcmp(argv[i], "-q")) {
      queueDepth = std::stoul(argv[i + 1]);
      //queueDepth = atoi(argv[i + 1]);
    } else if (0 == strcmp(argv[i], "-s")) {
      objectSize = std::stoul(argv[i + 1]);
      //objectSize = atoi(argv[i + 1]) * sizeof(int);
    } else if (0 == strcmp(argv[i], "-t")) {
      computeTimeMicroseconds = std::stoul(argv[i + 1]);
      //computeTimeMicroseconds = atoi(argv[i + 1]);
    } else if (0 == strcmp(argv[i], "-m")) {
      minimalOutput = true;
    }
  }

  QzSession_T session;
  session.internal = nullptr;

  panicIfNoGPU(minimalOutput);

  if (!minimalOutput) {
    std::cout << "Synthetic GPU data movement and compute benchmark.\n";
    //std::cout << "Moving " << queueDepth * objectSize / 1024 / 1024
    std::cout << "Moving " << queueDepth * objectSize << " B of data." << std::endl;
    std::cout << "Simulated compute time: " << computeTimeMicroseconds << " us."
              << std::endl;
  }

  decompressAndCopyAndSpin(queueDepth, objectSize, minimalOutput,
                           computeTimeMicroseconds, &session);

  exit(EXIT_SUCCESS);
}
