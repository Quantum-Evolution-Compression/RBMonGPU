#ifdef __CUDA_ARCH__
#define SHARED __shared__
#define SYNC __syncthreads()
#define SINGLE if(threadIdx.x == 0)
#define MULTI(index, length) const auto index = threadIdx.x; if(index < length)
#define LOOP(index, length) for(auto index = threadIdx.x; index < length; index += blockDim.x)
#else
#define SHARED
#define SYNC
#define SINGLE
#define MULTI(index, length) for(auto index = 0u; index < length; index++)
#define LOOP(index, length) for(auto index = 0u; index < length; index++)
#endif
