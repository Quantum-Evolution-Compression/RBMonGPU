#ifdef __CUDA_ARCH__

#define SHARED __shared__
#define REGISTER(name, n) name
#define SYNC __syncthreads()
#define SINGLE if(threadIdx.x == 0)
#define MULTI(index, length) const auto index = threadIdx.x; if(index < length)
#define LOOP(index, length) for(auto index = threadIdx.x; index < length; index += blockDim.x)
#define SHARED_MEM_LOOP_BEGIN(index, length) \
    __shared__ unsigned int index; if(threadIdx.x == 0) index = 0u; __syncthreads(); \
    while(index < (length))
#define SHARED_MEM_LOOP_END(index) if(threadIdx.x == 0u) index++; __syncthreads();

#else

#define SHARED
#define REGISTER(name, n) name[n]
#define SYNC
#define SINGLE
#define MULTI(index, length) for(auto index = 0u; index < length; index++)
#define LOOP(index, length) for(auto index = 0u; index < length; index++)
#define SHARED_MEM_LOOP_BEGIN(index, length) for(auto index = 0u; index < (length); index++)
#define SHARED_MEM_LOOP_END(index)

#endif
