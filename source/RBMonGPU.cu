#include "types.h"

namespace rbm_on_gpu {

void setDevice(int device) {
    cudaSetDevice(device);
}

} // namespace rbm_on_gpu
