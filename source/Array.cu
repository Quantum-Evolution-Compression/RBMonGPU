#include "Spins.h"
#include "Array.hpp"
#include <algorithm>


using namespace std;

namespace rbm_on_gpu {

template<typename T>
Array<T>::Array(const size_t& size, const bool gpu) : gpu(gpu), host(size) {
    if(gpu) {
        MALLOC(this->device, sizeof(T) * size, true);
    }
    else {
        this->device = this->host.data();
    }
}

template<typename T>
Array<T>::~Array() noexcept(false) {
    if(this->gpu) {
        FREE(this->device, true);
    }
}

template<typename T>
void Array<T>::clear() {
    if(this->gpu) {
        MEMSET(this->device, 0, sizeof(T) * this->size(), this->gpu);
    }
    else{
        fill(this->host.begin(), this->host.end(), 0);
    }
}

template<typename T>
void Array<T>::update_host() {
    if(this->gpu) {
        MEMCPY_TO_HOST(this->host.data(), this->device, sizeof(T) * this->size(), true);
    }
}

template<typename T>
void Array<T>::update_device() {
    if(this->gpu) {
        MEMCPY(this->device, this->host.data(), sizeof(T) * this->size(), true, false);
    }
}


template class Array<double>;
template class Array<complex_t>;
template class Array<Spins>;

} // namespace rbm_on_gpu
