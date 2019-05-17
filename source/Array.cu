#include "Array.hpp"

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
    MEMSET(this->data(), 0, sizeof(T) * this->size(), this->gpu);
}

template<typename T>
void Array<T>::update_host() {
    if(this->gpu) {
        MEMCPY_TO_HOST(this->host.data(), this->data(), sizeof(T) * this->size(), true);
    }
}


template class Array<double>;
template class Array<complex_t>;

} // namespace rbm_on_gpu
