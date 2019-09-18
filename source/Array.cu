#include "Spins.h"
#include "Array.hpp"
#include <algorithm>


using namespace std;

namespace rbm_on_gpu {

template<typename T>
Array<T>::Array(const size_t& size, const bool gpu) : vector<T>(size), gpu(gpu) {
    if(gpu) {
        MALLOC(this->device, sizeof(T) * size, true);
    }
}

template<typename T>
Array<T>::Array(const Array<T>& other)
    : vector<T>(other), gpu(other.gpu)
{
    if(gpu) {
        MALLOC(this->device, sizeof(T) * this->size(), true);
        MEMCPY(this->device, other.device, sizeof(T) * this->size(), true, true);
    }
}

template<typename T>
Array<T>::Array(Array<T>&& other)
    : vector<T>(other), gpu(other.gpu)
{
    this->device = other.device;
    other.device = nullptr;
}

template<typename T>
Array<T>::~Array() noexcept(false) {
    if(this->gpu) {
        FREE(this->device, true);
    }
}

template<typename T>
Array<T>& Array<T>::operator=(const Array& other) {
    std::vector<T>::operator=(other);
    this->gpu = other.gpu;
    this->update_device();
    return *this;
}

template<typename T>
Array<T>& Array<T>::operator=(Array&& other) {
    std::vector<T>::operator=(other);
    if(this->gpu) {
        FREE(this->device, true);
    }
    this->gpu = other.gpu;
    if(this->gpu) {
        this->device = other.device;
        other.device = nullptr;
    }

    this->update_device();
    return *this;
}

template<typename T>
void Array<T>::clear() {
    if(this->gpu) {
        MEMSET(this->device, 0, sizeof(T) * this->size(), this->gpu);
    }
    fill(this->begin(), this->end(), 0);
}

template<typename T>
void Array<T>::update_host() {
    if(this->gpu) {
        MEMCPY_TO_HOST(this->host_data(), this->device, sizeof(T) * this->size(), true);
    }
}

template<typename T>
void Array<T>::update_device() {
    if(this->gpu) {
        MEMCPY(this->device, this->host_data(), sizeof(T) * this->size(), true, false);
    }
}


template class Array<unsigned int>;
template class Array<float>;
template class Array<complex_t>;
template class Array<Spins>;

} // namespace rbm_on_gpu
