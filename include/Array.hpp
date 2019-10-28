#pragma once

#include "types.h"
#include <vector>
#include <builtin_types.h>
#include <type_traits>

#ifdef __PYTHONCC__
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>
#endif // __PYTHONCC__

namespace rbm_on_gpu {

using namespace std;


namespace detail {

template<typename T>
struct std_type {
    using type = T;
};

template<>
struct std_type<complex_t> {
    using type = std::complex<double>;
};

} // namespace detail


namespace kernel {

template<typename T>
struct Array {
    T* device;

    HDINLINE T* data() {
        return this->device;
    }

    HDINLINE const T* data() const {
        return this->device;
    }
};

} // namespace kernel


template<typename T>
struct Array : public vector<T>, public kernel::Array<T> {
    bool gpu;

    Array(const size_t& size, const bool gpu);
    Array(const Array<T>& other);
    Array(Array<T>&& other);
    ~Array() noexcept(false);

    inline T* data() {
        if(this->gpu) {
            return this->device;
        }
        else {
            return this->host_data();
        }
    }

    inline const T* data() const {
        if(this->gpu) {
            return this->device;
        }
        else {
            return this->host_data();
        }
    }

    inline T* host_data() {
        return vector<T>::data();
    }

    inline const T* host_data() const {
        return vector<T>::data();
    }

    Array<T>& operator=(const Array<T>& other);
    Array<T>& operator=(Array<T>&& other);

    void clear();
    void update_host();
    void update_device();

#ifdef __PYTHONCC__
    template<long unsigned int dim>
    inline Array<T>& operator=(const xt::pytensor<std::complex<double>, dim>& python_vec) {
        static_assert(is_same<T, complex_t>::value);
        memcpy(this->host_data(), python_vec.data(), sizeof(std::complex<double>) * this->size());
        this->update_device();
        return *this;
    }

    template<long unsigned int dim>
    inline Array<T>(const xt::pytensor<std::complex<double>, dim>& python_vec, const bool gpu) : Array<T>(python_vec.size(), gpu) {
        (*this) = python_vec;
    }

    template<unsigned int dim>
    inline xt::pytensor<typename detail::std_type<T>::type, dim> to_pytensor(shape_t<dim> shape={}) const {
        if(shape == shape_t<dim>()) {
            shape[0] = (long int)this->size();
        }

        xt::pytensor<typename detail::std_type<T>::type, dim> result(shape);
        memcpy(result.data(), this->host_data(), sizeof(T) * this->size());
        return result;
    }
#endif // __PYTHONCC__
};

} // namespace rbm_on_gpu
