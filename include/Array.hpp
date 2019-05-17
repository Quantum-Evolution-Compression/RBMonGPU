#pragma once

#include "types.h"
#include <vector>
#include <builtin_types.h>

namespace rbm_on_gpu {

using namespace std;


namespace kernel {

template<typename T>
struct Array {
    T* device;
};

} // namespace kernel


template<typename T>
struct Array : public kernel::Array<T> {
    bool        gpu;
    vector<T>   host;

    Array(const size_t& size, const bool gpu);
    ~Array() noexcept(false);

    inline size_t size() const {
        return this->host.size();
    }

    inline T* data() {
        return this->device;
    }

    void clear();
    void update_host();
};

} // namespace rbm_on_gpu
