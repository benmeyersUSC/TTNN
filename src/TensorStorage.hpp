#pragma once
#include <array>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <new>
#include <numeric>
#include <type_traits>

/*
 *
 * TensorStorage struct to allow for:
    * small Tensor optimization for no heap and
    * 64-aligned heap allocation for vectorization
 *
 * Included by Tensor.hpp
 */

namespace TTTN {
    template<size_t S, bool Small = (S <= 16)>
    struct TensorStorage;

    template<size_t S>
    struct TensorStorage<S, true> {
        alignas(64) float data[S]{};

        float *ptr() { return data; }
        [[nodiscard]] const float *ptr() const { return data; }

        TensorStorage() = default;

        // @doc: struct TensorStorage<size_t S, bool Small>
        /** ######### */
        TensorStorage(const TensorStorage &o) noexcept { std::memcpy(data, o.data, S * sizeof(float)); }

        TensorStorage &operator=(const TensorStorage &o) noexcept {
            std::memcpy(data, o.data, S * sizeof(float));
            return *this;
        }

        TensorStorage(TensorStorage &&) noexcept = default;

        TensorStorage &operator=(TensorStorage &&) noexcept = default;
    };

    template<size_t S>
    struct TensorStorage<S, false> {
        struct AlignedDeleter {
            void operator()(float *p) const noexcept { std::free(p); }
        };

        std::unique_ptr<float[], AlignedDeleter> heap_;

        float *ptr() { return heap_.get(); }
        const float *ptr() const { return heap_.get(); }

        static float *alloc() {
            constexpr size_t bytes = (S * sizeof(float) + 63) & ~size_t(63);
            void *p = std::aligned_alloc(64, bytes);
            if (!p) throw std::bad_alloc{};
            std::memset(p, 0, bytes);
            return static_cast<float *>(p);
        }

        TensorStorage() : heap_(alloc()) {
        }

        TensorStorage(const TensorStorage &o) : heap_(alloc()) {
            std::memcpy(heap_.get(), o.heap_.get(), S * sizeof(float));
        }

        TensorStorage &operator=(const TensorStorage &o) {
            if (this != &o) std::memcpy(heap_.get(), o.heap_.get(), S * sizeof(float));
            return *this;
        }

        TensorStorage(TensorStorage &&) noexcept = default;

        TensorStorage &operator=(TensorStorage &&) noexcept = default;
    };
}
