#pragma once
#include <cassert>
#include <memory>
#include <new>
#include <numeric>

/*
 *
 * TensorStorage struct to allow for:
    * small Tensor optimization for no heap and
    * 64-aligned heap allocation for vectorization
 *
 * Included by Tensor.hpp
 */

namespace TTTN {
    // @doc: template<size_t S, bool Small = (S <= 16)> struct TensorStorage
    /**
     * Struct wrapper for storage of `Tensor`'s `float[]`
     * `Tensor` owns one instance as `storage_`
     * Specialized for `bool Small = true` (inline 64-byte-aligned stack `float[]`) and `bool Small = false` (64-byte-aligned heap `float[]`)
     */
    template<size_t S, bool Small = (S <= 16)>
    struct TensorStorage;

    // @doc: template<size_t S> struct TensorStorage<S, true>
    /**
     * Specialization for *small `Tensor` optimization* (**STO**)
     * Member array is defined as: `alignas(64) float data[S]{}`
     */
    template<size_t S>
    struct TensorStorage<S, true> {
        alignas(64) float data[S]{};

        float *ptr() { return data; }
        [[nodiscard]] const float *ptr() const { return data; }

        TensorStorage() = default;

        TensorStorage(const TensorStorage &o) noexcept { std::memcpy(data, o.data, S * sizeof(float)); }

        TensorStorage &operator=(const TensorStorage &o) noexcept {
            std::memcpy(data, o.data, S * sizeof(float));
            return *this;
        }

        TensorStorage(TensorStorage &&) noexcept = default;

        TensorStorage &operator=(TensorStorage &&) noexcept = default;
    };

    // @doc: template<size_t S> struct TensorStorage<S, false>
    /**
     * Specialization for larger `Tensor` to be allocated on the heap at a 64-byte aligned address
     * Member array is defined as: `std::unique_ptr<float[], AlignedDeleter> heap_`
     */
    template<size_t S>
    struct TensorStorage<S, false> {
        // need to pass deleter to std::unique_ptr (just using std::free) to clear our floats
        struct AlignedDeleter {
            void operator()(float *p) const noexcept { std::free(p); }
        };

        std::unique_ptr<float[], AlignedDeleter> heap_;

        float *ptr() { return heap_.get(); }
        const float *ptr() const { return heap_.get(); }

        // default constructor calls this
        static float *alloc() {
            // +63 then ~63 mask just rounds you up to the next multiple of 64 that is bigger than your data
            // examples
            //      S = 5, 20 bytes + 63 = 83...83 & ~63 = 64
            //      S = 16, 64 bytes + 63 = 127...127 & ~63 = 64
            //      S = 17, 68 bytes + 63 = 131...131 & ~63 = 128
            constexpr size_t bytes = (S * sizeof(float) + 63) & ~size_t(63); // ~size_t(63) is 0b11000000
            // aligned alloc to 64 byte
            void *p = std::aligned_alloc(64, bytes);
            if (!p) {
                throw std::bad_alloc{};
            }
            // fill bytes-many bytes at p with 0
            std::memset(p, 0, bytes);
            return static_cast<float *>(p);
        }

        TensorStorage() : heap_(alloc()) {
        }

        // copy constructor and assignment just memcpy from behind the back of unique_ptr
        TensorStorage(const TensorStorage &o) : heap_(alloc()) {
            std::memcpy(heap_.get(), o.heap_.get(), S * sizeof(float));
        }

        TensorStorage &operator=(const TensorStorage &o) {
            if (this != &o) std::memcpy(heap_.get(), o.heap_.get(), S * sizeof(float));
            return *this;
        }

        // move is all handled by unique ptr
        TensorStorage(TensorStorage &&) noexcept = default;

        TensorStorage &operator=(TensorStorage &&) noexcept = default;
    };
}
