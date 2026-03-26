#pragma once
#include <array>
#include <numeric>
#include <__ranges/iota_view.h>
#include <concepts>
#if defined(__APPLE__)
#  define PSTLD_HEADER_ONLY
#  define PSTLD_HACK_INTO_STD
#  include "pstld.h"
#else
#  include <algorithm>
#  include <execution>
#endif

/*
 *
 * Several key prerequisites or otherwise upstream functionality:
    * Template metaprogramming used in definition of Tensor in Tensor.hpp
    * Parallel function helper used everywhere
    * Definition of FloatUnaryOp and FloatBinaryOp concepts
 */

namespace TTTN {
    template<typename F>
    concept FloatUnaryOp = std::regular_invocable<F, float> &&
                           std::same_as<std::invoke_result_t<F, float>, float>;

    template<typename F>
    concept FloatBinaryOp = std::regular_invocable<F, float, float> &&
                            std::same_as<std::invoke_result_t<F, float, float>, float>;


    // @doc: template<std::invocable<size_t> F> void ParForEach(size_t n, F f)
    /** Helper to parallel-execute `std::for_each` on a `std::views::iota(size_t{0}, n)`, calling `f` (something `std::invocable` on `size_t`) on each index */
    template<std::invocable<size_t> F>
    void ParForEach(size_t n, F f) {
        auto range = std::ranges::views::iota(size_t{0}, n);
        std::for_each(std::execution::par_unseq, range.begin(), range.end(), f);
    }


    template<size_t... Ds>
    struct TensorDimsProduct;

    // @doc: struct TensorDimsProduct<size_t... Ds>
    /** Template-specialization-based recursion to collapse variadic template `<size_t...Ds>` into single `size_t`, stored statically as `TensorDimsProduct<size_t...Ds>::value` */
    template<>
    struct TensorDimsProduct<> {
        static constexpr size_t value = 1;
    };

    template<size_t First, size_t... Rest>
    struct TensorDimsProduct<First, Rest...> {
        static constexpr size_t value = First * TensorDimsProduct<Rest...>::value;
    };


    template<size_t N, size_t... Ds>
    struct SizeTemplateGet;

    template<size_t First, size_t... Rest>
    struct SizeTemplateGet<0, First, Rest...> {
        static constexpr size_t value = First;
    };

    template<size_t N, size_t First, size_t... Rest>
    struct SizeTemplateGet<N, First, Rest...> {
        // peel off, ditch first, until N = 0
        // @doc: struct SizeTemplateGet<size_t N, size_t... Ds>
        /**
         * Template-specialization-based recursion grab `N`-th `size_t` from `<size_t...Ds>`
         * Uses functional-style aggregation and pattern-matching to decrement `N` and peel off `size_t`s from variadic array until reaching basecase where `N = 0`
         * Used for clean, compile-time syntax in [TensorOps.hpp](src/TensorOps.hpp)
         */
        static constexpr size_t value = SizeTemplateGet<N - 1, Rest...>::value;
    };


    template<size_t... Ds>
    struct ComputeStrides;

    // @doc: struct ComputeStrides<size_t... Ds>
    /**
     * Template-specialization-based recursion to compute `Tensor::Strides` array
     * The `Tensor::Strides` array is vital to mapping from indices into `Tensor::Shape` to flat indices for the backing array
     * In general, for a `Tensor` with `Tensor::Shape = [A, B, ..., N]`, its `Tensor::Strides = [A * B * ... * N, B * ... * N, ..., 1]`
     * Specialize for `<>` and `<size_t D>`
     * `value = ` `[]` and `[1]`, respectively
     * Specialize for `<size_t First, size_t Second, size_t... Rest>`
     * recursively compute `tail = ComputeStrides<Second, Rest...>::value`
     * `value[0] = TensorDimsProduct<Second, Rest...>::value`
     * `value[i] = tail[i + 1]`
     */
    template<>
    struct ComputeStrides<> {
        static constexpr std::array<size_t, 0> value = {};
    };

    template<size_t D>
    struct ComputeStrides<D> {
        static constexpr std::array<size_t, 1> value = {1};
    };

    template<size_t First, size_t Second, size_t... Rest>
    struct ComputeStrides<First, Second, Rest...> {
        static constexpr auto tail = ComputeStrides<Second, Rest...>::value;
        static constexpr size_t N = 1 + tail.size();

        static constexpr std::array<size_t, N> compute() {
            std::array<size_t, N> result{};
            // stride[0] = Second * Rest... = product of everything after first
            result[0] = TensorDimsProduct<Second, Rest...>::value;
            for (size_t i = 0; i < tail.size(); i++) {
                result[i + 1] = tail[i];
            }
            return result;
        }

        static constexpr auto value = compute();
    };
}
