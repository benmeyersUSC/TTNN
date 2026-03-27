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
    // @doc: template<typename F> concept FloatUnaryOp
    /** Concept to enforce `F :: float -> float` operations on `Tensor`s */
    template<typename F>
    concept FloatUnaryOp = std::regular_invocable<F, float> &&
                           std::same_as<std::invoke_result_t<F, float>, float>;
    // @doc: template<typename F> concept FloatBinaryOp
    /** Concept to enforce `F :: float -> float -> float` operations on two `Tensor`s */
    template<typename F>
    concept FloatBinaryOp = std::regular_invocable<F, float, float> &&
                            std::same_as<std::invoke_result_t<F, float, float>, float>;


    // @doc: template<std::invocable<size_t> F> void ParForEach(size_t n, F f)
    /** Helper function used throughout library to run `std::invocable<size_t> F` `n` times using `std::execution::par_unseq` policy */
    template<std::invocable<size_t> F>
    void ParForEach(size_t n, F f) {
        auto range = std::ranges::views::iota(size_t{0}, n);
        std::for_each(std::execution::par_unseq, range.begin(), range.end(), f);
    }

    // @doc: struct TensorDimsProduct<size_t... Ds>
    /** Template recursion to store product of `size_t...` variadic in `static constexpr size_t value` */
    template<size_t... Ds>
    struct TensorDimsProduct;

    template<>
    struct TensorDimsProduct<> {
        static constexpr size_t value = 1;
    };

    template<size_t First, size_t... Rest>
    struct TensorDimsProduct<First, Rest...> {
        static constexpr size_t value = First * TensorDimsProduct<Rest...>::value;
    };


    // @doc: struct SizeTemplateGet<size_t N, size_t... Ds>
    /** Template recursion to store `N`-th element of `size_t...` variadic in `static constexpr size_t value` */
    template<size_t N, size_t... Ds>
    struct SizeTemplateGet;

    template<size_t First, size_t... Rest>
    struct SizeTemplateGet<0, First, Rest...> {
        static constexpr size_t value = First;
    };

    template<size_t N, size_t First, size_t... Rest>
    struct SizeTemplateGet<N, First, Rest...> {
        // peel off, ditch first, until N = 0
        static constexpr size_t value = SizeTemplateGet<N - 1, Rest...>::value;
    };

    // @doc: struct ComputeStrides<size_t... Ds>
    /** Template recursion to store `Tensor<Ds...>` stride array in `static constexpr std::array<size_t, sizeof...(Ds)> value` */
    template<size_t... Ds>
    struct ComputeStrides;

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
