#pragma once
#include "Tensor.hpp"
#include <array>


/*
 *
 * Operations on Tensor shapes...nothing about values, only getting types!
 *
 */


namespace TTTN {
    // =========================================================================
    // Tensor Type Algebra
    // Shape-only metaprogramming: no data, no runtime — purely type-level
    // =========================================================================

    // @doc: struct TensorConcat<typename T1, typename T2>
    /** ######### */
    template<typename T1, typename T2>
    struct TensorConcat;

    template<size_t... Ds1, size_t... Ds2>
    struct TensorConcat<Tensor<Ds1...>, Tensor<Ds2...> > {
        using type = Tensor<Ds1..., Ds2...>;
    };


    // @doc: struct ArrayToTensor<typename KeptIdxs, typename Iota>
    /** ######### */
    template<typename KeptIdxs, typename Iota>
    struct ArrayToTensor;

    template<typename KeptIdxs, size_t... Iota>
    struct ArrayToTensor<KeptIdxs, std::index_sequence<Iota...> > {
        static constexpr auto arr = KeptIdxs::value;
        using type = Tensor<arr[Iota]...>;
    };


    // @doc: struct KeptDimsHolder<size_t Skip, size_t... Dims>
    /** ######### */
    template<size_t Skip, size_t... Dims>
    struct KeptDimsHolder {
        static constexpr auto value = [] {
            constexpr std::array<size_t, sizeof...(Dims)> all = {Dims...};
            std::array<size_t, sizeof...(Dims) - 1> result{};
            size_t out = 0;
            for (size_t i = 0; i < sizeof...(Dims); ++i)
                if (i != Skip) result[out++] = all[i];
            return result;
        }();
    };

    // @doc: struct RemoveAxis<size_t Skip, size_t... Dims>
    /** ######### */
    template<size_t Skip, size_t... Dims>
    struct RemoveAxis {
        using type = ArrayToTensor<
            KeptDimsHolder<Skip, Dims...>,
            std::make_index_sequence<sizeof...(Dims) - 1>
        >::type;
    };


    // @doc: struct InsertAxisHolder<size_t Axis, size_t N, size_t... Dims>
    /** ######### */
    template<size_t Axis, size_t N, size_t... Dims>
    struct InsertAxisHolder {
        static constexpr auto value = [] {
            constexpr std::array<size_t, sizeof...(Dims)> all = {Dims...};
            std::array<size_t, sizeof...(Dims) + 1> result{};
            for (size_t i = 0; i < Axis; ++i) result[i] = all[i];
            result[Axis] = N;
            for (size_t i = Axis; i < sizeof...(Dims); ++i) result[i + 1] = all[i];
            return result;
        }();
    };

    // @doc: struct InsertAxis<size_t Axis, size_t N, size_t... Dims>
    /** ######### */
    template<size_t Axis, size_t N, size_t... Dims>
    struct InsertAxis {
        using type = ArrayToTensor<
            InsertAxisHolder<Axis, N, Dims...>,
            std::make_index_sequence<sizeof...(Dims) + 1>
        >::type;
    };


    // @doc: struct SliceDimsHolder<size_t Start, size_t Len, size_t... Dims>
    /** ######### */
    template<size_t Start, size_t Len, size_t... Dims>
    struct SliceDimsHolder {
        static_assert(Start + Len <= sizeof...(Dims), "slice out of range");
        static constexpr auto value = [] {
            constexpr std::array<size_t, sizeof...(Dims)> all = {Dims...};
            std::array<size_t, Len> result{};
            for (size_t i = 0; i < Len; ++i) result[i] = all[Start + i];
            return result;
        }();
    };

    // @doc: struct TensorSlice<size_t Start, size_t Len, size_t... Dims>
    /** ######### */
    template<size_t Start, size_t Len, size_t... Dims>
    struct TensorSlice {
        using type = ArrayToTensor<
            SliceDimsHolder<Start, Len, Dims...>,
            std::make_index_sequence<Len>
        >::type;
    };


    // @doc: struct PermutedTensorType<typename T, size_t... Perm>
    /** ######### */
    template<typename T, size_t... Perm>
    struct PermutedTensorType;

    template<size_t... Dims, size_t... Perm>
    struct PermutedTensorType<Tensor<Dims...>, Perm...> {
        static_assert(sizeof...(Dims) == sizeof...(Perm),
                      "permutation length must match tensor rank");
        static constexpr std::array<size_t, sizeof...(Dims)> shape = {Dims...};
        using type = Tensor<shape[Perm]...>;
    };
}
