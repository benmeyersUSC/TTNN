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
    /**
     * Concatenate the dimension packs of two `Tensor` types
     * `TensorConcat<Tensor<A,B>, Tensor<C>>::type == Tensor<A,B,C>`
     */
    template<typename T1, typename T2>
    struct TensorConcat;

    template<size_t... Ds1, size_t... Ds2>
    struct TensorConcat<Tensor<Ds1...>, Tensor<Ds2...> > {
        using type = Tensor<Ds1..., Ds2...>;
    };


    // @doc: struct ArrayToTensor<typename KeptIdxs, typename Iota>
    /**
     * Convert a compile-time `std::array<size_t, N>` (held in `KeptIdxs::value`) into a `Tensor` type
     * `type = Tensor<arr[Iota]...>` where `Iota` is an `index_sequence` over `[0, N)`
     */
    template<typename KeptIdxs, typename Iota>
    struct ArrayToTensor;

    template<typename KeptIdxs, size_t... Iota>
    struct ArrayToTensor<KeptIdxs, std::index_sequence<Iota...> > {
        static constexpr auto arr = KeptIdxs::value;
        using type = Tensor<arr[Iota]...>;
    };


    // @doc: struct KeptDimsHolder<size_t Skip, size_t... Dims>
    /**
     * Compute `Dims...` with the axis at position `Skip` removed
     * `value` holds the resulting `std::array<size_t, sizeof...(Dims) - 1>`
     */
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
    /**
     * `Tensor<Dims...>` with axis `Skip` dropped
     * `RemoveAxis<1, A, B, C>::type == Tensor<A, C>`
     */
    template<size_t Skip, size_t... Dims>
    struct RemoveAxis {
        using type = ArrayToTensor<
            KeptDimsHolder<Skip, Dims...>,
            std::make_index_sequence<sizeof...(Dims) - 1>
        >::type;
    };


    // @doc: struct InsertAxisHolder<size_t Axis, size_t N, size_t... Dims>
    /**
     * Compute `Dims...` with dimension `N` inserted at position `Axis`
     * `value` holds the resulting `std::array<size_t, sizeof...(Dims) + 1>`
     */
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
    /**
     * `Tensor<Dims...>` with dimension `N` inserted at position `Axis`
     * `InsertAxis<1, 4, A, C>::type == Tensor<A, 4, C>`
     */
    template<size_t Axis, size_t N, size_t... Dims>
    struct InsertAxis {
        using type = ArrayToTensor<
            InsertAxisHolder<Axis, N, Dims...>,
            std::make_index_sequence<sizeof...(Dims) + 1>
        >::type;
    };


    // @doc: struct SliceDimsHolder<size_t Start, size_t Len, size_t... Dims>
    /**
     * Extract `Len` contiguous dimensions starting at `Start` from `Dims...`
     * `value` holds the resulting `std::array<size_t, Len>`
     */
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
    /**
     * `Tensor` type formed from dimensions `[Start, Start+Len)` of `Tensor<Dims...>`
     * `TensorSlice<1, 2, A, B, C, D>::type == Tensor<B, C>`
     */
    template<size_t Start, size_t Len, size_t... Dims>
    struct TensorSlice {
        using type = ArrayToTensor<
            SliceDimsHolder<Start, Len, Dims...>,
            std::make_index_sequence<Len>
        >::type;
    };


    // @doc: struct PermutedTensorType<typename T, size_t... Perm>
    /**
     * `Tensor` type with dimensions reordered according to `Perm`
     * `PermutedTensorType<Tensor<4,5,3>, 2,0,1>::type == Tensor<3,4,5>`
     */
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
