#pragma once
#include "Tensor.hpp"
#include <array>


/*
 *
 * Operations on Tensor shapes...nothing about values, only getting types!
 *
 */


namespace TTTN {
    template<typename T1, typename T2>
    struct TensorConcat;

    // @doc: template<size_t... Ds1, size_t... Ds2> struct TensorConcat<Tensor<Ds1...>, Tensor<Ds2...> >
    /**
     * One layer unpacking of two `Tensor`s' dimensions
     * Two variadic lists as template parameters match with the shape arrays of the two input `Tensor`s
     */
    template<size_t... Ds1, size_t... Ds2>
    struct TensorConcat<Tensor<Ds1...>, Tensor<Ds2...> > {
        using type = Tensor<Ds1..., Ds2...>;
    };

    // @doc: template<size_t Skip, size_t... Dims> struct KeptDimsHolder
    /**
     * Helper for `RemoveAxis`
     * Takes a dimension/axis to skip and variadic `size_t...` for existing dims and defines `std::array<size_t, sizeof...(Dims)> value` filled with `Dims...` sans `Skip`
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


    template<typename KeptIdxs, typename Iota>
    struct ArrayToTensor;;

    // @doc: template<typename KeptIdxs, size_t... Iota> requires requires { KeptIdxs::value; } struct ArrayToTensor<KeptIdxs, std::index_sequence<Iota...>
    /** Take in a holder of a `size_t...` dimensions pack (which must have static `value`) and a `std::index_sequence` of the `Rank` of the `Tensor`-to-be and unpack dimensions into new `Tensor` type */
    template<typename KeptIdxs, size_t... Iota>
        requires requires { KeptIdxs::value; }
    struct ArrayToTensor<KeptIdxs, std::index_sequence<Iota...> > {
        static constexpr auto arr = KeptIdxs::value;
        using type = Tensor<arr[Iota]...>;
    };


    // @doc: struct RemoveAxis<size_t Skip, size_t... Dims>
    /**
     * Helper to create new `Tensor` type from given `size_t...Dims` and an axis to `Skip`
     * Calls `ArrayToTensor<KeptDimsHolder<...>>`
     */
    template<size_t Skip, size_t... Dims>
    struct RemoveAxis {
        using type = ArrayToTensor<
            KeptDimsHolder<Skip, Dims...>,
            std::make_index_sequence<sizeof...(Dims) - 1>
        >::type;
    };


    // @doc: template<size_t Axis, size_t N, size_t... Dims> struct InsertAxisHolder
    /**
     * Like `KeptDimsHolder`, a helper holder of an array of `size_t` dimensions to be used to construct a new `Tensor` type
     * Used in `InsertAxis`
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


    // @doc: template<size_t Axis, size_t N, size_t... Dims> struct InsertAxis
    /**
     * Like `RemoveAxis`, insert a dimension at a specified `size_t Axis`, with a specified magnitude `size_t N`
     * Calls `ArrayToTensor<InsertAxisHolder<...>>`
     */
    template<size_t Axis, size_t N, size_t... Dims>
    struct InsertAxis {
        using type = ArrayToTensor<
            InsertAxisHolder<Axis, N, Dims...>,
            std::make_index_sequence<sizeof...(Dims) + 1>
        >::type;
    };


    // @doc: template<size_t Start, size_t Len, size_t... Dims> struct SliceDimsHolder
    /**
     * Like `InsertAxisHolder` and `KeptDimsHolder`, helper class to define an array of `size_t`, of length `Len`, containing the axes from `size_t...Dims` that start at `size_t Start`
     * Used in `TensorSlice`
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


    // @doc: template<size_t Start, size_t Len, size_t... Dims> struct TensorSlice
    /**
     * Specify `size_t Start`, `size_t Len`, and a starting set of `size_t...Dims` and construct a `Tensor` whose shape only contains the `Len` dimensions starting at `Start`
     * Calls `ArrayToTensor<SliceDimsHolder<...>>`
     */
    template<size_t Start, size_t Len, size_t... Dims>
    struct TensorSlice {
        using type = ArrayToTensor<
            SliceDimsHolder<Start, Len, Dims...>,
            std::make_index_sequence<Len>
        >::type;
    };


    template<typename T, size_t... Perm>
    struct PermutedTensorType;


    // @doc: template<size_t... Dims, size_t... Perm> struct PermutedTensorType<Tensor<Dims...>, Perm...>
    /** Helper to define a `Tensor` type, whose shape is `size_t...Dims` reorganized according to the indices specified by `size_t...Perm` */
    template<size_t... Dims, size_t... Perm>
    struct PermutedTensorType<Tensor<Dims...>, Perm...> {
        static_assert(sizeof...(Dims) == sizeof...(Perm),
                      "permutation length must match tensor rank");
        static constexpr std::array<size_t, sizeof...(Dims)> shape = {Dims...};
        using type = Tensor<shape[Perm]...>;
    };
}
