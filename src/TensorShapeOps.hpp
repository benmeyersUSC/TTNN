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


    // @doc: template<size_t N_out, size_t N_in> struct SwapNDims
    /** Define permutation-ready `std::array<size_t, N_out + N_in>` that moves the final `N_out` axes to the front and the first `N_in` axes after */
    template<size_t N_out, size_t N_in>
    struct SwapNDims {
        static constexpr auto value = [] {
            std::array<size_t, N_out + N_in> p{};
            for (size_t i = 0; i < N_in; ++i) p[i] = N_out + i;
            for (size_t i = 0; i < N_out; ++i) p[N_in + i] = i;
            return p;
        }();
    };


    template<size_t Batch, typename T>
    struct PrependBatch;

    // @doc: template<size_t Batch, size_t... Dims> struct PrependBatch
    /** Prepend a `Batch` axis, define `type = Tensor<Batch, Dims...>` */
    template<size_t Batch, size_t... Dims>
    struct PrependBatch<Batch, Tensor<Dims...> > {
        using type = Tensor<Batch, Dims...>;
    };

    template<typename T>
    struct TensorFirstDim;

    // @doc: template<size_t D0, size_t... Rest> struct TensorFirstDim<Tensor<D0, Rest...> >
    /** Get the first axis size from a `Tensor` */
    template<size_t D0, size_t... Rest>
    struct TensorFirstDim<Tensor<D0, Rest...> > {
        static constexpr size_t value = D0;
    };

    template<typename A, typename B>
    struct ConcatSeqs;

    // @doc: template<size_t... As, size_t... Bs> struct ConcatSeqs<std::integer_sequence<size_t, As...>, std::integer_sequence<size_t, Bs...> >
    /** Given two `std::integer_sequence`s, concatenate their values into one `std::integer_sequence` */
    template<size_t... As, size_t... Bs>
    struct ConcatSeqs<std::integer_sequence<size_t, As...>, std::integer_sequence<size_t, Bs...> > {
        using type = std::integer_sequence<size_t, As..., Bs...>;
    };


    template<typename Seq>
    struct SeqToTensor;

    // @doc: template<size_t... Ds> struct SeqToTensor<std::integer_sequence<size_t, Ds...> >
    /** Given a `std::integer_sequence`, forward its values to a new `Tensor` type */
    template<size_t... Ds>
    struct SeqToTensor<std::integer_sequence<size_t, Ds...> > {
        using type = Tensor<Ds...>;
    };


    template<typename Seq>
    struct SeqProduct;

    // @doc: template<> struct SeqProduct<std::integer_sequence<size_t> >
    /**
     * Fold a `std::integer_sequence` with multiplication to compute the product of all values
     * Base case for empty sequence: `value = 1`
     */
    template<>
    struct SeqProduct<std::integer_sequence<size_t> > {
        static constexpr size_t value = 1;
    };

    // @doc: template<size_t D0, size_t... Rest> struct SeqProduct<std::integer_sequence<size_t, D0, Rest...> >
    /**
     * Fold a `std::integer_sequence` with multiplication to compute the product of all values
     * Recursive case: `value = D0 * SeqProduct<std::integer_sequence<size_t, Rest...> >::value`
     */
    template<size_t D0, size_t... Rest>
    struct SeqProduct<std::integer_sequence<size_t, D0, Rest...> > {
        static constexpr size_t value = D0 * SeqProduct<std::integer_sequence<size_t, Rest...> >::value;
    };


    template<size_t N, typename Collected, typename Remaining>
    struct SplitAtImpl;

    // @doc: template<size_t... Collected, size_t... Remaining> struct SplitAtImpl<0, std::integer_sequence<size_t, Collected...>, std::integer_sequence<size_t, Remaining...> >
    /**
     * Recursive implementation of `SplitAt`
     * Base case, `N == 0`: `Collected...` and `Rest...` are forwarded into `std::integer_sequence head, tail`
     */
    template<size_t... Collected, size_t... Remaining>
    struct SplitAtImpl<0, std::integer_sequence<size_t, Collected...>, std::integer_sequence<size_t, Remaining...> > {
        using head = std::integer_sequence<size_t, Collected...>;
        using tail = std::integer_sequence<size_t, Remaining...>;
    };

    // @doc: template<size_t N, size_t... Collected, size_t D0, size_t... Rest> requires (N > 0) struct SplitAtImpl<N, std::integer_sequence<size_t, Collected...>, std::integer_sequence<size_t, D0, Rest...> >
    /**
     * Recursive implementation of `SplitAt`
     * Recursive case: peel off leftmost value of `Rest...`, append to `Collected...`, call on `N - 1`
     */
    template<size_t N, size_t... Collected, size_t D0, size_t... Rest> requires (N > 0)
    struct SplitAtImpl<N, std::integer_sequence<size_t, Collected...>, std::integer_sequence<size_t, D0, Rest...> > {
        // peel leftmost value from Rest..., add to Collected, N--
        using next = SplitAtImpl<N - 1,
            std::integer_sequence<size_t, Collected..., D0>,
            std::integer_sequence<size_t, Rest...> >;
        using head = next::head;
        using tail = next::tail;
    };

    // @doc: template<size_t N, size_t... Dims> struct SplitAt
    /**
     * Given axes `size_t... Dims` and `size_t N`, internally store `head` and `tail` `std::integer_sequence`s, where `head` contains the first `N` values of `Dims...` and `tail` the rest
     * Useful to define weight `Tensor`
     */
    template<size_t N, size_t... Dims>
    struct SplitAt {
        using impl = SplitAtImpl<N,
            std::integer_sequence<size_t>,
            std::integer_sequence<size_t, Dims...> >;
        using head = impl::head;
        using tail = impl::tail;
    };


    template<size_t N, typename T>
    struct PrependOnes;

    // @doc: template<size_t... Dims> struct PrependOnes<0, Tensor<Dims...> >
    /**
     * Prepend `N` `1`s to a `size_t...Dims`, creating a new `Tensor` type
     * Base case: `type = Tensor<Dims...>`
     */
    template<size_t... Dims>
    struct PrependOnes<0, Tensor<Dims...> > {
        using type = Tensor<Dims...>;
    };

    // @doc: template<size_t N, size_t... Dims> struct PrependOnes<N, Tensor<Dims...> >
    /**
     * Prepend `N` `1`s to a `size_t...Dims`, creating a new `Tensor` type
     * Recursive case: `type = PrependOnes<N - 1, Tensor<1, Dims...> >::type`
     */
    template<size_t N, size_t... Dims>
    struct PrependOnes<N, Tensor<Dims...> > {
        using type = PrependOnes<N - 1, Tensor<1, Dims...> >::type;
    };


    template<size_t Axis, size_t NewVal, typename T>
    struct ReplaceAxisDims;

    // @doc: template<size_t Axis, size_t NewVal, size_t... Dims> struct ReplaceAxisDims<Axis, NewVal, Tensor<Dims...>>
    /** Helper to define a `Tensor` type, whose shape is `size_t...Dims` reorganized according to the indices specified by `size_t...Perm` */
    template<size_t Axis, size_t NewVal, size_t... Dims>
    struct ReplaceAxisDims<Axis, NewVal, Tensor<Dims...>> {
        static_assert(Axis < sizeof...(Dims), "Axis out of range");
        struct Holder {
            static constexpr auto value = [] {
                std::array<size_t, sizeof...(Dims)> r = {Dims...};
                r[Axis] = NewVal;
                return r;
            }();
        };
        using type = ArrayToTensor<Holder, std::make_index_sequence<sizeof...(Dims)>>::type;
    };
}
