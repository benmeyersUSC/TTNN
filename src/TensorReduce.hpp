#pragma once
#include "TensorOps.hpp"

namespace TTTN {
    // @doc: template<size_t Axis, size_t... Dims> struct ReduceKernel
    /**
     * Shared kernel for all axis-reduction and broadcast operations
     * Compile-time computes convenient types/shapes, values, and `constexpr` functions for `offset` and `base` flat indexing required in `Broadcast` or `Reduce` functions
     */
    template<size_t Axis, size_t... Dims>
    struct ReduceKernel {
        using Source = Tensor<Dims...>;
        using Result = typename RemoveAxis<Axis, Dims...>::type;
        static constexpr size_t axis_dim = SizeTemplateGet<Axis, Dims...>::value;
        static constexpr size_t axis_stride = Source::Strides[Axis];

        // base(out_i) = flat index in Source with axis set to 0
        // closed form: axis_dim * (out_i rounded down to axis_stride boundary) + remainder
        static constexpr size_t base(size_t out_i) {
            return axis_dim * (out_i - out_i % axis_stride) + out_i % axis_stride;
        }

        static constexpr size_t project(size_t i) {
            const size_t after = i % axis_stride;
            const size_t d_axis = (i / axis_stride) % axis_dim;
            const size_t before = i - after - d_axis * axis_stride;
            return before / axis_dim + after;
        }
    };

    // @doc: template<size_t Axis, typename Op, size_t... Dims> requires FloatBinaryOp<Op> && std::default_initializable<Op> && requires { { Op::identity } -> std::convertible_to<float>; } RemoveAxis<Axis, Dims...>::type Reduce(const Tensor<Dims...> &src)
    /** `Reduce` a `Tensor` along some `Axis` using a `FloatBinaryOp` */
    template<size_t Axis, typename Op, size_t... Dims> requires
        FloatBinaryOp<Op> && std::default_initializable<Op> && requires
        {
            { Op::identity } -> std::convertible_to<float>;
        }
    RemoveAxis<Axis, Dims...>::type Reduce(const Tensor<Dims...> &src) {
        using K = ReduceKernel<Axis, Dims...>;
        auto k_range = std::views::iota(size_t{0}, K::axis_dim);
        typename K::Result dst;
        ParForEach(K::Result::Size, [&](size_t out_i) {
            dst.flat(out_i) = std::transform_reduce(
                std::execution::unseq,
                k_range.begin(), k_range.end(),
                Op::identity, Op{},
                [&](size_t k) { return src.flat(K::base(out_i) + k * K::axis_stride); });
        });
        return dst;
    }

    // @doc: template<size_t Axis, size_t N, size_t... Dims> InsertAxis<Axis, N, Dims...>::type Expand(const Tensor<Dims...> &src)
    /**
     * `Expand` a `Tensor`, copying `N` times over the `Axis` passed as a template argument
     * Identity `Broadcast`
     */
    template<size_t Axis, size_t N, size_t... Dims>
    InsertAxis<Axis, N, Dims...>::type Expand(const Tensor<Dims...> &src) {
        using Full = typename InsertAxis<Axis, N, Dims...>::type;
        return [&]<size_t... FullDims>(std::type_identity<Tensor<FullDims...> >) {
            using K = ReduceKernel<Axis, FullDims...>;
            Tensor<FullDims...> result;
            ParForEach(Full::Size, [&](size_t i) {
                result.flat(i) = src.flat(K::project(i));
            });
            return result;
        }(std::type_identity<Full>{});
    }

    // @doc: template<size_t Axis, typename Op, size_t... Dims> requires FloatBinaryOp<Op> && std::default_initializable<Op> Tensor<Dims...> BroadcastMap(const Tensor<Dims...> &A, const typename RemoveAxis<Axis, Dims...>::type &b)
    /**
     * `Broadcast` a `Tensor` of type `RemoveAxis<Axis, Dims...>` across a specified `Axis` of `Tensor<Dims...> A` using specified `FloatBinaryOp`
     * Copies `A` and returns copy
     */
    template<size_t Axis, typename Op, size_t... Dims> requires FloatBinaryOp<Op> && std::default_initializable<Op>
    Tensor<Dims...> BroadcastMap(const Tensor<Dims...> &A, const typename RemoveAxis<Axis, Dims...>::type &b) {
        using K = ReduceKernel<Axis, Dims...>;
        Tensor<Dims...> result;
        ParForEach(Tensor<Dims...>::Size, [&](size_t i) {
            result.flat(i) = Op{}(A.flat(i), b.flat(K::project(i)));
        });
        return result;
    }


    // @doc: template<size_t Axis, typename Op, size_t... Dims> requires FloatBinaryOp<Op> && std::default_initializable<Op> Tensor<Dims...> BroadcastMapMove(Tensor<Dims...> &&A, const typename RemoveAxis<Axis, Dims...>::type &b)
    /**
     * `Broadcast` a `Tensor` of type `RemoveAxis<Axis, Dims...>` across a specified `Axis` of `Tensor<Dims...> A` using specified `FloatBinaryOp`
     * Moves `A`, overwrites its data, returns moved/overwritten `A`
     */
    template<size_t Axis, typename Op, size_t... Dims> requires FloatBinaryOp<Op> && std::default_initializable<Op>
    Tensor<Dims...> BroadcastMapMove(Tensor<Dims...> &&A, const typename RemoveAxis<Axis, Dims...>::type &b) {
        using K = ReduceKernel<Axis, Dims...>;
        ParForEach(Tensor<Dims...>::Size, [&](size_t i) {
            A.flat(i) = Op{}(A.flat(i), b.flat(K::project(i)));
        });
        return std::move(A);
    }

    // @doc: template<size_t Axis, typename Op, size_t... Dims> requires FloatBinaryOp<Op> && std::default_initializable<Op> void BroadcastApply(Tensor<Dims...> &A, const typename RemoveAxis<Axis, Dims...>::type &b)
    /**
     * `Broadcast` a `Tensor` of type `RemoveAxis<Axis, Dims...>` across a specified `Axis` of `Tensor<Dims...> A` using specified `FloatBinaryOp`
     * Overwrites `A` inplace, no return
     */
    template<size_t Axis, typename Op, size_t... Dims> requires FloatBinaryOp<Op> && std::default_initializable<Op>
    void BroadcastApply(Tensor<Dims...> &A, const typename RemoveAxis<Axis, Dims...>::type &b) {
        using K = ReduceKernel<Axis, Dims...>;
        ParForEach(Tensor<Dims...>::Size, [&](size_t i) {
            A.flat(i) = Op{}(A.flat(i), b.flat(K::project(i)));
        });
    }


    // @doc: template<size_t Axis, typename ApplyOp, typename ReduceOp, size_t... Dims> requires FloatBinaryOp<ApplyOp> && FloatBinaryOp<ReduceOp> && std::default_initializable<ApplyOp> && std::default_initializable<ReduceOp> && requires { { ReduceOp::identity } -> std::convertible_to<float>; } Tensor<Dims...> BroadcastReduce(const Tensor<Dims...> &src)
    /**
     * Fused composition of `Broadcast` and `Reduce`; "`Broadcast` after `Reduce`"
     * `Reduce` with `ReduceOp`, then `Broadcast` that result back onto `Tensor<Dims...> src`
     * Copies `Tensor<Dims...> src` and returns copy
     */
    template<size_t Axis, typename ApplyOp, typename ReduceOp, size_t... Dims> requires
        FloatBinaryOp<ApplyOp> && FloatBinaryOp<ReduceOp> && std::default_initializable<ApplyOp> &&
        std::default_initializable<ReduceOp> && requires { { ReduceOp::identity } -> std::convertible_to<float>; }
    Tensor<Dims...> BroadcastReduce(const Tensor<Dims...> &src) {
        return BroadcastMap<Axis, ApplyOp>(src, Reduce<Axis, ReduceOp>(src));
    }

    // @doc: template<size_t Axis, typename ApplyOp, typename ReduceOp, size_t... Dims> requires FloatBinaryOp<ApplyOp> && FloatBinaryOp<ReduceOp> && std::default_initializable<ApplyOp> && std::default_initializable<ReduceOp> && requires { { ReduceOp::identity } -> std::convertible_to<float>; } Tensor<Dims...> BroadcastReduceMove(Tensor<Dims...> &&src)
    /**
     * Fused composition of `Broadcast` and `Reduce`; "`Broadcast` after `Reduce`"
     * `Reduce` with `ReduceOp`, then `Broadcast` that result back onto `Tensor<Dims...> src`
     * Calls `Reduce` on `Tensor<Dims...> src`, moves and overwrites `Tensor<Dims...> src`, returns moved version
     */
    template<size_t Axis, typename ApplyOp, typename ReduceOp, size_t... Dims> requires
        FloatBinaryOp<ApplyOp> && FloatBinaryOp<ReduceOp> && std::default_initializable<ApplyOp> &&
        std::default_initializable<ReduceOp> && requires { { ReduceOp::identity } -> std::convertible_to<float>; }
    Tensor<Dims...> BroadcastReduceMove(Tensor<Dims...> &&src) {
        auto reduced = Reduce<Axis, ReduceOp>(src);
        return BroadcastMapMove<Axis, ApplyOp>(std::move(src), reduced);
    }

    // @doc: template<size_t Axis, typename ApplyOp, typename ReduceOp, size_t... Dims> requires FloatBinaryOp<ApplyOp> && FloatBinaryOp<ReduceOp> && std::default_initializable<ApplyOp> && std::default_initializable<ReduceOp> && requires { { ReduceOp::identity } -> std::convertible_to<float>; } void BroadcastReduceInplace(Tensor<Dims...> &src)
    /**
     * Fused composition of `Broadcast` and `Reduce`; "`Broadcast` after `Reduce`"
     * `Reduce` with `ReduceOp`, then `Broadcast` that result back onto `Tensor<Dims...> src`
     * Calls `Reduce` on `Tensor<Dims...> src`, overwrites `Tensor<Dims...> src` inplace, no return
     */
    template<size_t Axis, typename ApplyOp, typename ReduceOp, size_t... Dims> requires
        FloatBinaryOp<ApplyOp> && FloatBinaryOp<ReduceOp> && std::default_initializable<ApplyOp> &&
        std::default_initializable<ReduceOp> && requires { { ReduceOp::identity } -> std::convertible_to<float>; }
    void BroadcastReduceInplace(Tensor<Dims...> &src) {
        auto reduced = Reduce<Axis, ReduceOp>(src);
        BroadcastApply<Axis, ApplyOp>(src, reduced);
    }
}
