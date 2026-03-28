#pragma once
#include "TensorOps.hpp"

namespace TTTN {
    // @doc: template<size_t Axis, size_t... Dims> struct ReduceKernel
    /**
     * Shared kernel for all axis-reduction and broadcast operations
     * Compile-time `static constexpr`:
     * `axis_dim = SizeTemplateGet<Axis, Dims...>::value`
     * `axis_stride = Source::Strides[Axis]`
     * `std::array<size_t, Result::Size> bases` — flat index in `Source` for each output index with axis set to 0
     * `static constexpr size_t project(size_t i)` — flat index in `Result` for source flat index `i` (axis contribution stripped); closed-form `i - ((i / axis_stride) % axis_dim) * axis_stride`; no table, `axis_stride` compile-time so division compiles to multiply-shift
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

    template<size_t Axis, typename Op, size_t... Dims>
        requires FloatBinaryOp<Op> && std::default_initializable<Op> &&
                 requires { { Op::identity } -> std::convertible_to<float>; }
    typename RemoveAxis<Axis, Dims...>::type ReduceApply(const Tensor<Dims...> &src) {
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

    // @doc: template<size_t Axis, size_t N, size_t... Dims> InsertAxis<Axis, N, Dims...>::type Expand(const Tensor<Dims...>& src)
    /**
     * Broadcasts a reduced tensor back up by repeating it `N` times along `Axis`
     * `Expand<0, 5>(Tensor<3>)` → `Tensor<5, 3>` — 5 copies stacked along axis 0
     */
    template<size_t Axis, size_t N, size_t... Dims>
    typename InsertAxis<Axis, N, Dims...>::type Expand(const Tensor<Dims...> &src) {
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

    // All tag-only. Op is default-constructed; no runtime-lambda overloads.
    // Shape is preserved — these are the only broadcast ops that justify Move/Inplace variants.

    // Copy: result[i] = Op{}(A[i], b_broadcast[i])
    template<size_t Axis, typename Op, size_t... Dims>
        requires FloatBinaryOp<Op> && std::default_initializable<Op>
    Tensor<Dims...> BroadcastApply(const Tensor<Dims...> &A,
                                   const typename RemoveAxis<Axis, Dims...>::type &b) {
        using K = ReduceKernel<Axis, Dims...>;
        Tensor<Dims...> result;
        ParForEach(Tensor<Dims...>::Size, [&](size_t i) {
            result.flat(i) = Op{}(A.flat(i), b.flat(K::project(i)));
        });
        return result;
    }

    // Move: overwrites A's storage in-place, returns it — no alloc
    template<size_t Axis, typename Op, size_t... Dims>
        requires FloatBinaryOp<Op> && std::default_initializable<Op>
    Tensor<Dims...> BroadcastApplyMove(Tensor<Dims...> &&A,
                                       const typename RemoveAxis<Axis, Dims...>::type &b) {
        using K = ReduceKernel<Axis, Dims...>;
        ParForEach(Tensor<Dims...>::Size, [&](size_t i) {
            A.flat(i) = Op{}(A.flat(i), b.flat(K::project(i)));
        });
        return std::move(A);
    }

    // Inplace: void, modifies A directly — no alloc
    // @doc: template<size_t Axis, FloatBinaryOp F, size_t... Dims> Tensor<Dims...> BroadcastApply(const Tensor<Dims...>& A, const typename RemoveAxis<Axis, Dims...>::type& b, F f)
    /**
     * Apply binary `f(a_elem, b_elem)` element-wise between `A` and `b` broadcast along `Axis`
     * **Tag-param overload**: `BroadcastApply<Axis, F>(A, b)` — default-constructs `F`
     * `BroadcastApply<0, Add>(Z, bias)` adds bias to every row
     */
    template<size_t Axis, typename Op, size_t... Dims>
        requires FloatBinaryOp<Op> && std::default_initializable<Op>
    void BroadcastApplyInplace(Tensor<Dims...> &A,
                               const typename RemoveAxis<Axis, Dims...>::type &b) {
        using K = ReduceKernel<Axis, Dims...>;
        ParForEach(Tensor<Dims...>::Size, [&](size_t i) {
            A.flat(i) = Op{}(A.flat(i), b.flat(K::project(i)));
        });
    }

    // Reduce along Axis, then broadcast result back with ApplyOp.
    // Powers Softmax: BroadcastReduce<Axis, Compose<Exp,Sub>, Max> then BroadcastReduceMove<Axis, Div, Add>.

    // Copy
    template<size_t Axis, typename ApplyOp, typename ReduceOp, size_t... Dims>
        requires FloatBinaryOp<ApplyOp> && FloatBinaryOp<ReduceOp> &&
                 std::default_initializable<ApplyOp> && std::default_initializable<ReduceOp> &&
                 requires { { ReduceOp::identity } -> std::convertible_to<float>; }
    // @doc: template<size_t Axis, FloatBinaryOp ApplyFn, FloatBinaryOp ReduceFn, size_t... Dims> Tensor<Dims...> BroadcastReduce(const Tensor<Dims...>& src, float init, ApplyFn afn, ReduceFn rfn)
    /**
     * Reduce along `Axis` then broadcast the result back with a second op — `BroadcastApply<Axis>(src, ReduceApply<Axis>(src, init, rfn), afn)`
     * **Tag-param overload**: `BroadcastReduce<Axis, ApplyOp, ReduceOp>(src)` — `ReduceOp::identity` as init; requires monoid `ReduceOp`
     * Powers `Softmax`: `BroadcastReduce<Axis, Compose<Exp,Sub>, Max>(x)` then `BroadcastReduce<Axis, Div, Add>(exps)`
     */
    Tensor<Dims...> BroadcastReduce(const Tensor<Dims...> &src) {
        return BroadcastApply<Axis, ApplyOp>(src, ReduceApply<Axis, ReduceOp>(src));
    }

    // Move: one small temp alloc for reduced, then steals src storage
    template<size_t Axis, typename ApplyOp, typename ReduceOp, size_t... Dims>
        requires FloatBinaryOp<ApplyOp> && FloatBinaryOp<ReduceOp> &&
                 std::default_initializable<ApplyOp> && std::default_initializable<ReduceOp> &&
                 requires { { ReduceOp::identity } -> std::convertible_to<float>; }
    Tensor<Dims...> BroadcastReduceMove(Tensor<Dims...> &&src) {
        auto reduced = ReduceApply<Axis, ReduceOp>(src);
        return BroadcastApplyMove<Axis, ApplyOp>(std::move(src), reduced);
    }

    // Inplace: void
    template<size_t Axis, typename ApplyOp, typename ReduceOp, size_t... Dims>
        requires FloatBinaryOp<ApplyOp> && FloatBinaryOp<ReduceOp> &&
                 std::default_initializable<ApplyOp> && std::default_initializable<ReduceOp> &&
                 requires { { ReduceOp::identity } -> std::convertible_to<float>; }
    void BroadcastReduceInplace(Tensor<Dims...> &src) {
        auto reduced = ReduceApply<Axis, ReduceOp>(src);
        BroadcastApplyInplace<Axis, ApplyOp>(src, reduced);
    }
}
