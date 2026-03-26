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

        static constexpr auto bases = [] {
            std::array<size_t, Result::Size> t{};
            for (size_t out_i = 0; out_i < Result::Size; ++out_i) {
                const auto dm = Result::FlatToMulti(out_i);
                std::array<size_t, Source::Rank> sm{};
                size_t d = 0;
                for (size_t sd = 0; sd < Source::Rank; ++sd)
                    sm[sd] = (sd == Axis) ? 0 : dm[d++];
                t[out_i] = Source::MultiToFlat(sm);
            }
            return t;
        }();

        static constexpr size_t project(size_t i) {
            const size_t after = i % axis_stride;
            const size_t d_axis = (i / axis_stride) % axis_dim;
            const size_t before = i - after - d_axis * axis_stride;
            return before / axis_dim + after;
        }
    };

    // @doc: template<size_t Axis, FloatBinaryOp ReduceFn, size_t... Dims> typename RemoveAxis<Axis, Dims...>::type ReduceApply(const Tensor<Dims...>& src, float init, ReduceFn rfn)
    /**
     * Generalized axis reduction: collapses `Axis` by folding elements with `rfn(acc, val)` starting from `init`
     * **Tag-param overload**: `ReduceApply<Axis, Op>(src)` — `Op::identity` used as init; requires monoid `Op`
     * `ReduceApply<Axis, Add>(src)` == sum; `ReduceApply<Axis, Max>(src)` == max
     */
    template<size_t Axis, FloatBinaryOp ReduceFn, size_t... Dims>
    typename RemoveAxis<Axis, Dims...>::type ReduceApply(const Tensor<Dims...> &src, float init, ReduceFn rfn) {
        using K = ReduceKernel<Axis, Dims...>;
        auto k_range = std::views::iota(size_t{0}, K::axis_dim);
        typename K::Result dst;
        ParForEach(K::Result::Size, [&](size_t out_i) {
            dst.flat(out_i) = std::transform_reduce(
                std::execution::unseq,
                k_range.begin(), k_range.end(),
                init, rfn,
                [&](size_t k) { return src.flat(K::bases[out_i] + k * K::axis_stride); });
        });
        return dst;
    }

    // Tag overload: ReduceApply<Axis, Op>(src) — Op::identity used as init
    template<size_t Axis, typename Op, size_t... Dims>
        requires FloatBinaryOp<Op> && requires { { Op::identity } -> std::convertible_to<float>; }
    typename RemoveAxis<Axis, Dims...>::type ReduceApply(const Tensor<Dims...> &src) {
        return ReduceApply<Axis>(src, Op::identity, Op{});
    }

    // MOVE VERSIONS (these are not actually move, just for API unification)
    template<size_t Axis, FloatBinaryOp ReduceFn, size_t... Dims>
    typename RemoveAxis<Axis, Dims...>::type
    ReduceApplyMove(Tensor<Dims...>&& src, float init, ReduceFn rfn) {
        return ReduceApply<Axis>(src, init, rfn);
    }

    template<size_t Axis, typename Op, size_t... Dims>
        requires FloatBinaryOp<Op> && requires { { Op::identity } -> std::convertible_to<float>; }
    typename RemoveAxis<Axis, Dims...>::type
    ReduceApplyMove(Tensor<Dims...>&& src) {
        return ReduceApply<Axis>(src, Op::identity, Op{});
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

    // MOVE VERSION (also not actually a move, must change shape)
    template<size_t Axis, size_t N, size_t... Dims>
    typename InsertAxis<Axis, N, Dims...>::type
    ExpandMove(Tensor<Dims...>&& src) {
        // must allocate new tensor
        return Expand<Axis, N>(src);
    }

    // @doc: template<size_t Axis, FloatBinaryOp F, size_t... Dims> Tensor<Dims...> BroadcastApply(const Tensor<Dims...>& A, const typename RemoveAxis<Axis, Dims...>::type& b, F f)
    /**
     * Apply binary `f(a_elem, b_elem)` element-wise between `A` and `b` broadcast along `Axis`
     * **Tag-param overload**: `BroadcastApply<Axis, F>(A, b)` — default-constructs `F`
     * `BroadcastApply<0, Add>(Z, bias)` adds bias to every row
     */
    template<size_t Axis, FloatBinaryOp F, size_t... Dims>
    Tensor<Dims...> BroadcastApply(const Tensor<Dims...> &A, const typename RemoveAxis<Axis, Dims...>::type &b, F f) {
        using K = ReduceKernel<Axis, Dims...>;
        Tensor<Dims...> result;
        ParForEach(Tensor<Dims...>::Size, [&](size_t i) {
            result.flat(i) = f(A.flat(i), b.flat(K::project(i)));
        });
        return result;
    }

    // Tag overload: BroadcastApply<Axis, F>(A, b) — no op arg required
    template<size_t Axis, typename F, size_t... Dims>
        requires FloatBinaryOp<F> && std::default_initializable<F>
    Tensor<Dims...> BroadcastApply(const Tensor<Dims...> &A, const typename RemoveAxis<Axis, Dims...>::type &b) {
        return BroadcastApply<Axis>(A, b, F{});
    }

    // MOVE VERSIONS OF BROADCAST: TRUE MOVE
    template<size_t Axis, FloatBinaryOp F, size_t... Dims>
    Tensor<Dims...>
    BroadcastApplyMove(Tensor<Dims...>&& A,
                    const typename RemoveAxis<Axis, Dims...>::type& b,
                    F f) {
        using K = ReduceKernel<Axis, Dims...>;

        // overwrite on A
        ParForEach(Tensor<Dims...>::Size, [&](size_t i) {
            A.flat(i) = f(A.flat(i), b.flat(K::project(i)));
        });

        return std::move(A);
    }
    // tag version
    template<size_t Axis, typename F, size_t... Dims>
    requires FloatBinaryOp<F> && std::default_initializable<F>
    Tensor<Dims...>
    BroadcastApplyMove(Tensor<Dims...>&& A,
                    const typename RemoveAxis<Axis, Dims...>::type& b) {
        return BroadcastApplyMove<Axis>(std::move(A), b, F{});
    }

    // these are basically in-place, but wrap in move and can delete old A
        template<size_t Axis, FloatBinaryOp F, size_t... Dims>
    Tensor<Dims...>&
    BroadcastApplyInplace(Tensor<Dims...>& A,
                    const typename RemoveAxis<Axis, Dims...>::type& b,
                    F f) {
        using K = ReduceKernel<Axis, Dims...>;

        // overwrite on A
        ParForEach(Tensor<Dims...>::Size, [&](size_t i) {
            A.flat(i) = f(A.flat(i), b.flat(K::project(i)));
        });

        return A;
    }
    // tag version
    template<size_t Axis, typename F, size_t... Dims>
    requires FloatBinaryOp<F> && std::default_initializable<F>
    Tensor<Dims...>&
    BroadcastApplyInplace(Tensor<Dims...>& A,
                    const typename RemoveAxis<Axis, Dims...>::type& b) {
        return BroadcastApplyInplace<Axis>(A, b, F{});
    }




    // @doc: template<size_t Axis, FloatBinaryOp ApplyFn, FloatBinaryOp ReduceFn, size_t... Dims> Tensor<Dims...> BroadcastReduce(const Tensor<Dims...>& src, float init, ApplyFn afn, ReduceFn rfn)
    /**
     * Reduce along `Axis` then broadcast the result back with a second op — `BroadcastApply<Axis>(src, ReduceApply<Axis>(src, init, rfn), afn)`
     * **Tag-param overload**: `BroadcastReduce<Axis, ApplyOp, ReduceOp>(src)` — `ReduceOp::identity` as init; requires monoid `ReduceOp`
     * Powers `Softmax`: `BroadcastReduce<Axis, Compose<Exp,Sub>, Max>(x)` then `BroadcastReduce<Axis, Div, Add>(exps)`
     */
    template<size_t Axis, FloatBinaryOp ApplyFn, FloatBinaryOp ReduceFn, size_t... Dims>
    Tensor<Dims...> BroadcastReduce(const Tensor<Dims...> &src, float init, ApplyFn afn, ReduceFn rfn) {
        return BroadcastApply<Axis>(src, ReduceApply<Axis>(src, init, rfn), afn);
    }

    template<size_t Axis, typename ApplyOp, typename ReduceOp, size_t... Dims>
        requires FloatBinaryOp<ApplyOp> && FloatBinaryOp<ReduceOp> &&
                 std::default_initializable<ApplyOp> && std::default_initializable<ReduceOp> &&
                 requires { { ReduceOp::identity } -> std::convertible_to<float>; }
    Tensor<Dims...> BroadcastReduce(const Tensor<Dims...> &src) {
        return BroadcastApply<Axis>(src, ReduceApply<Axis>(src, ReduceOp::identity, ReduceOp{}), ApplyOp{});
    }


    // MOVE VERSION (still needs temp alloc for reduced, but then overwrites src)
    template<size_t Axis, FloatBinaryOp ApplyFn, FloatBinaryOp ReduceFn, size_t... Dims>
    Tensor<Dims...>
    BroadcastReduceMove(Tensor<Dims...>&& src,
                        float init,
                        ApplyFn afn,
                        ReduceFn rfn) {
        // must do this
        auto reduced = ReduceApply<Axis>(src, init, rfn);
        // here, we can use Move!
        return BroadcastApplyMove<Axis>(std::move(src), reduced, afn);
    }
    template<size_t Axis, typename ApplyOp, typename ReduceOp, size_t... Dims>
    requires FloatBinaryOp<ApplyOp> && FloatBinaryOp<ReduceOp> &&
             std::default_initializable<ApplyOp> &&
             std::default_initializable<ReduceOp> &&
             requires { { ReduceOp::identity } -> std::convertible_to<float>; }
    Tensor<Dims...>
    BroadcastReduceMove(Tensor<Dims...>&& src) {
        auto reduced = ReduceApply<Axis>(src, ReduceOp::identity, ReduceOp{});
        return BroadcastApplyMove<Axis>(std::move(src), reduced, ApplyOp{});
    }


    // inplace
        template<size_t Axis, FloatBinaryOp ApplyFn, FloatBinaryOp ReduceFn, size_t... Dims>
    Tensor<Dims...>&
    BroadcastReduceInplace(Tensor<Dims...>& src,
                        float init,
                        ApplyFn afn,
                        ReduceFn rfn) {
        // must do this
        auto reduced = ReduceApply<Axis>(src, init, rfn);
        // here, we can use Inplace!
        return BroadcastApplyInplace<Axis>(src, reduced, afn);
    }
        template<size_t Axis, typename ApplyOp, typename ReduceOp, size_t... Dims>
    requires FloatBinaryOp<ApplyOp> && FloatBinaryOp<ReduceOp> &&
             std::default_initializable<ApplyOp> &&
             std::default_initializable<ReduceOp> &&
             requires { { ReduceOp::identity } -> std::convertible_to<float>; }
    Tensor<Dims...>&
    BroadcastReduceMove(Tensor<Dims...>& src) {
        auto reduced = ReduceApply<Axis>(src, ReduceOp::identity, ReduceOp{});
        return BroadcastApplyInplace<Axis>(src, reduced, ApplyOp{});
    }


}
