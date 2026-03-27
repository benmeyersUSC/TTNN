#pragma once
#include "TensorShapeOps.hpp"


namespace TTTN {
    struct Add {
        constexpr float operator()(float a, float b) const { return a + b; }
        static constexpr float identity = 0.f;
    };

    struct Mul {
        constexpr float operator()(float a, float b) const { return a * b; }
        static constexpr float identity = 1.f;
    };

    struct Max {
        constexpr float operator()(float a, float b) const { return std::max(a, b); }
        static constexpr float identity = -std::numeric_limits<float>::infinity();
    };

    struct Sub {
        constexpr float operator()(float a, float b) const { return a - b; }
    };

    struct Div {
        constexpr float operator()(float a, float b) const { return a / b; }
    };

    struct AbsDiff {
        constexpr float operator()(float a, float b) const { return std::abs(a - b); }
    };

    // Unary tags — satisfy FloatUnaryOp; used with Map<Op> and Tensor::map()
    struct Log {
        constexpr float operator()(float x) const { return std::log(x); }
    };

    struct Exp {
        constexpr float operator()(float x) const { return std::exp(x); }
    };

    struct Neg {
        constexpr float operator()(float x) const { return -x; }
    };

    struct Sq {
        constexpr float operator()(float x) const { return x * x; }
    };

    struct Abs {
        constexpr float operator()(float x) const { return std::abs(x); }
    };

    struct OneMinus {
        constexpr float operator()(float x) const { return 1.f - x; }
    };

    // Clamp<Lo, Hi>: clamp x to [Lo, Hi]. Omit Hi for one-sided ClampMin.
    template<float Lo, float Hi = std::numeric_limits<float>::infinity()>
    struct Clamp {
        constexpr float operator()(float x) const { return std::min(std::max(x, Lo), Hi); }
    };

    // Step<T>: 1 if x < T, else 0. Useful for counting elements below a threshold.
    template<float T>
    struct Step {
        constexpr float operator()(float x) const { return x < T ? 1.f : 0.f; }
    };


    // ======================== MAP ========================

    // Map → copy: returns new tensor with Op applied element-wise
    template<typename Op, size_t... Dims>
        requires FloatUnaryOp<Op> && std::default_initializable<Op>
    Tensor<Dims...> Map(const Tensor<Dims...> &src) {
        return src.map(Op{});
    }

    // MapMove → move: consumes src, applies Op in-place, returns it (no extra alloc)
    template<typename Op, size_t... Dims>
        requires FloatUnaryOp<Op> && std::default_initializable<Op>
    Tensor<Dims...> MapMove(Tensor<Dims...> &&src) {
        src.apply(Op{});
        return std::move(src);
    }

    // ======================== ZIP ========================

    // Zip → copy: returns new tensor; result[i] = Op{}(A[i], B[i])
    template<typename Op, size_t... Dims>
        requires FloatBinaryOp<Op> && std::default_initializable<Op>
    Tensor<Dims...> Zip(const Tensor<Dims...> &A, const Tensor<Dims...> &B) {
        Tensor<Dims...> result(A);
        result.zip_apply(B, Op{});
        return result;
    }

    // ZipMove → move: consumes A, overwrites it with Op{}(A[i], B[i]), returns (no extra alloc)
    template<typename Op, size_t... Dims>
        requires FloatBinaryOp<Op> && std::default_initializable<Op>
    Tensor<Dims...> ZipMove(Tensor<Dims...> &&A, const Tensor<Dims...> &B) {
        A.zip_apply(B, Op{});
        return std::move(A);
    }


    template<size_t Axis, FloatBinaryOp F, size_t... Dims>
    void TensorIndexApply(Tensor<Dims...> &dst, size_t idx,
                          const typename RemoveAxis<Axis, Dims...>::type &src, F f) {
        using Dest = Tensor<Dims...>;
        using Slice = RemoveAxis<Axis, Dims...>::type;
        ParForEach(Slice::Size, [&](const size_t i) {
            auto src_multi = Slice::flat_to_multi(i);
            std::array<size_t, Dest::Rank> dst_multi{};
            size_t src_d = 0;
            for (size_t d = 0; d < Dest::Rank; ++d) {
                dst_multi[d] = d == Axis ? idx : src_multi[src_d++];
            }
            const size_t flat = Dest::multi_to_flat(dst_multi);
            dst.flat(flat) = f(dst.flat(flat), src.flat(i));
        });
        for (size_t i = 0; i < Slice::Size; ++i) {
        }
    }


    // Physical permutation — rearranges data, changes the type.
    // Permute<1,0>(Tensor<3,5>) → Tensor<5,3> with data physically transposed.
    // result(i0, i1, ...) = src(i_{perm[0]}, i_{perm[1]}, ...)
    template<size_t... Perm, size_t... Dims>
    Tensor<SizeTemplateGet<Perm, Dims...>::value...>
    Permute(const Tensor<Dims...> &src) {
        static_assert(sizeof...(Perm) == sizeof...(Dims), "Permutation length must match rank");
        using Source = Tensor<Dims...>;
        using Result = Tensor<SizeTemplateGet<Perm, Dims...>::value...>;
        constexpr std::array<size_t, sizeof...(Dims)> perm = {Perm...};
        Result dst;
        ParForEach(Result::Size, [&](size_t i) {
            auto dst_multi = Result::flat_to_multi(i);
            std::array<size_t, sizeof...(Dims)> src_multi{};
            for (size_t j = 0; j < sizeof...(Dims); ++j)
                src_multi[perm[j]] = dst_multi[j];
            dst.flat(i) = src.flat(Source::multi_to_flat(src_multi));
        });
        return dst;
    }
    
    // Transpose — reverse all axes.
    // Transpose(Tensor<A,B,C>) → Tensor<C,B,A>
    template<size_t... Dims>
    auto Transpose(const Tensor<Dims...> &src) {
        return []<size_t... I>(const auto &s, std::index_sequence<I...>) {
            return Permute<(sizeof...(Dims) - 1 - I)...>(s);
        }(src, std::make_index_sequence<sizeof...(Dims)>{});
    }


    // @doc: template<size_t Axis, size_t... Dims> typename RemoveAxis<Axis, Dims...>::type TensorIndex(const Tensor<Dims...>& src, size_t idx)
    /** ######### */
    template<size_t Axis, size_t... Dims>
    RemoveAxis<Axis, Dims...>::type TensorIndex(const Tensor<Dims...> &src, size_t idx) {
        using Source = Tensor<Dims...>;
        using Result = RemoveAxis<Axis, Dims...>::type;

        Result dst;
        ParForEach(Result::Size, [&](const size_t i) {
            auto dst_multi = Result::flat_to_multi(i);
            std::array<size_t, Source::Rank> src_multi{};
            size_t dst_d = 0;
            for (size_t d = 0; d < Source::Rank; ++d) {
                src_multi[d] = d == Axis ? idx : dst_multi[dst_d++];
            }
            dst.flat(i) = src.flat(Source::multi_to_flat(src_multi));
        });
        return dst;
    }
}
