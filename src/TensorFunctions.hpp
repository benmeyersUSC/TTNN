#pragma once
#include "TensorShapeOps.hpp"


// THESE NEED TO BE FIXED !!!!
// USE INPLACE VERSIONS IN TENSOR!

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

    // MapCopy (current map) → returns new tensor (deep copy)
    template<FloatUnaryOp F, size_t... Dimensions>
    static Tensor<Dimensions...> MapCopy(const Tensor<Dimensions...> &src) {
        return src.map(F{});
    }

    // MapInplace → applies F in-place, modifies src
    template<FloatUnaryOp F, size_t... Dimensions>
    static void MapInplace(Tensor<Dimensions...> &src) {
        src.apply(F{});
    }

    // MapMove → consumes src, applies F, reuses storage, returns tensor
    template<FloatUnaryOp F, size_t... Dimensions>
    static Tensor<Dimensions...> MapMove(Tensor<Dimensions...> &&src) {
        src.apply(F{});
        return std::move(src);
    }

    // ======================== ZIP ========================

    // ZipCopy (current zip) → returns new tensor (deep copy)
    template<FloatBinaryOp F, size_t... Dimensions>
    static Tensor<Dimensions...> ZipCopy(const Tensor<Dimensions...> &a, const Tensor<Dimensions...> &b) {
        return a.zip(b, F{});
    }

    // ZipInplace → writes into first argument, modifies it
    template<FloatBinaryOp F, size_t... Dimensions>
    static void ZipInplace(Tensor<Dimensions...> &dst, const Tensor<Dimensions...> &src, F f) {
        dst.zip_apply(src, f);
    }

    // ZipMove → consumes dst, overwrites it, reuses storage
    template<FloatBinaryOp F, size_t... Dimensions>
    static Tensor<Dimensions...> ZipMove(Tensor<Dimensions...> &&dst, const Tensor<Dimensions...> &src, F f) {
        dst.zip_apply(src, f);
        return std::move(dst);
    }


    // ======================== MAP ========================

    // MapCopy → returns new tensor (deep copy)
    template<typename Op, size_t... Dims>
        requires FloatUnaryOp<Op> && std::default_initializable<Op>
    Tensor<Dims...> MapCopy(const Tensor<Dims...> &src) {
        return src.map(Op{});
    }

    // MapInplace → applies Op in-place, modifies src
    template<typename Op, size_t... Dims>
        requires FloatUnaryOp<Op> && std::default_initializable<Op>
    void MapInplace(Tensor<Dims...> &src) {
        src.apply(Op{});
    }

    // MapMove → consumes src, applies Op, reuses storage
    template<typename Op, size_t... Dims>
        requires FloatUnaryOp<Op> && std::default_initializable<Op>
    Tensor<Dims...> MapMove(Tensor<Dims...> &&src) {
        src.apply(Op{});
        return std::move(src);
    }

    // ======================== ZIP ========================

    // ZipCopy → returns new tensor (deep copy)
    template<typename Op, size_t... Dims>
        requires FloatBinaryOp<Op>
    Tensor<Dims...> ZipCopy(const Tensor<Dims...> &A, const Tensor<Dims...> &B) {
        return A.zip(B, Op{});
    }

    // ZipInplace → writes into first argument, modifies it
    template<typename Op, size_t... Dims>
        requires FloatBinaryOp<Op>
    void ZipInplace(Tensor<Dims...> &dst, const Tensor<Dims...> &src) {
        dst.zip_apply(src, Op{});
    }

    // ZipMove → consumes dst, overwrites it, reuses storage
    template<typename Op, size_t... Dims>
        requires FloatBinaryOp<Op>
    Tensor<Dims...> ZipMove(Tensor<Dims...> &&dst, const Tensor<Dims...> &src) {
        dst.zip_apply(src, Op{});
        return std::move(dst);
    }

    // ======================== EXISTING TEMPLATED FUNCTIONS ========================

    // Original Map → just calls copy
    template<typename Op, size_t... Dims>
        requires FloatUnaryOp<Op> && std::default_initializable<Op>
    Tensor<Dims...> Map(const Tensor<Dims...> &src) {
        return MapCopy<Op>(src);
    }

    // Original Zip → just calls copy
    template<typename Op, size_t... Dims>
        requires FloatBinaryOp<Op>
    Tensor<Dims...> Zip(const Tensor<Dims...> &A, const Tensor<Dims...> &B) {
        return ZipCopy<Op>(A, B);
    }


    template<size_t Axis, FloatBinaryOp F, size_t... Dims>
    void TensorIndexApply(Tensor<Dims...> &dst, size_t idx,
                          const typename RemoveAxis<Axis, Dims...>::type &src, F f) {
        using Dest = Tensor<Dims...>;
        using Slice = RemoveAxis<Axis, Dims...>::type;
        ParForEach(Slice::Size, [&](const size_t i) {
            auto src_multi = Slice::FlatToMulti(i);
            std::array<size_t, Dest::Rank> dst_multi{};
            size_t src_d = 0;
            for (size_t d = 0; d < Dest::Rank; ++d) {
                dst_multi[d] = d == Axis ? idx : src_multi[src_d++];
            }
            const size_t flat = Dest::MultiToFlat(dst_multi);
            dst.flat(flat) = f(dst.flat(flat), src.flat(i));
        });
        for (size_t i = 0; i < Slice::Size; ++i) {
        }
    }


    // -------------------
    // 1) In-place permutation
    // -------------------
    template<size_t... Perm, size_t... Dims>
    void PermuteInplace(Tensor<Dims...> &t) {
        t.permute_inplace({Perm...});
    }

    template<size_t... Dims>
    void TransposeInplace(Tensor<Dims...> &t) {
        std::array<size_t, sizeof...(Dims)> perm{};
        for (size_t i = 0; i < sizeof...(Dims); ++i) perm[i] = sizeof...(Dims) - 1 - i;
        t.permute_inplace(perm);
    }

    // -------------------
    // 2) Move / zero-copy permutation
    // -------------------
    template<size_t... Perm, size_t... Dims>
    Tensor<Dims...> PermuteMove(Tensor<Dims...> &&src) {
        Tensor<Dims...> dst = std::move(src); // move storage
        dst.permute_inplace({Perm...});
        return dst;
    }

    template<size_t... Dims>
    Tensor<Dims...> TransposeMove(Tensor<Dims...> &&src) {
        Tensor<Dims...> dst = std::move(src); // move storage
        std::array<size_t, sizeof...(Dims)> perm{};
        for (size_t i = 0; i < sizeof...(Dims); ++i) perm[i] = sizeof...(Dims) - 1 - i;
        dst.permute_inplace(perm);
        return dst;
    }

    // -------------------
    // 3) Deep-copy permutation
    // -------------------
    template<size_t... Perm, size_t... Dims>
    Tensor<Dims...> Permute(const Tensor<Dims...> &src) {
        Tensor<Dims...> dst = src; // deep copy storage
        dst.permute_inplace({Perm...});
        return dst;
    }

    template<size_t... Dims>
    Tensor<Dims...> Transpose(const Tensor<Dims...> &src) {
        Tensor<Dims...> dst = src; // deep copy storage
        std::array<size_t, sizeof...(Dims)> perm{};
        for (size_t i = 0; i < sizeof...(Dims); ++i) perm[i] = sizeof...(Dims) - 1 - i;
        dst.permute_inplace(perm);
        return dst;
    }


    template<size_t Axis, size_t... Dims>
    RemoveAxis<Axis, Dims...>::type TensorIndex(const Tensor<Dims...> &src, size_t idx) {
        using Source = Tensor<Dims...>;
        using Result = RemoveAxis<Axis, Dims...>::type;

        Result dst;
        ParForEach(Result::Size, [&](const size_t i) {
            auto dst_multi = Result::FlatToMulti(i);
            std::array<size_t, Source::Rank> src_multi{};
            size_t dst_d = 0;
            for (size_t d = 0; d < Source::Rank; ++d) {
                src_multi[d] = d == Axis ? idx : dst_multi[dst_d++];
            }
            dst.flat(i) = src.flat(Source::MultiToFlat(src_multi));
        });
        return dst;
    }
}
