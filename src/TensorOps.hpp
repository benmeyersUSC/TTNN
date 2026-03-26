#pragma once
#include <ranges>
#include "TensorFunctions.hpp"


namespace TTTN {
    // @doc: template<typename F, typename G> struct Compose
    /**
     * Chain application of `FloatUnaryOp` with either another `FloatUnaryOp` or a `FloatBinaryOp`
     * Struct has two specialized overloads of `operator()` for the two following cases:
     * Unary ∘ Unary → Unary:  `Compose<Log, Abs>{}(x) == log(|x|)`
     * Unary ∘ Binary → Binary: `Compose<Exp, Sub>{}(a, b) == exp(a - b)`
     */
    template<typename F, typename G>
    struct Compose {
        constexpr float operator()(float x) const
            requires FloatUnaryOp<F> && FloatUnaryOp<G> { return F{}(G{}(x)); }

        constexpr float operator()(float a, float b) const
            requires FloatUnaryOp<F> && FloatBinaryOp<G> { return F{}(G{}(a, b)); }
    };


    // @doc: template<size_t... Dims> Tensor<Dims...> operator+(const Tensor<Dims...>& a, const Tensor<Dims...>& b)
    /** Element-wise add, uses parallel functional `zip` */
    template<size_t... Dims>
    Tensor<Dims...> operator+(const Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        return Zip<Add>(a, b);
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> operator-(const Tensor<Dims...>& a, const Tensor<Dims...>& b)
    /** Element-wise subtract, uses parallel functional `zip` */
    template<size_t... Dims>
    Tensor<Dims...> operator-(const Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        return Zip<Sub>(a, b);
    }

    template<size_t... Dims>
    Tensor<Dims...> &operator+=(Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        a.zip_apply(b, Add{});
        return a;
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> operator*(const Tensor<Dims...>& a, float s)
    /** Scalar multiply, uses parallel functional `map` */
    template<size_t... Dims>
    Tensor<Dims...> operator*(const Tensor<Dims...> &a, float s) {
        return a.map([s](const float x) { return x * s; });
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> operator*(float s, const Tensor<Dims...>& a)
    /** Scalar multiply, uses parallel functional `map` */
    template<size_t... Dims>
    Tensor<Dims...> operator*(float s, const Tensor<Dims...> &a) { return a * s; }

    // @doc: template<size_t... Dims> Tensor<Dims...> operator*(const Tensor<Dims...>& a, const Tensor<Dims...>& b)
    /** Hadamard (element-wise) product, uses parallel functional `zip` */
    template<size_t... Dims>
    Tensor<Dims...> operator*(const Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        return Zip<Mul>(a, b);
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> operator/(const Tensor<Dims...>& a, const Tensor<Dims...>& b)
    /** Element-wise) division, uses parallel functional `zip` */
    template<size_t... Dims>
    Tensor<Dims...> operator/(const Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        return Zip<Div>(a, b);
    }
}
