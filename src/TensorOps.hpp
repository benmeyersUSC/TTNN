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


    // ---- tensor OP= tensor (inplace, no alloc) ----

    template<size_t... Dims>
    Tensor<Dims...> &operator+=(Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        a.zip_apply(b, Add{}); return a;
    }

    template<size_t... Dims>
    Tensor<Dims...> &operator-=(Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        a.zip_apply(b, Sub{}); return a;
    }

    template<size_t... Dims>
    Tensor<Dims...> &operator*=(Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        a.zip_apply(b, Mul{}); return a;
    }

    template<size_t... Dims>
    Tensor<Dims...> &operator/=(Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        a.zip_apply(b, Div{}); return a;
    }

    // ---- tensor OP tensor (copy) ----

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

    // ---- scalar OP= (inplace, no alloc) ----

    template<size_t... Dims>
    Tensor<Dims...> &operator*=(Tensor<Dims...> &a, float s) {
        a.apply([s](float x) { return x * s; }); return a;
    }

    template<size_t... Dims>
    Tensor<Dims...> &operator/=(Tensor<Dims...> &a, float s) {
        a.apply([s](float x) { return x / s; }); return a;
    }

    template<size_t... Dims>
    Tensor<Dims...> &operator+=(Tensor<Dims...> &a, float s) {
        a.apply([s](float x) { return x + s; }); return a;
    }

    template<size_t... Dims>
    Tensor<Dims...> &operator-=(Tensor<Dims...> &a, float s) {
        a.apply([s](float x) { return x - s; }); return a;
    }

    // ---- scalar OP (copy) ----

    // @doc: template<size_t... Dims> Tensor<Dims...> operator*(const Tensor<Dims...>& a, float s)
    /** Scalar multiply, uses parallel functional `map` */
    template<size_t... Dims>
    Tensor<Dims...> operator*(const Tensor<Dims...> &a, float s) {
        return a.map([s](float x) { return x * s; });
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> operator*(float s, const Tensor<Dims...>& a)
    /** Scalar multiply, uses parallel functional `map` */
    template<size_t... Dims>
    Tensor<Dims...> operator*(float s, const Tensor<Dims...> &a) { return a * s; }

    template<size_t... Dims>
    Tensor<Dims...> operator/(const Tensor<Dims...> &a, float s) {
        return a.map([s](float x) { return x / s; });
    }

    template<size_t... Dims>
    Tensor<Dims...> operator+(const Tensor<Dims...> &a, float s) {
        return a.map([s](float x) { return x + s; });
    }

    template<size_t... Dims>
    Tensor<Dims...> operator-(const Tensor<Dims...> &a, float s) {
        return a.map([s](float x) { return x - s; });
    }
}
