#pragma once
#include <ranges>
#include "TensorFunctions.hpp"


namespace TTTN {
    // @doc: template<size_t... Dims> Tensor<Dims...> &operator+=(Tensor<Dims...> &a, const Tensor<Dims...> &b)
    /** `A += B` (inplace) */
    template<size_t... Dims>
    Tensor<Dims...> &operator+=(Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        a.zip_apply(b, Add{});
        return a;
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> &operator-=(Tensor<Dims...> &a, const Tensor<Dims...> &b)
    /** `A -= B` (inplace) */
    template<size_t... Dims>
    Tensor<Dims...> &operator-=(Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        a.zip_apply(b, Sub{});
        return a;
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> &operator*=(Tensor<Dims...> &a, const Tensor<Dims...> &b)
    /** `A *= B` (inplace) */
    template<size_t... Dims>
    Tensor<Dims...> &operator*=(Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        a.zip_apply(b, Mul{});
        return a;
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> &operator/=(Tensor<Dims...> &a, const Tensor<Dims...> &b)
    /** `A /= B` (inplace) */
    template<size_t... Dims>
    Tensor<Dims...> &operator/=(Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        a.zip_apply(b, Div{});
        return a;
    }


    // @doc: template<size_t... Dims> Tensor<Dims...> operator+(const Tensor<Dims...> &a, const Tensor<Dims...> &b)
    /** `A + B` (returns copy) */
    template<size_t... Dims>
    Tensor<Dims...> operator+(const Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        return Zip<Add>(a, b);
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> operator-(const Tensor<Dims...> &a, const Tensor<Dims...> &b)
    /** `A - B` (returns copy) */
    template<size_t... Dims>
    Tensor<Dims...> operator-(const Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        return Zip<Sub>(a, b);
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> operator*(const Tensor<Dims...> &a, const Tensor<Dims...> &b)
    /** `A * B` (returns copy) */
    template<size_t... Dims>
    Tensor<Dims...> operator*(const Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        return Zip<Mul>(a, b);
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> operator/(const Tensor<Dims...> &a, const Tensor<Dims...> &b)
    /** `A / B` (returns copy) */
    template<size_t... Dims>
    Tensor<Dims...> operator/(const Tensor<Dims...> &a, const Tensor<Dims...> &b) {
        return Zip<Div>(a, b);
    }


    // @doc: template<size_t... Dims> Tensor<Dims...> &operator*=(Tensor<Dims...> &a, float s)
    /** `A *= b` (inplace, `b` is scalar) */
    template<size_t... Dims>
    Tensor<Dims...> &operator*=(Tensor<Dims...> &a, float s) {
        a.apply([s](float x) { return x * s; });
        return a;
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> &operator/=(Tensor<Dims...> &a, float s)
    /** `A /= b` (inplace, `b` is scalar) */
    template<size_t... Dims>
    Tensor<Dims...> &operator/=(Tensor<Dims...> &a, float s) {
        a.apply([s](float x) { return x / s; });
        return a;
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> &operator+=(Tensor<Dims...> &a, float s)
    /** `A += b` (inplace, `b` is scalar) */
    template<size_t... Dims>
    Tensor<Dims...> &operator+=(Tensor<Dims...> &a, float s) {
        a.apply([s](float x) { return x + s; });
        return a;
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> &operator-=(Tensor<Dims...> &a, float s)
    /** `A -= b` (inplace, `b` is scalar) */
    template<size_t... Dims>
    Tensor<Dims...> &operator-=(Tensor<Dims...> &a, float s) {
        a.apply([s](float x) { return x - s; });
        return a;
    }


    // @doc: template<size_t... Dims> Tensor<Dims...> operator*(const Tensor<Dims...> &a, float s)
    /** `A * b` (returns copy, `b` is scalar) */
    template<size_t... Dims>
    Tensor<Dims...> operator*(const Tensor<Dims...> &a, float s) {
        return a.map([s](float x) { return x * s; });
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> operator/(const Tensor<Dims...> &a, float s)
    /** `A / b` (returns copy, `b` is scalar) */
    template<size_t... Dims>
    Tensor<Dims...> operator/(const Tensor<Dims...> &a, float s) {
        return a.map([s](float x) { return x / s; });
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> operator+(const Tensor<Dims...> &a, float s)
    /** `A + b` (returns copy, `b` is scalar) */
    template<size_t... Dims>
    Tensor<Dims...> operator+(const Tensor<Dims...> &a, float s) {
        return a.map([s](float x) { return x + s; });
    }

    // @doc: template<size_t... Dims> Tensor<Dims...> operator-(const Tensor<Dims...> &a, float s)
    /** `A - b` (returns copy, `b` is scalar) */
    template<size_t... Dims>
    Tensor<Dims...> operator-(const Tensor<Dims...> &a, float s) {
        return a.map([s](float x) { return x - s; });
    }
}
