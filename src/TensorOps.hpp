#pragma once
#include <cassert>
#include <concepts>
#include <limits>
#include <ranges>
#include "Tensor.hpp"


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


                                                           // @doc: template<typename F, typename G> struct Compose
                                                           /**
                                                            * Chain application of `FloatUnaryOp` with either another `FloatUnaryOp` or a `FloatBinaryOp`
                                                            * Struct has two specialized overloads of `operator()` for the two following cases:
                                                            *   - Unary ∘ Unary → Unary:  `Compose<Log, Abs>{}(x) == log(|x|)`
                                                            *   - Unary ∘ Binary → Binary: `Compose<Exp, Sub>{}(a, b) == exp(a - b)`
                                                            */
    template<typename F, typename G>
    struct Compose {
        constexpr float operator()(float x) const
            requires FloatUnaryOp<F> && FloatUnaryOp<G> { return F{}(G{}(x)); }

        constexpr float operator()(float a, float b) const
            requires FloatUnaryOp<F> && FloatBinaryOp<G> { return F{}(G{}(a, b)); }
    };


    // @doc: template<std::invocable<size_t> F> void ParForEach(size_t n, F f)
    /** Helper to parallel-execute `std::for_each` on a `std::views::iota(size_t{0}, n)`, calling `f` (something `std::invocable` on `size_t`) on each index */
    template<std::invocable<size_t> F>
    void ParForEach(size_t n, F f) {
        auto range = std::views::iota(size_t{0}, n);
        std::for_each(std::execution::par_unseq, range.begin(), range.end(), f);
    }


    // @doc: template<typename Op, size_t... Dims> Tensor<Dims...> Map(const Tensor<Dims...>& src)
    /**
     * Apply unary tag `Op` element-wise. Equivalent to `src.map(Op{})` but named and passable as a type.
     * `Map<Log>(t)`, `Map<Exp>(t)`, `Map<Neg>(t)`, `Map<Sq>(t)`, `Map<Abs>(t)`
     */
    template<typename Op, size_t... Dims>
        requires FloatUnaryOp<Op> && std::default_initializable<Op>
    Tensor<Dims...> Map(const Tensor<Dims...> &src) {
        return src.map(Op{});
    }

    // @doc: template<typename Op, size_t... Dims> Tensor<Dims...> Zip(const Tensor<Dims...> &A, const Tensor<Dims...> &B)
    /** Apply binary tag `Op` element-wise. Equivalent to `A.zip(B, Op{})` but named and passable as a type. */
    template<typename Op, size_t... Dims>
        requires FloatBinaryOp<Op>
    Tensor<Dims...> Zip(const Tensor<Dims...> &A, const Tensor<Dims...> &B) {
        return A.zip(B, Op{});
    }

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

    // @doc: template<size_t... Dims> Tensor<Dims...>& operator+=(Tensor<Dims...>& a, const Tensor<Dims...>& b)
    /** Element-wise add-to, uses parallel functional `zip_apply` */
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


    // @doc: template<size_t... Perm, size_t... Dims> auto Permute(const Tensor<Dims...>& src)
    /**
     * Parallelized arbitrary permutation of `Tensor`'s indices
     * Returns `PermutedTensorType`
     * Algorithm:
     *   - `using Source = Tensor<Dims...>`
     *   - `using Result = PermutedTensorType<Source, Perm>::type`
     *   - `std::array<size_t, Rank> perm_arr = {Perm...}`
     *   - `Result dst;`
     *   - For each (parallelized) individual index in `Result::Size`:
     *   - `auto dst_idx = Result::FlatToMulti(i)`
     *   - get `Result` multi-index
     *   - `std::array<size_t, Rank> src_multi = [perm_arr[Rank]...]`
     *   - get `Source` multi-index
     *   - `size_t src_index = Source::MultiToFlat(src_multi)`
     *   - get `Source` flat index
     *   - dst.flat(i) = src.flat(src_index)
     *   - Assign `Source` value at that index to `Result`
     */
    template<size_t... Perm, size_t... Dims>
    auto Permute(const Tensor<Dims...> &src) {
        using Source = Tensor<Dims...>;
        using Result = PermutedTensorType<Source, Perm...>::type;
        static constexpr size_t Rank = sizeof...(Dims);
        static constexpr std::array<size_t, Rank> perm_arr = {Perm...};

        Result dst;
        ParForEach(Result::Size, [&](size_t i) {
            auto dst_idx = Result::FlatToMulti(i);
            auto src_multi = [&]<size_t... Is>(std::index_sequence<Is...>) {
                std::array<size_t, Rank> m{};
                ((m[perm_arr[Is]] = dst_idx[Is]), ...);
                return m;
            }(std::make_index_sequence<Rank>{});
            dst.flat(i) = src.flat(Source::MultiToFlat(src_multi));
        });
        return dst;
    }

    // @doc: template<size_t... Dims> auto Transpose(const Tensor<Dims...>& t)
    /**
     * Reverse all dimensions of a `Tensor<Dims...>`
     * `Permute<(sizeof...(Dims) - 1 - I)...>(s)`, for `I` in `Dims...`
     * so if `Dims` are `<3, 5, 4>` (`sizeof = 3`) we will have:
     *   - `Permute<`(3-1) - 0 = `2,` (3-1) - 1 = `1,` (3-1) - 2 = `0>`
     */
    template<size_t... Dims>
    auto Transpose(const Tensor<Dims...> &t) {
        auto f = []<size_t... I>(const Tensor<I...> &s, std::index_sequence<I...>) {
            return Permute<(sizeof...(Dims) - 1 - I)...>(s);
        };
        return f(t, std::make_index_sequence<sizeof...(Dims)>{});
    }
}
