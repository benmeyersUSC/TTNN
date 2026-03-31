#pragma once
#include <random>
#include "TensorContract.hpp"
#include "TensorReduce.hpp"

namespace TTTN {
    // @doc: static constexpr float EPS
    /** Error constant used throughout ML file to allow divisions by `0` */
    static constexpr float EPS = 1e-8f;

    // @doc: struct ReLU
    /**
     * `ActivationOp` for ***Rectified Linear Unit*** (***ReLU***)
     * `operator()` -> `[0, infinity)`
     * `prime` -> `1.0f || 0.0f`
     */
    struct ReLU {
        constexpr float operator()(const float x) const { return x > 0.f ? x : 0.f; }
        static constexpr float prime(const float a) { return a > 0.f ? 1.f : 0.f; }
    };

    // @doc: Sigmoid
    /**
     * `ActivationOp` for ***Sigmoid***
     * `operator()` -> `[0, 1.0f]`
     * `prime` -> `(0.0f, 0.25f]`
     */
    struct Sigmoid {
        constexpr float operator()(const float x) const { return 1.f / (1.f + std::exp(-x)); }
        static constexpr float prime(const float a) { return a * (1.f - a); }
    };

    // @doc: Tanh
    /**
     * `ActivationOp` for ***Hyperbolic Tangent*** (***Tanh***)
     * `operator()` -> `[-1.0f, 1.0f]`
     * `prime` -> `(0.0f, 1.0f]`
     */
    struct Tanh {
        constexpr float operator()(const float x) const { return std::tanh(x); }
        static constexpr float prime(const float a) { return 1.f - a * a; }
    };

    // @doc: Linear
    /**
     * `ActivationOp` for ***Linear*** (no activation)
     * `operator()` -> `(-infinity, infinity)`
     * `prime` -> `(-infinity, infinity)`
     */
    struct Linear {
        constexpr float operator()(const float x) const { return x; }
        static constexpr float prime(float) { return 1.f; }
    };

    // @doc: template<typename T> concept ActivationOp = FloatUnaryOp<T> && requires(float a)
    /**
     * `concept` requiring:
     * `constexpr float operator()(float x)`
     * `constexpr float prime(float a)`
     */
    template<typename T> concept ActivationOp = FloatUnaryOp<T> && requires(float a)
    {
        { T::prime(a) } -> std::convertible_to<float>;
    };


    // @doc: template<size_t N> float CrossEntropyLoss(const Tensor<N> &output, const Tensor<N> &target)
    /**
     * Computes ***Cross Entropy*** between two `Tensor`s, `output` and `target`, and returns `float`
     * Calls `Collapse<Mul, Add>(target, Map<Compose<Log, Clamp<EPS>>>(output)) * -1.f`
     */
    template<size_t N>
    float CrossEntropyLoss(const Tensor<N> &output, const Tensor<N> &target) {
        return Collapse<Mul, Add>(target, Map<Compose<Log, Clamp<EPS> > >(output)) * -1.f;
    }


    // @doc: template<size_t... Dims> void XavierInitMD(Tensor<Dims...> &W, const size_t fan_in, const size_t fan_out)
    /** ***Xavier Initializes*** a `Tensor` inplace, given `fan_in` and `fan_out` values denoting net size of input and output to a neural network layer */
    template<size_t... Dims>
    void XavierInitMD(Tensor<Dims...> &W, const size_t fan_in, const size_t fan_out) {
        static std::mt19937 rng{std::random_device{}()};
        const float limit = std::sqrt(6.f / static_cast<float>(fan_in + fan_out));
        std::uniform_real_distribution<float> dist{-limit, limit};
        for (size_t i = 0; i < Tensor<Dims...>::Size; ++i)
            W.flat(i) = dist(rng);
    }


    // @doc: template<size_t Axis, size_t... Dims> Tensor<Dims...> Softmax(const Tensor<Dims...> &x)
    /**
     * Given an `Axis` on which to normalize, perform ***Softmax*** normalization
     * Elegantly calls `BroadcastReduceMove<Axis, Div, Add>(BroadcastReduce<Axis, Compose<Exp, Sub>, Max>(x))` to first map to `a = e^(x - max)` and then to `b = a / sum(a)`
     * Shape-preserving
     */
    template<size_t Axis, size_t... Dims>
    Tensor<Dims...> Softmax(const Tensor<Dims...> &x) {
        return BroadcastReduceMove<Axis, Div, Add>(BroadcastReduce<Axis, Compose<Exp, Sub>, Max>(x));
    }

    // @doc: template<size_t Axis, size_t... Dims> Tensor<Dims...> SoftmaxPrime(const Tensor<Dims...> &grad, const Tensor<Dims...> &a)
    /**
     * Computes derivative of `Softmax`
     * Calls (efficient equivalent of) `a * BroadcastMap<Axis, Sub>(grad, BroadcastReduce<Axis, Add, Mul>(a, grad))`
     * Generalization of `a * (g - (g . a))`
     * Shape-preserving
     */
    template<size_t Axis, size_t... Dims>
    Tensor<Dims...> SoftmaxPrime(const Tensor<Dims...> &grad, const Tensor<Dims...> &a) {
        // sum of elementwise products along Axis
        auto axisDots = Reduce<Axis, Add>(a * grad);
        // copy grad in
        auto g = grad;
        // subtract dot inplace
        BroadcastApply<Axis, Sub>(g, axisDots);
        // multiply by activation inplace
        g *= a;
        return g;
    }


    template<size_t Axis, typename TensorT>
    class SoftmaxBlock;

    template<size_t Axis, size_t... Dims>
    class SoftmaxBlock<Axis, Tensor<Dims...> > {
    public:
        using InputTensor = Tensor<Dims...>;
        using OutputTensor = Tensor<Dims...>;
        auto all_params() { return std::tuple<>{}; }
        auto all_params() const { return std::tuple<>{}; }

        OutputTensor Forward(const InputTensor &x) const {
            return Softmax<Axis>(x);
        }

        // delta_A: dL/dA, a: post-softmax activation, a_prev: pre-block input (unused)
        InputTensor Backward(const OutputTensor &delta_A, const OutputTensor &a,
                             const InputTensor & /*a_prev*/) {
            return SoftmaxPrime<Axis>(delta_A, a);
        }

        template<size_t Batch>
        Tensor<Batch, Dims...> BatchedForward(const Tensor<Batch, Dims...> &X) const {
            return Softmax<Axis + 1>(X);
        }

        template<size_t Batch>
        Tensor<Batch, Dims...> BatchedBackward(const Tensor<Batch, Dims...> &delta_A,
                                               const Tensor<Batch, Dims...> &a,
                                               const Tensor<Batch, Dims...> & /*a_prev*/) {
            return SoftmaxPrime<Axis + 1>(delta_A, a);
        }
    };


    template<size_t Axis>
    struct SoftmaxLayer {
        using OutputTensor = Tensor<1>;
        template<typename InputT>
        using Resolve = SoftmaxBlock<Axis, InputT>;
    };


    template<typename L, typename TensorT>
    concept LossFunction =
            IsTensor<TensorT> &&
            requires
            {
                { L::Loss(std::declval<const TensorT &>(), std::declval<const TensorT &>()) } -> std::same_as<float>;
                { L::Grad(std::declval<const TensorT &>(), std::declval<const TensorT &>()) } -> std::same_as<TensorT>;
            };


    struct MSE {
        template<size_t... Dims>
        static float Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            static constexpr float Inv = 1.f / Tensor<Dims...>::Size;
            return Collapse<Compose<Sq, Sub>, Add>(pred, target) * Inv;
        }

        template<size_t... Dims>
        static Tensor<Dims...> Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            constexpr float inv = 2.f / static_cast<float>(Tensor<Dims...>::Size);
            return (pred - target) * inv;
        }
    };


    struct BinaryCEL {
        template<size_t... Dims>
        static float Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            const auto p_c = Map<Clamp<EPS, 1.f - EPS> >(pred);
            return -(Collapse<Mul, Add>(target, Map<Log>(p_c)) +
                     Collapse<Mul, Add>(Map<OneMinus>(target), Map<Compose<Log, OneMinus> >(p_c)));
        }

        template<size_t... Dims>
        static Tensor<Dims...> Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            return pred.zip(target, [](float p, float t) {
                return (p - t) / (p * (1.f - p) + EPS);
            });
        }
    };


    struct CEL {
        template<size_t... Dims>
        static float Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            return Collapse<Mul, Add>(target, Map<Compose<Log, Clamp<EPS> > >(pred)) * -1.f;
        }

        template<size_t... Dims>
        static Tensor<Dims...> Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            return Zip<Compose<Neg, Div> >(target, Map<Clamp<EPS> >(pred));
        }
    };


    template<size_t Batch, size_t N>
    float BatchAccuracy(const Tensor<Batch, N> &pred, const Tensor<Batch, N> &labels) {
        const auto p_correct = Reduce<1, Add>(pred * labels); // Tensor<Batch>
        const auto p_max = Reduce<1, Max>(pred); // Tensor<Batch>
        const float n = Reduce<0, Add>(Map<Step<1e-5f> >(p_max - p_correct)).flat(0);
        return 100.f * n / static_cast<float>(Batch);
    }
};
