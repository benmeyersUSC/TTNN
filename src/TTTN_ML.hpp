#pragma once
#include <random>
#include "TensorOps.hpp"

namespace TTTN {
    static constexpr float EPS = 1e-8f;

    // ── Activation op tags ───────────────────────────────────────────────────────
    // Each tag satisfies FloatUnaryOp: use with z.map(Act{}), MapApply<Act>(z), or
    // Tensor::zip for the prime. prime(a) is the derivative wrt the post-activation value.

    struct ReLU {
        constexpr float operator()(float x)  const { return x > 0.f ? x : 0.f; }
        static constexpr float prime(float a)       { return a > 0.f ? 1.f : 0.f; }
    };
    struct Sigmoid {
        constexpr float operator()(float x)  const { return 1.f / (1.f + std::exp(-x)); }
        static constexpr float prime(float a)       { return a * (1.f - a); }
    };
    struct Tanh {
        constexpr float operator()(float x)  const { return std::tanh(x); }
        static constexpr float prime(float a)       { return 1.f - a * a; }
    };
    struct Linear {
        constexpr float operator()(float x)  const { return x; }
        static constexpr float prime(float)         { return 1.f; }
    };

    // Concept: FloatUnaryOp + has prime(float) -> float
    template<typename T>
    concept ActivationOp = FloatUnaryOp<T> && requires(float a) {
        { T::prime(a) } -> std::convertible_to<float>;
    };

    // CROSS ENTROPY LOSS
    template<size_t N>
    float CrossEntropyLoss(const Tensor<N> &output, const Tensor<N> &target) {
        // collapse via sum of all elementwise pairs, where pairs are target[i] and log(output[i])
        return -Contract(
            target, output.map([](const float p) { return std::log(std::max(p, EPS)); })
        );
    }

    // Xavier init for arbitrary-rank tensors.
    // fan_in / fan_out are the total element counts of the input and output spaces.
    template<size_t... Dims>
    void XavierInitMD(Tensor<Dims...> &W, const size_t fan_in, const size_t fan_out) {
        static std::mt19937 rng{std::random_device{}()};
        const float limit = std::sqrt(6.f / static_cast<float>(fan_in + fan_out));
        std::uniform_real_distribution<float> dist{-limit, limit};
        W.apply([&](float &x) {
            x = dist(rng);
        });
    }

    // SOFTMAX
    // Two ReduceBroadcast passes:
    //   1. reduce=max, apply=exp(a-m)   → numerically stable exps
    //   2. reduce=sum, apply=e/s        → normalize
    template<size_t Axis, size_t... Dims>
    Tensor<Dims...> Softmax(const Tensor<Dims...> &x) {
        // reduce via/to max
        // broadcast back with e^(v - max)
        const auto exps = ReduceBroadcast<Axis>(x, Max::identity, Max{}, Compose<Exp, Sub>{});
        // reduce via/to sum
        // broadcast scaling by sum
        return ReduceBroadcast<Axis, Add, Div>(exps);
    }

    // VJP of softmax: δx_i = a_i * (δy_i − dot(δy, a))  per pool along Axis.
    template<size_t Axis, size_t... Dims>
    Tensor<Dims...> SoftmaxPrime(const Tensor<Dims...> &grad, const Tensor<Dims...> &a) {
        return a * BroadcastApply<Axis, Sub>(grad, ReduceApply<Axis, Add>(a * grad));
    }


    // SOFTMAX BLOCK
    // Shape-preserving, parameter-free block: applies Softmax<Axis> forward and
    // the softmax VJP (SoftmaxPrime<Axis>) backward.
    //
    // Forward:  a = Softmax<Axis>(x)
    // Backward: δx_i = a_i * (δy_i - dot(δy, a))   per pool along Axis
    //
    // InputTensor == OutputTensor == TensorT (shape flows through unchanged).
    // all_params() returns std::tuple<>{} — no parameters, TTN handles the rest.
    //
    // Batched: prepending a Batch axis shifts the target axis by 1, so
    // BatchedForward/Backward delegate to Softmax<Axis+1> / SoftmaxPrime<Axis+1>.
    template<size_t Axis, typename TensorT>
    class SoftmaxBlock;

    template<size_t Axis, size_t... Dims>
    class SoftmaxBlock<Axis, Tensor<Dims...> > {
    public:
        using InputTensor = Tensor<Dims...>;
        using OutputTensor = Tensor<Dims...>;
        // @doc: auto all_params()
        /** Returns `std::tuple<>{}` — no parameters; TTN bulk helpers become no-ops automatically */
        auto all_params()       { return std::tuple<>{}; }
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

    // SoftmaxLayer<Axis>: recipe for SoftmaxBlock.
    // OutputTensor = Tensor<1> is a placeholder satisfying the Block concept self-check;
    // actual output shape is always identical to the input shape (determined via Resolve).
    template<size_t Axis>
    struct SoftmaxLayer {
        using OutputTensor = Tensor<1>;
        template<typename InputT>
        using Resolve = SoftmaxBlock<Axis, InputT>;
    };

    // ── Loss functions ────────────────────────────────────────────────────────
    //
    // LossFunction<L, TensorT>: L must expose static Loss and Grad methods
    // matching the concrete output tensor type of the network.

    template<typename L, typename TensorT>
    concept LossFunction =
            IsTensor<TensorT> &&
            requires
            {
                { L::Loss(std::declval<const TensorT &>(), std::declval<const TensorT &>()) } -> std::same_as<float>;
                { L::Grad(std::declval<const TensorT &>(), std::declval<const TensorT &>()) } -> std::same_as<TensorT>;
            };

    // MSE: mean squared error, any-rank tensor.
    struct MSE {
        template<size_t... Dims>
        static float Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            static constexpr float Inv = 1.f / Tensor<Dims...>::Size;
            //
            return Collapse(pred, target,
                            0.0f,
                            [](float p, float t) { float d = p - t; return d * d; },
                            std::plus<float>{}
                   ) * Inv;
        }

        template<size_t... Dims>
        static Tensor<Dims...> Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            constexpr float inv = 2.f / static_cast<float>(Tensor<Dims...>::Size);
            return pred.zip(target, [](const float p, const float t) { return inv * (p - t); });
        }
    };

    // BinaryCEL: binary cross-entropy, any-rank tensor.
    // Assumes sigmoid output (one output per independent binary prediction).
    // Loss = -[t*log(p) + (1-t)*log(1-p)]
    // Grad = (p-t) / (p*(1-p) + eps)  — after peeling sigmoid in Backward, net grad = p-t.
    struct BinaryCEL {
        template<size_t... Dims>
        static float Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            float s = 0.f;
            for (size_t i = 0; i < Tensor<Dims...>::Size; ++i) {
                const float p = std::max(std::min(pred.flat(i), 1.f - EPS), EPS);
                s -= target.flat(i) * std::log(p) + (1.f - target.flat(i)) * std::log(1.f - p);
            }
            return s;
        }

        template<size_t... Dims>
        static Tensor<Dims...> Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            return pred.zip(target, [](float p, float t) {
                return (p - t) / (p * (1.f - p) + EPS);
            });
        }
    };

    // CEL: cross-entropy loss, any-rank tensor.
    // Assumes softmax output; gradient = −target / pred.
    struct CEL {
        template<size_t... Dims>
        static float Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            return Collapse(target, pred.map([](float p) { return std::log(std::max(p, EPS)); }),
                            0.0f,
                            [](float t, float log_p) { return t * log_p; },
                            std::plus<float>{}) * -1.f;
        }

        template<size_t... Dims>
        static Tensor<Dims...> Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            return pred.zip(target, [](float p, float t) { return -t / std::max(p, EPS); });
        }
    };

    // @doc: template<size_t Batch, size_t N> float BatchAccuracy(const Tensor<Batch, N>& pred, const Tensor<Batch, N>& labels)
    /** Returns the percentage of correctly classified samples in a batch. `labels` must be one-hot. Correct iff `argmax(pred[b]) == argmax(labels[b])`, computed via `ReduceApply<1, Add>(pred ⊙ labels)` (probability assigned to the true class) vs `ReduceApply<1, Max>(pred)` (highest predicted probability) — no explicit argmax loop required. */
    template<size_t Batch, size_t N>
    float BatchAccuracy(const Tensor<Batch, N> &pred, const Tensor<Batch, N> &labels) {
        const auto p_correct = ReduceApply<1, Add>(pred * labels); // Tensor<Batch>
        const auto p_max     = ReduceApply<1, Max>(pred);          // Tensor<Batch>
        int n = 0;
        for (size_t b = 0; b < Batch; ++b)
            if (p_max.flat(b) - p_correct.flat(b) < 1e-5f) ++n;
        return 100.f * n / static_cast<float>(Batch);
    }
};
