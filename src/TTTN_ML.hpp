#pragma once
#include <random>
#include "TensorOps.hpp"

namespace TTTN {
    static constexpr float EPS = 1e-8f;

    enum class ActivationFunction { Linear, Sigmoid, ReLU, Tanh };

    // CROSS ENTROPY LOSS
    template<size_t N>
    float CrossEntropyLoss(const Tensor<N> &output, const Tensor<N> &target) {
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
        W.apply([&](float &x) { x = dist(rng); });
    }

    // SOFTMAX
    // Two ReduceBroadcast passes:
    //   1. reduce=max, apply=exp(a-m)   → numerically stable exps
    //   2. reduce=sum, apply=e/s        → normalize
    template<size_t Axis, size_t... Dims>
    Tensor<Dims...> Softmax(const Tensor<Dims...> &x) {
        const auto exps = ReduceBroadcast<Axis>(x,
                                                -std::numeric_limits<float>::infinity(),
                                                [](float a, float b) { return std::max(a, b); },
                                                [](float a, float m) { return std::exp(a - m); });
        return ReduceBroadcast<Axis>(exps,
                                     0.f,
                                     std::plus<float>{},
                                     std::divides<float>{});
    }

    // VJP of softmax: δx_i = a_i * (δy_i − dot(δy, a))  per pool along Axis.
    template<size_t Axis, size_t... Dims>
    Tensor<Dims...> SoftmaxPrime(const Tensor<Dims...> &grad, const Tensor<Dims...> &a) {
        return a * BroadcastApply<Axis>(grad, ReduceSum<Axis>(a * grad),
                                        [](float g, float d) { return g - d; });
    }

    // activation function — element-wise only.
    // Softmax is not an element-wise activation; use SoftmaxLayer<Axis> as a standalone block.
    template<size_t N>
    Tensor<N> Activate(const Tensor<N> &z, const ActivationFunction act) {
        switch (act) {
            case ActivationFunction::ReLU: return z.map([](float x) { return x > 0.f ? x : 0.f; });
            case ActivationFunction::Sigmoid: return z.map([](float x) { return 1.f / (1.f + std::exp(-x)); });
            case ActivationFunction::Tanh: return z.map([](float x) { return std::tanh(x); });
            case ActivationFunction::Linear:
            default: return z;
        }
    }

    // given upstream gradient (dL/da) and post-activation a --> dL/dz
    template<size_t N>
    Tensor<N> ActivatePrime(const Tensor<N> &grad, const Tensor<N> &a, const ActivationFunction act) {
        switch (act) {
            case ActivationFunction::ReLU:
                return grad.zip(a, [](float g, float ai) { return g * (ai > 0.f ? 1.f : 0.f); });
            case ActivationFunction::Sigmoid:
                return grad.zip(a, [](float g, float ai) { return g * ai * (1.f - ai); });
            case ActivationFunction::Tanh:
                return grad.zip(a, [](float g, float ai) { return g * (1.f - ai * ai); });
            case ActivationFunction::Linear:
            default: return grad;
        }
    }

    // BATCHED ACTIVATE — element-wise activations over Tensor<Batch, N>.
    template<size_t Batch, size_t N>
    Tensor<Batch, N> BatchedActivate(const Tensor<Batch, N> &Z, const ActivationFunction act) {
        switch (act) {
            case ActivationFunction::ReLU: return Z.map([](float x) { return x > 0.f ? x : 0.f; });
            case ActivationFunction::Sigmoid: return Z.map([](float x) { return 1.f / (1.f + std::exp(-x)); });
            case ActivationFunction::Tanh: return Z.map([](float x) { return std::tanh(x); });
            case ActivationFunction::Linear:
            default: return Z;
        }
    }

    // BATCHED ACTIVATE PRIME — element-wise activation derivatives over Tensor<Batch, N>.
    template<size_t Batch, size_t N>
    Tensor<Batch, N> BatchedActivatePrime(const Tensor<Batch, N> &grad, const Tensor<Batch, N> &a,
                                          const ActivationFunction act) {
        switch (act) {
            case ActivationFunction::ReLU:
                return grad.zip(a, [](float g, float ai) { return g * (ai > 0.f ? 1.f : 0.f); });
            case ActivationFunction::Sigmoid:
                return grad.zip(a, [](float g, float ai) { return g * ai * (1.f - ai); });
            case ActivationFunction::Tanh:
                return grad.zip(a, [](float g, float ai) { return g * (1.f - ai * ai); });
            case ActivationFunction::Linear:
            default: return grad;
        }
    }

    // SOFTMAX BLOCK
    // Shape-preserving, parameter-free block: applies Softmax<Axis> forward and
    // the softmax VJP (SoftmaxPrime<Axis>) backward.
    //
    // Forward:  a = Softmax<Axis>(x)
    // Backward: δx_i = a_i * (δy_i - dot(δy, a))   per pool along Axis
    //
    // InputTensor == OutputTensor == TensorT (shape flows through unchanged).
    // ParamCount == 0; Update/ZeroGrad/Save/Load are all no-ops.
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
        static constexpr size_t ParamCount = 0;

        OutputTensor Forward(const InputTensor &x) const { return Softmax<Axis>(x); }

        static void ZeroGrad() {
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

        static void Update(float, float, float, float, float, float) {
        }

        static void Save(std::ofstream &) {
        }

        static void Load(std::ifstream &) {
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
            const auto diff = pred.zip(target, [](float p, float t) { return p - t; });
            return Contract(diff, diff) / static_cast<float>(Tensor<Dims...>::Size);
        }

        template<size_t... Dims>
        static Tensor<Dims...> Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            constexpr float inv = 2.f / static_cast<float>(Tensor<Dims...>::Size);
            return pred.zip(target, [](float p, float t) { return inv * (p - t); });
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
            return -Contract(target, pred.map([](float p) { return std::log(std::max(p, EPS)); }));
        }

        template<size_t... Dims>
        static Tensor<Dims...> Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            return pred.zip(target, [](float p, float t) { return -t / std::max(p, EPS); });
        }
    };

    // @doc: template<size_t Batch, size_t N> float BatchAccuracy(const Tensor<Batch, N>& pred, const Tensor<Batch, N>& labels)
    /** Returns the percentage of correctly classified samples in a batch. `labels` must be one-hot. Correct iff `argmax(pred[b]) == argmax(labels[b])`, computed via `ReduceSum<1>(pred ⊙ labels)` (probability assigned to the true class) vs `ReduceMax<1>(pred)` (highest predicted probability) — no explicit argmax loop required. */
    template<size_t Batch, size_t N>
    float BatchAccuracy(const Tensor<Batch, N> &pred, const Tensor<Batch, N> &labels) {
        const auto p_correct = ReduceSum<1>(pred * labels); // Tensor<Batch>
        const auto p_max = ReduceMax<1>(pred); // Tensor<Batch>
        int n = 0;
        for (size_t b = 0; b < Batch; ++b)
            if (p_max.flat(b) - p_correct.flat(b) < 1e-5f) ++n;
        return 100.f * n / static_cast<float>(Batch);
    }
};
