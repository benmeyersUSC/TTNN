#pragma once
#include <random>
#include "TensorOps.hpp"

namespace TTTN {
    static constexpr float EPS = 1e-8f;

    enum class ActivationFunction { Linear, Sigmoid, ReLU, Softmax, Tanh };

    // CROSS ENTROPY LOSS
    template<size_t N>
    float CrossEntropyLoss(const Tensor<N> &output, const Tensor<N> &target) {
        auto indices = std::views::iota(size_t{0}, N);

        return std::accumulate(indices.begin(), indices.end(), 0.0f,
                               [&target, &output](float current_loss, size_t i) {
                                   return current_loss - target.flat(i) * std::log(std::max(output.flat(i), EPS));
                               }
        );
    }

    // Xavier init for arbitrary-rank tensors.
    // fan_in / fan_out are the total element counts of the input and output spaces.
    template<size_t... Dims>
    void XavierInitMD(Tensor<Dims...> &W, size_t fan_in, size_t fan_out) {
        static std::mt19937 rng{std::random_device{}()};
        const float limit = std::sqrt(6.f / static_cast<float>(fan_in + fan_out));
        std::uniform_real_distribution<float> dist{-limit, limit};
        W.apply([&](float &x) { x = dist(rng); });
    }

    // activation function
    template<size_t N>
    Tensor<N> Activate(const Tensor<N> &z, const ActivationFunction act) {
        switch (act) {
            case ActivationFunction::ReLU:
                return z.map([](const float x) {
                    return x > 0.f ? x : 0.f;
                });
            case ActivationFunction::Sigmoid:
                return z.map([](const float x) {
                    return 1.f / (1.f + std::exp(-x));
                });
            case ActivationFunction::Tanh: return z.map([](const float x) { return std::tanh(x); });
            case ActivationFunction::Softmax: {
                // subtract max for numerical stability
                float maxV = z.flat(0);
                for (size_t i = 1; i < N; ++i) {
                    if (z.flat(i) > maxV) {
                        maxV = z.flat(i);
                    }
                }
                auto a = z.map([maxV](const float x) { return std::exp(x - maxV); });
                float sum = 0.f;
                for (size_t i = 0; i < N; ++i) {
                    sum += a.flat(i);
                }
                a.apply([sum](float &x) { x /= sum; });
                return a;
            }
            case ActivationFunction::Linear:
            default: return z;
        }
    }

    // given upstream gradient (dL/da) and post-activation a --> dL/dz
    template<size_t N>
    Tensor<N> ActivatePrime(const Tensor<N> &grad, const Tensor<N> &a, const ActivationFunction act) {
        switch (act) {
            case ActivationFunction::ReLU:
                return grad.zip(a, [](const float g, const float ai) { return g * (ai > 0.f ? 1.f : 0.f); });
            case ActivationFunction::Sigmoid:
                return grad.zip(a, [](const float g, const float ai) { return g * ai * (1.f - ai); });
            case ActivationFunction::Tanh:
                return grad.zip(a, [](const float g, const float ai) { return g * (1.f - ai * ai); });
            case ActivationFunction::Softmax: {
                float dot = 0.f;
                for (size_t i = 0; i < N; ++i) {
                    dot += a.flat(i) * grad.flat(i);
                }
                return a.zip(grad, [dot](const float ai, const float gi) { return ai * (gi - dot); });
            }
            case ActivationFunction::Linear:
            default: return grad;
        }
    }

    // BATCHED ACTIVATE
    // Apply activation per-sample to Tensor<Batch, N>.
    // For element-wise activations, this degenerates to a flat map/zip over all Batch*N elements.
    // For Softmax, applies per-row (each row is one sample's pre-activation vector).
    template<size_t Batch, size_t N>
    Tensor<Batch, N> BatchedActivate(const Tensor<Batch, N> &Z, const ActivationFunction act) {
        switch (act) {
            case ActivationFunction::ReLU:
                return Z.map([](const float x) { return x > 0.f ? x : 0.f; });
            case ActivationFunction::Sigmoid:
                return Z.map([](const float x) { return 1.f / (1.f + std::exp(-x)); });
            case ActivationFunction::Tanh:
                return Z.map([](const float x) { return std::tanh(x); });
            case ActivationFunction::Softmax: {
                Tensor<Batch, N> A;
                for (size_t b = 0; b < Batch; b++) {
                    float maxV = Z(b, 0);
                    for (size_t i = 1; i < N; i++) {
                        if (Z(b, i) > maxV) maxV = Z(b, i);
                    }
                    float sum = 0.f;
                    for (size_t i = 0; i < N; i++) {
                        A(b, i) = std::exp(Z(b, i) - maxV);
                        sum += A(b, i);
                    }
                    for (size_t i = 0; i < N; i++) {
                        A(b, i) /= sum;
                    }
                }
                return A;
            }
            case ActivationFunction::Linear:
            default: return Z;
        }
    }

    // BATCHED ACTIVATE PRIME
    // Peel off activation derivative for Tensor<Batch, N>.
    // For element-wise activations, flat zip over all Batch*N elements.
    // For Softmax, per-row Jacobian-vector product.
    template<size_t Batch, size_t N>
    Tensor<Batch, N> BatchedActivatePrime(const Tensor<Batch, N> &grad, const Tensor<Batch, N> &a,
                                          const ActivationFunction act) {
        switch (act) {
            case ActivationFunction::ReLU:
                return grad.zip(a, [](const float g, const float ai) { return g * (ai > 0.f ? 1.f : 0.f); });
            case ActivationFunction::Sigmoid:
                return grad.zip(a, [](const float g, const float ai) { return g * ai * (1.f - ai); });
            case ActivationFunction::Tanh:
                return grad.zip(a, [](const float g, const float ai) { return g * (1.f - ai * ai); });
            case ActivationFunction::Softmax: {
                Tensor<Batch, N> result;
                for (size_t b = 0; b < Batch; b++) {
                    float dot = 0.f;
                    for (size_t i = 0; i < N; i++) {
                        dot += a(b, i) * grad(b, i);
                    }
                    for (size_t i = 0; i < N; i++) {
                        result(b, i) = a(b, i) * (grad(b, i) - dot);
                    }
                }
                return result;
            }
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
    template<size_t Axis, typename TensorT>
    class SoftmaxBlock;

    template<size_t Axis, size_t... Dims>
    class SoftmaxBlock<Axis, Tensor<Dims...>> {
    public:
        using InputTensor  = Tensor<Dims...>;
        using OutputTensor = Tensor<Dims...>;
        static constexpr size_t ParamCount = 0;

        OutputTensor Forward(const InputTensor& x) const { return Softmax<Axis>(x); }

        void ZeroGrad() {}

        // delta_A: dL/dA, a: post-softmax activation, a_prev: pre-block input (unused)
        InputTensor Backward(const OutputTensor& delta_A, const OutputTensor& a,
                             const InputTensor& /*a_prev*/) {
            return SoftmaxPrime<Axis>(delta_A, a);
        }

        template<size_t Batch>
        Tensor<Batch, Dims...> BatchedForward(const Tensor<Batch, Dims...>& X) const {
            Tensor<Batch, Dims...> result;
            for (size_t b = 0; b < Batch; ++b) {
                InputTensor x_b;
                for (size_t i = 0; i < InputTensor::Size; ++i)
                    x_b.flat(i) = X.flat(b * InputTensor::Size + i);
                const auto out = Forward(x_b);
                for (size_t i = 0; i < OutputTensor::Size; ++i)
                    result.flat(b * OutputTensor::Size + i) = out.flat(i);
            }
            return result;
        }

        template<size_t Batch>
        Tensor<Batch, Dims...> BatchedBackward(const Tensor<Batch, Dims...>& delta_A,
                                               const Tensor<Batch, Dims...>& a,
                                               const Tensor<Batch, Dims...>& /*a_prev*/) {
            Tensor<Batch, Dims...> result;
            for (size_t b = 0; b < Batch; ++b) {
                OutputTensor dA_b, a_b;
                for (size_t i = 0; i < OutputTensor::Size; ++i) {
                    dA_b.flat(i) = delta_A.flat(b * OutputTensor::Size + i);
                    a_b.flat(i)  = a.flat(b * OutputTensor::Size + i);
                }
                const auto upstream = SoftmaxPrime<Axis>(dA_b, a_b);
                for (size_t i = 0; i < InputTensor::Size; ++i)
                    result.flat(b * InputTensor::Size + i) = upstream.flat(i);
            }
            return result;
        }

        void Update(float, float, float, float, float, float) {}
        void Save(std::ofstream&) const {}
        void Load(std::ifstream&) {}
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
        requires {
            { L::Loss(std::declval<const TensorT&>(), std::declval<const TensorT&>()) } -> std::same_as<float>;
            { L::Grad(std::declval<const TensorT&>(), std::declval<const TensorT&>()) } -> std::same_as<TensorT>;
        };

    // MSE: mean squared error, any-rank tensor.
    struct MSE {
        template<size_t... Dims>
        static float Loss(const Tensor<Dims...>& pred, const Tensor<Dims...>& target) {
            float s = 0.f;
            for (size_t i = 0; i < Tensor<Dims...>::Size; ++i) {
                const float d = pred.flat(i) - target.flat(i);
                s += d * d;
            }
            return s / static_cast<float>(Tensor<Dims...>::Size);
        }
        template<size_t... Dims>
        static Tensor<Dims...> Grad(const Tensor<Dims...>& pred, const Tensor<Dims...>& target) {
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
        static float Loss(const Tensor<Dims...>& pred, const Tensor<Dims...>& target) {
            float s = 0.f;
            for (size_t i = 0; i < Tensor<Dims...>::Size; ++i) {
                const float p = std::max(std::min(pred.flat(i), 1.f - EPS), EPS);
                s -= target.flat(i) * std::log(p) + (1.f - target.flat(i)) * std::log(1.f - p);
            }
            return s;
        }
        template<size_t... Dims>
        static Tensor<Dims...> Grad(const Tensor<Dims...>& pred, const Tensor<Dims...>& target) {
            return pred.zip(target, [](float p, float t) {
                return (p - t) / (p * (1.f - p) + EPS);
            });
        }
    };

    // CEL: cross-entropy loss, any-rank tensor.
    // Assumes softmax output; gradient = −target / pred.
    struct CEL {
        template<size_t... Dims>
        static float Loss(const Tensor<Dims...>& pred, const Tensor<Dims...>& target) {
            float s = 0.f;
            for (size_t i = 0; i < Tensor<Dims...>::Size; ++i)
                s -= target.flat(i) * std::log(std::max(pred.flat(i), EPS));
            return s;
        }
        template<size_t... Dims>
        static Tensor<Dims...> Grad(const Tensor<Dims...>& pred, const Tensor<Dims...>& target) {
            return pred.zip(target, [](float p, float t) { return -t / std::max(p, EPS); });
        }
    };
};
