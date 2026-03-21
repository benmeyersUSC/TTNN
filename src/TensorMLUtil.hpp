#pragma once
#include <random>

namespace TTTN {
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

    // xavier init for controlled init variance
    template<size_t In, size_t Out>
    void XavierInit(Tensor<Out, In> &W) {
        static std::mt19937 rng{std::random_device{}()};
        const float limit = std::sqrt(6.f / static_cast<float>(In + Out));
        std::uniform_real_distribution<float> dist{-limit, limit};
        W.apply([&dist](float &x) { x = dist(rng); });
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
};
