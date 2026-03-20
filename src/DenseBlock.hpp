#pragma once
#include "TensorOps.hpp"

namespace TTTN {
    enum class ActivationFunction { Linear, Sigmoid, ReLU, Softmax, Tanh };

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


    // DENSE BLOCK
    // weights, bias, activation, Adam moments
    // implements the Block interface for a fully-connected layer with activation
    // InSize and OutSize define the Block's tensor contract; Act is applied in Forward/Backward
    template<size_t In, size_t Out, ActivationFunction Act_ = ActivationFunction::Linear>
    struct DenseBlock {
        static constexpr size_t InSize = In;
        static constexpr size_t OutSize = Out;
        static constexpr ActivationFunction Act = Act_;

        Tensor<Out, In> W;
        Tensor<Out> b{};

        Tensor<Out, In> dW{};
        Tensor<Out> dB{};

        // Adam first and second moments (0 init)
        Tensor<Out, In> mW{}, vW{};
        Tensor<Out> mb{}, vb{};

        DenseBlock() { XavierInit(W); }

        // forward pass
        // uses Einsum to contract 2nd and 1st dimensions from W and x, respectively
        // then calls activation
        Tensor<Out> Forward(const Tensor<In> &x) const {
            // MATVEC contracts matrix's columns with column-vec's rows --> Tensor<NumRowsInWeight>
            auto z = Einsum<1, 0>(W, x) + b;
            return Activate(z, Act);
        }

        // delta_A is dL/dA (gradient wrt my output activation)
        // a is my output (for ActivatePrime), a_prev is my input (for dW)
        // returns dL/dA_prev
        Tensor<In> Backward(const Tensor<Out> &delta_A, const Tensor<Out> &a,
                            const Tensor<In> &a_prev) {
            // peel off activation derivative to get dL/dZ
            const auto delta_Z = ActivatePrime(delta_A, a, Act);

            // dW = OUTER(delta_Z, a_prev)
            // delta_Z: Tensor<Out>, a_prev: Tensor<In>....outer prod --> Tensor<Out,In>, same dim as W
            dW = Einsum(delta_Z, a_prev);
            dB = delta_Z;

            // pass gradient upstream, defining dL/dA_prev:
            //      W: Tensor<Out,In>, delta_Z: Tensor<Out>...contract first axis of each --> Tensor<In>, same dim as a_prev
            // (same thing as DOT(W.Transpose(), delta_Z), but Einsum obviates Transpose!)
            return Einsum<0, 0>(W, delta_Z);
        }

        // Adam update.
        // mCorr and vCorr are precomputed by NN. at the beginning, (low mT), corrections amplify
        // moments from 0-bias, but eventually corrections approach 1
        void Update(float adamBeta1, float adamBeta2, float lr, float mCorr, float vCorr, float eps = 1e-8) {
            // for each Weight and Bias, subtract LR * adjusted_First_Moment / sqrt(adjusted_Second_Moment)
            //      first moment approximates consistency of direction of update
            //      second moment approximates inverse of smoothness of local terrain on loss landscape
            for (size_t i = 0; i < Out * In; ++i) {
                const float g = dW.flat(i);
                mW.flat(i) = adamBeta1 * mW.flat(i) + (1.f - adamBeta1) * g;
                vW.flat(i) = adamBeta2 * vW.flat(i) + (1.f - adamBeta2) * g * g;
                W.flat(i) -= lr * (mW.flat(i) * mCorr) / (std::sqrt(vW.flat(i) * vCorr) + eps);
            }
            for (size_t i = 0; i < Out; ++i) {
                const float g = dB.flat(i);
                mb.flat(i) = adamBeta1 * mb.flat(i) + (1.f - adamBeta1) * g;
                vb.flat(i) = adamBeta2 * vb.flat(i) + (1.f - adamBeta2) * g * g;
                b.flat(i) -= lr * (mb.flat(i) * mCorr) / (std::sqrt(vb.flat(i) * vCorr) + eps);
            }
        }

        void Save(std::ofstream &f) const {
            W.Save(f);
            b.Save(f);
        }

        void Load(std::ifstream &f) {
            W.Load(f);
            b.Load(f);
        }
    };
};
