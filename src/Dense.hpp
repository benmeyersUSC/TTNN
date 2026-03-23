#pragma once
#include "TensorOps.hpp"
#include "TTTN_ML.hpp"

namespace TTTN {

    // Generalized element-wise activate / activate-prime for any tensor shape.
    // Softmax is intentionally omitted — it requires a designated reduction axis
    // and doesn't generalize to arbitrary rank without specifying it explicitly.
    // Use SoftmaxLayer<Axis> as a standalone block instead.
    template<size_t... Dims>
    Tensor<Dims...> ActivateMD(const Tensor<Dims...> &z, ActivationFunction act) {
        switch (act) {
            case ActivationFunction::ReLU:
                return z.map([](float x) { return x > 0.f ? x : 0.f; });
            case ActivationFunction::Sigmoid:
                return z.map([](float x) { return 1.f / (1.f + std::exp(-x)); });
            case ActivationFunction::Tanh:
                return z.map([](float x) { return std::tanh(x); });
            case ActivationFunction::Linear:
            default:
                return z;
        }
    }

    template<size_t... Dims>
    Tensor<Dims...> ActivatePrimeMD(const Tensor<Dims...> &grad, const Tensor<Dims...> &a,
                                    ActivationFunction act) {
        switch (act) {
            case ActivationFunction::ReLU:
                return grad.zip(a, [](float g, float ai) { return g * (ai > 0.f ? 1.f : 0.f); });
            case ActivationFunction::Sigmoid:
                return grad.zip(a, [](float g, float ai) { return g * ai * (1.f - ai); });
            case ActivationFunction::Tanh:
                return grad.zip(a, [](float g, float ai) { return g * (1.f - ai * ai); });
            case ActivationFunction::Linear:
            default:
                return grad;
        }
    }

    // Permutation that block-swaps the two halves of W's axis list:
    //   W   = Tensor<OutDims..., InDims...>   (N_out axes then N_in axes)
    //   W_T = Tensor<InDims..., OutDims...>   (N_in  axes then N_out axes)
    // Perm[i]          = N_out + i   for i < N_in    (InDims move to front)
    // Perm[N_in + i]   = i           for i < N_out   (OutDims move to back)
    template<size_t N_out, size_t N_in>
    struct WTBlockSwapPerm {
        static constexpr auto value = [] {
            std::array<size_t, N_out + N_in> p{};
            for (size_t i = 0; i < N_in;  ++i) p[i]        = N_out + i;
            for (size_t i = 0; i < N_out; ++i) p[N_in + i] = i;
            return p;
        }();
    };


    // DENSE BLOCK
    // Weights, bias, activation, Adam moments.
    // Implements the Block interface for a fully-connected layer with activation.
    // InputTensor and OutputTensor can be any rank — this is the fully general version.
    //
    // W = Tensor<OutDims..., InDims...>   linear map from In-space to Out-space
    // b = Tensor<OutDims...>              bias in Out-space
    //
    // Forward:  z = ΣΠ<N_in>(W, x) + b        contracts W's last N_in axes with x (generalised matvec)
    //           a = Activate(z)
    //
    // Backward: delta_z  = ActivatePrime(delta_A, a)          peel off activation derivative → dL/dZ
    //           dW      += ΣΠ<0>(delta_z, a_prev)             outer product → Tensor<OutDims...,InDims...>
    //           dB      += delta_z
    //           dL/dx    = ΣΠ<N_out>(W_T, delta_z)            generalises W^T · delta_z; W_T = Tensor<InDims...,OutDims...>
    //
    // caller must ZeroGrad() before the first Backward in each training step

    template<typename InT, typename OutT, ActivationFunction Act_ = ActivationFunction::Linear>
    class DenseMDBlock;

    template<size_t... InDims, size_t... OutDims, ActivationFunction Act_>
    class DenseMDBlock<Tensor<InDims...>, Tensor<OutDims...>, Act_> {
    public:
        using InputTensor  = Tensor<InDims...>;
        using OutputTensor = Tensor<OutDims...>;
        static constexpr ActivationFunction Act = Act_;

    private:
        static constexpr size_t N_in  = sizeof...(InDims);
        static constexpr size_t N_out = sizeof...(OutDims);
        using W_Type = Tensor<OutDims..., InDims...>;

        W_Type       W;
        OutputTensor b{};
        W_Type       dW{};
        OutputTensor dB{};
        W_Type       mW{}, vW{};
        OutputTensor mb{}, vb{};

    public:
        static constexpr size_t ParamCount = W_Type::Size + OutputTensor::Size;

        DenseMDBlock() { XavierInitMD(W, InputTensor::Size, OutputTensor::Size); }

        // ΣΠ<N_in>(W, x) contracts W's last N_in axes with x — generalises matrix-vector product
        OutputTensor Forward(const InputTensor &x) const {
            return ActivateMD(ΣΠ<N_in>(W, x) + b, Act);
        }

        void ZeroGrad() { dW.fill(0.f); dB.fill(0.f); }

        // delta_A is dL/dA (gradient wrt my output activation)
        // a is my output (for ActivatePrime), a_prev is my input (for dW)
        // returns dL/dA_prev
        InputTensor Backward(const OutputTensor &delta_A, const OutputTensor &a,
                             const InputTensor  &a_prev) {
            // peel off activation derivative to get dL/dZ
            const auto delta_z = ActivatePrimeMD(delta_A, a, Act);

            // dW += outer product of delta_z and a_prev → Tensor<OutDims..., InDims...>, same shape as W
            dW += ΣΠ<0>(delta_z, a_prev);
            dB += delta_z;

            // pass gradient upstream: W_T contracts its OutDims axes with delta_z → Tensor<InDims...>
            // generalises W^T · delta_z
            const auto W_T = PermuteFromHolder<WTBlockSwapPerm<N_out, N_in>>(
                W, std::make_index_sequence<N_out + N_in>{}
            );
            return ΣΠ<N_out>(W_T, delta_z);
        }

        // batched forward: ONE ΣΠ over the whole batch — no per-sample loop.
        // ΣΠ<N_in>(X, W_T) contracts InDims of X with InDims of W_T → Tensor<Batch, OutDims...>
        // This gives Batch×OutSize parallel jobs in a single dispatch instead of Batch serial dispatches.
        template<size_t Batch>
        Tensor<Batch, OutDims...> BatchedForward(const Tensor<Batch, InDims...> &X) const {
            // permute to Tensor<In, Out>
            const auto W_T = PermuteFromHolder<WTBlockSwapPerm<N_out, N_in>>(W, std::make_index_sequence<N_out + N_in>{});
            // contract only on input dimensions
            // add bias
            // apply activation
            return ActivateMD(BroadcastAdd<0>(ΣΠ<N_in>(X, W_T), b), Act);
        }

        // batched backward: ONE ΣΠ per gradient term — no per-sample loop.
        //   delta_z = ActivatePrime(delta_A, a)             element-wise, any rank
        //   dW     += (1/B) * ΣΠ<1>(delta_z_BL, a_prev)   batch dim moved to last in delta_z
        //   dB     += (1/B) * ReduceSum<0>(delta_z)
        //   upstream = ΣΠ<N_out>(delta_z, W)               contracts OutDims of delta_z with W
        template<size_t Batch>
        Tensor<Batch, InDims...> BatchedBackward(const Tensor<Batch, OutDims...> &delta_A,
                                                 const Tensor<Batch, OutDims...> &a,
                                                 const Tensor<Batch, InDims...>  &a_prev) {
            const float inv_batch = 1.f / static_cast<float>(Batch);

            const auto delta_z = ActivatePrimeMD(delta_A, a, Act);

            // move batch dim from front to back: Tensor<Batch, OutDims...> → Tensor<OutDims..., Batch>
            constexpr size_t BatchOutRank = 1 + N_out;
            const auto delta_z_BL = PermuteFromHolder<MoveToLastPerm<0, BatchOutRank>>(
                delta_z, std::make_index_sequence<BatchOutRank>{});
            dW += ΣΠ<1>(delta_z_BL, a_prev) * inv_batch;

            dB += ReduceSum<0>(delta_z) * inv_batch;

            return ΣΠ<N_out>(delta_z, W);
        }

        // Adam update. mCorr and vCorr are precomputed by the network.
        // first moment approximates consistency of direction of update
        // second moment approximates inverse of smoothness of local terrain on loss landscape
        void Update(float adamBeta1, float adamBeta2, float lr,
                    float mCorr, float vCorr, float eps = 1e-8f) {
            for (size_t i = 0; i < W_Type::Size; ++i) {
                const float g = dW.flat(i);
                mW.flat(i) = adamBeta1 * mW.flat(i) + (1.f - adamBeta1) * g;
                vW.flat(i) = adamBeta2 * vW.flat(i) + (1.f - adamBeta2) * g * g;
                W.flat(i) -= lr * (mW.flat(i) * mCorr) / (std::sqrt(vW.flat(i) * vCorr) + eps);
            }
            for (size_t i = 0; i < OutputTensor::Size; ++i) {
                const float g = dB.flat(i);
                mb.flat(i) = adamBeta1 * mb.flat(i) + (1.f - adamBeta1) * g;
                vb.flat(i) = adamBeta2 * vb.flat(i) + (1.f - adamBeta2) * g * g;
                b.flat(i) -= lr * (mb.flat(i) * mCorr) / (std::sqrt(vb.flat(i) * vCorr) + eps);
            }
        }

        void Save(std::ofstream &f) const { W.Save(f); b.Save(f); }
        void Load(std::ifstream &f)       { W.Load(f); b.Load(f); }
    };


    // DenseMD recipe: specify the output tensor type; input is inferred at chain time via Resolve.
    //
    // Usage:
    //   NetworkBuilder<
    //       Input<3, 4>,                              // Tensor<3,4> input
    //       DenseMD<Tensor<5>>,                       // → Tensor<5>
    //       DenseMD<Tensor<2, 3>, ReLU>               // → Tensor<2,3>
    //   >::type net;
    template<typename OutT, ActivationFunction Act_ = ActivationFunction::Linear>
    struct DenseMD {
        using OutputTensor = OutT;
        template<typename InputT>
        using Resolve = DenseMDBlock<InputT, OutT, Act_>;
    };

    // Dense<N, Act>: rank-1 shorthand for DenseMD<Tensor<N>, Act>
    template<size_t N, ActivationFunction Act_ = ActivationFunction::Linear>
    using Dense = DenseMD<Tensor<N>, Act_>;
}
