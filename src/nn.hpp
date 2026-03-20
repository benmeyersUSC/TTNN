#pragma once
#include "Tensor.hpp"
#include <array>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>
#include <tuple>
#include <numeric>
#include <ranges>
#include <algorithm>

using namespace TTTN;

static constexpr float EPS = 1e-8f;
static constexpr float ADAM_BETA_1 = 0.9f;
static constexpr float ADAM_BETA_2 = 0.999f;

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


// Block concept
// *any object* which satisfies these criteria may be a Block in a Trainable Tensor Network
template<typename T>
concept Block = requires(T t, const T ct,
                         Tensor<T::InSize> in,
                         Tensor<T::OutSize> out,
                         std::ofstream &of, std::ifstream &inf)
{
    { T::InSize } -> std::convertible_to<size_t>;
    { T::OutSize } -> std::convertible_to<size_t>;
    { ct.Forward(in) } -> std::same_as<Tensor<T::OutSize> >;
    { t.Backward(out, out, in, 0.f, 0.f, 0.f) } -> std::same_as<Tensor<T::InSize> >;
    { ct.Save(of) };
    { t.Load(inf) };
};


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
                        const Tensor<In> &a_prev, float lr, float mCorr, float vCorr) {
        // peel off activation derivative to get dL/dZ
        const auto delta_Z = ActivatePrime(delta_A, a, Act);

        // dW = OUTER(delta_Z, a_prev)
        // delta_Z: Tensor<Out>, a_prev: Tensor<In>....outer prod --> Tensor<Out,In>, same dim as W
        AdamUpdate(Einsum(delta_Z, a_prev), delta_Z, lr, mCorr, vCorr);

        // pass gradient upstream, defining dL/dA_prev:
        //      W: Tensor<Out,In>, delta_Z: Tensor<Out>...contract first axis of each --> Tensor<In>, same dim as a_prev
        // (same thing as DOT(W.Transpose(), delta_Z), but Einsum obviates Transpose!)
        return Einsum<0, 0>(W, delta_Z);
    }

    // Adam update.
    // mCorr and vCorr are precomputed by NN. at the beginning, (low mT), corrections amplify
    // moments from 0-bias, but eventually corrections approach 1
    void AdamUpdate(const Tensor<Out, In> &dW, const Tensor<Out> &db, float lr, float mCorr, float vCorr) {
        // for each Weight and Bias, subtract LR * adjusted_First_Moment / sqrt(adjusted_Second_Moment)
        //      first moment approximates consistency of direction of update
        //      second moment approximates inverse of smoothness of local terrain on loss landscape
        for (size_t i = 0; i < Out * In; ++i) {
            const float g = dW.flat(i);
            mW.flat(i) = ADAM_BETA_1 * mW.flat(i) + (1.f - ADAM_BETA_1) * g;
            vW.flat(i) = ADAM_BETA_2 * vW.flat(i) + (1.f - ADAM_BETA_2) * g * g;
            W.flat(i) -= lr * (mW.flat(i) * mCorr) / (std::sqrt(vW.flat(i) * vCorr) + EPS);
        }
        for (size_t i = 0; i < Out; ++i) {
            const float g = db.flat(i);
            mb.flat(i) = ADAM_BETA_1 * mb.flat(i) + (1.f - ADAM_BETA_1) * g;
            vb.flat(i) = ADAM_BETA_2 * vb.flat(i) + (1.f - ADAM_BETA_2) * g * g;
            b.flat(i) -= lr * (mb.flat(i) * mCorr) / (std::sqrt(vb.flat(i) * vCorr) + EPS);
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


// NEURAL NETWORK CLASS
// templatized by Block-concept-compliant types
//      Blocks[0]   = first block  (network InSize  = its InSize)
//      Blocks[N-1] = last block   (network OutSize = its OutSize)
// network is a std::tuple<Blocks...>; connectivity Blocks[I]::OutSize == Blocks[I+1]::InSize
//      is enforced at compile time

template<Block... Blocks>
class TrainableTensorNetwork {
    static_assert(sizeof...(Blocks) >= 1, "Need at least one block");

    // ACTIVATION TUPLE BUILDER
    // walks Blocks to produce
    //      std::tuple<Tensor<B0::InSize>, Tensor<B0::OutSize>, Tensor<B1::OutSize>, ...>
    // (consecutive OutSize/InSize are equal by the connectivity check, so no duplicates)

    template<typename... Bs>
    struct ActivationsTupleBuilder;

    // base case: single block --> (input, output)
    template<typename Last>
    struct ActivationsTupleBuilder<Last> {
        using type = std::tuple<Tensor<Last::InSize>, Tensor<Last::OutSize> >;
    };

    // recursive case: emit Tensor<First::InSize>, then recurse on <Rest...>
    template<typename First, typename... Rest>
    struct ActivationsTupleBuilder<First, Rest...> {
        using type = decltype(std::tuple_cat(
            std::declval<std::tuple<Tensor<First::InSize> > >(),
            std::declval<typename ActivationsTupleBuilder<Rest...>::type>()
        ));
    };

    static constexpr size_t NumLayers = sizeof...(Blocks);

    using BlockTuple = std::tuple<Blocks...>;

    // connectivity check: every Blocks[I]::OutSize must equal Blocks[I+1]::InSize
    static constexpr bool check_connected() {
        return []<size_t... Is>(std::index_sequence<Is...>) -> bool {
            return ((std::tuple_element_t<Is, BlockTuple>::OutSize ==
                     std::tuple_element_t<Is + 1, BlockTuple>::InSize) && ...);
        }(std::make_index_sequence<NumLayers - 1>{});
    }

    static_assert(check_connected(), "Block output/input sizes don't chain");

    BlockTuple mBlocks;
    int mT = 0;
    float mCorr = 1.0f;
    float vCorr = 1.0f;

public:
    static constexpr size_t InSize = std::tuple_element_t<0, BlockTuple>::InSize;
    static constexpr size_t OutSize = std::tuple_element_t<NumLayers - 1, BlockTuple>::OutSize;

    using InputTensor = Tensor<InSize>;
    using OutputTensor = Tensor<OutSize>;

    // tuple of Tensors representing intermediate network activations
    using ActivationsTuple = typename ActivationsTupleBuilder<Blocks...>::type;

    TrainableTensorNetwork() = default;

    [[nodiscard]] ActivationsTuple ForwardAll(const InputTensor &x) const {
        // declare tuple of activation Tensors
        ActivationsTuple A;
        // assign InputTensor to first activation Tensor
        std::get<0>(A) = x;
        // populate tuple with each block
        forward_all_impl(A);
        // return activation Tensors
        return A;
    }

    [[nodiscard]] OutputTensor Forward(const InputTensor &x) const {
        // call ForwardAll and grab last activation Tensor
        return std::get<NumLayers>(ForwardAll(x));
    }

    // raw full train step: pass in gradient wrt final layer output
    // grad should be dL/da
    void TrainStep(const InputTensor &x, const OutputTensor &grad, float lr) {
        const auto A = ForwardAll(x);
        tick_adam();
        backward_update_impl<NumLayers>(A, grad, lr);
    }

    void Save(const std::string &path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) {
            throw std::runtime_error("Cannot write: " + path);
        }
        // for each block, call Block.Save()
        auto writeBlocks = [&]<size_t... Is>(std::index_sequence<Is...>) {
            (std::get<Is>(mBlocks).Save(f), ...);
        };
        writeBlocks(std::make_index_sequence<NumLayers>{});
    }

    void Load(const std::string &path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            throw std::runtime_error("Cannot read: " + path);
        }
        auto readBlocks = [&]<size_t... Is>(std::index_sequence<Is...>) {
            (std::get<Is>(mBlocks).Load(f), ...);
        };
        readBlocks(std::make_index_sequence<NumLayers>{});
    }

private:
    // increment Adam ticker and update moment corrections
    void tick_adam() {
        ++mT;
        mCorr = 1.0f / (1.0f - std::pow(ADAM_BETA_1, static_cast<float>(mT)));
        vCorr = 1.0f / (1.0f - std::pow(ADAM_BETA_2, static_cast<float>(mT)));
    }

    // internal recursive implementation of forward pass
    // uses Block.Forward to populate ActivationsTuple with activation Tensors
    template<size_t I = 0>
    void forward_all_impl(ActivationsTuple &A) const {
        if constexpr (I < NumLayers) {
            // next activation Tensor = Block[I].Forward(prev activation Tensor)
            // recall there are NumLayers+1 activation Tensors but NumLayers actual blocks
            std::get<I + 1>(A) = std::get<I>(mBlocks).Forward(std::get<I>(A));
            // recurse forward
            forward_all_impl<I + 1>(A);
        }
        // base case is just termination because we have no return
    }

    // recursively backpropagate gradient and return InputTensor-sized/typed gradient
    // calls Block.Backward --> which peels off ActivatePrime then calls AdamUpdate!

    // first template param is actually used, second is just for delta Tensor<_> that we take in...it's effectively free
    // 'I' starts as NumLayers, so it starts by peeling off last block (that's how backprop works)
    // delta is dL/dA[I]: gradient wrt block I-1's output activation
    template<size_t I, size_t DeltaSize>
    InputTensor backward_update_impl(const ActivationsTuple &A, const Tensor<DeltaSize> &delta, float lr) {
        // block I-1 outputs A[I] and takes input A[I-1]
        // Backward peels off ActivatePrime, does AdamUpdate, returns dL/dA[I-1]
        const auto grad = std::get<I - 1>(mBlocks).
                Backward(delta, std::get<I>(A), std::get<I - 1>(A), lr, mCorr, vCorr);
        if constexpr (I > 1) {
            return backward_update_impl<I - 1>(A, grad, lr);
        }
        // because we have a if-constexpr (compile time if), we must pair it with an else.
        // even when I > 1, this code (if not else-wrapped) would run, causing type errors!
        else {
            // base case: gradient is already wrt Input
            return grad;
        }
    }
};
