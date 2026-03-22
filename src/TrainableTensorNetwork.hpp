#pragma once
#include <stdexcept>
#include "NetworkUtil.hpp"

namespace TTTN {
    // TRAINABLE TENSOR NETWORK
    // templatized by Block-concept-compliant types
    //      Blocks[0]   = first block  (network InSize  = its InSize)
    //      Blocks[N-1] = last block   (network OutSize = its OutSize)
    // network is a std::tuple<Blocks...>; connectivity Blocks[I]::OutSize == Blocks[I+1]::InSize
    //      is enforced at compile time

    template<ConcreteBlock... Blocks>
    class TrainableTensorNetwork {
        static_assert(sizeof...(Blocks) >= 1, "Need at least one block");

        static constexpr size_t NumBlocks = sizeof...(Blocks);
        using BlockTuple = std::tuple<Blocks...>;

        // connectivity check: every Blocks[I]::OutSize must equal Blocks[I+1]::InSize
        static constexpr bool check_connected() {
            return []<size_t... Is>(std::index_sequence<Is...>) -> bool {
                return (
                    (std::tuple_element_t<Is, BlockTuple>::OutSize ==
                     std::tuple_element_t<Is + 1, BlockTuple>::InSize) &&
                    ...);
            }(std::make_index_sequence<NumBlocks - 1>{});
        }

        static_assert(check_connected(), "Block output/input sizes don't chain");


        // TENSOR TUPLE BUILDER
        // walks Blocks to produce
        //      std::tuple<Tensor<B0::InSize>, Tensor<B0::OutSize>, Tensor<B1::OutSize>, ...>
        // (consecutive OutSize/InSize are equal by the connectivity check, so no duplicates)

        template<typename... Bs>
        struct TensorTupleBuilder;

        // base case: single block --> (input, output)
        template<typename Last>
        struct TensorTupleBuilder<Last> {
            using type = std::tuple<Tensor<Last::InSize>, Tensor<Last::OutSize> >;
        };

        // recursive case: emit Tensor<First::InSize>, then recurse on <Rest...>
        template<typename First, typename... Rest>
        struct TensorTupleBuilder<First, Rest...> {
            using type = decltype(std::tuple_cat(
                std::declval<std::tuple<Tensor<First::InSize> > >(),
                std::declval<typename TensorTupleBuilder<Rest...>::type>()
            ));
        };

        // BATCHED TENSOR TUPLE BUILDER
        // like TensorTupleBuilder but prepends Batch to every tensor dimension
        // produces std::tuple<Tensor<Batch, B0::InSize>, Tensor<Batch, B0::OutSize>, Tensor<Batch, B1::OutSize>, ...>

        template<size_t Batch, typename... Bs>
        struct BatchedTensorTupleBuilder;

        template<size_t Batch, typename Last>
        struct BatchedTensorTupleBuilder<Batch, Last> {
            using type = std::tuple<Tensor<Batch, Last::InSize>, Tensor<Batch, Last::OutSize> >;
        };

        template<size_t Batch, typename First, typename... Rest>
        struct BatchedTensorTupleBuilder<Batch, First, Rest...> {
            using type = decltype(std::tuple_cat(
                std::declval<std::tuple<Tensor<Batch, First::InSize> > >(),
                std::declval<typename BatchedTensorTupleBuilder<Batch, Rest...>::type>()
            ));
        };

        BlockTuple mBlocks;
        int mT = 0;
        float mCorr = 1.f;
        float vCorr = 1.f;

    public:
        static constexpr size_t InSize = std::tuple_element_t<0, BlockTuple>::InSize;
        static constexpr size_t OutSize = std::tuple_element_t<NumBlocks - 1, BlockTuple>::OutSize;

        using InputTensor = Tensor<InSize>;
        using OutputTensor = Tensor<OutSize>;

        // tuple of Tensors representing every activation: input + one per block output
        using ActivationsTuple = TensorTupleBuilder<Blocks...>::type;

        TrainableTensorNetwork() = default;

        [[nodiscard]] ActivationsTuple ForwardAll(const InputTensor &x) const {
            // declare tuple of activation Tensors
            ActivationsTuple A;
            // assign InputTensor to first activation Tensor
            std::get<0>(A) = x;
            // populate tuple with each block
            forward_impl(A);
            // return activation Tensors
            return A;
        }

        [[nodiscard]] OutputTensor Forward(const InputTensor &x) const {
            // call ForwardAll and grab last activation Tensor
            return std::get<NumBlocks>(ForwardAll(x));
        }

        // grad is dL/dA for the final block's output
        // each block stores its own dW/db; call Update() after to apply them
        void BackwardAll(const ActivationsTuple &A, const OutputTensor &grad) {
            backward_impl<NumBlocks>(A, grad);
        }

        // apply Adam to every block's stored gradients
        void Update(float lr) {
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (std::get<Is>(mBlocks).Update(ADAM_BETA_1, ADAM_BETA_2, lr, mCorr, vCorr, EPS), ...);
            }(std::make_index_sequence<NumBlocks>{});
        }

        // zero all Blocks' params (however they're defined)
        void ZeroGrad() {
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (std::get<Is>(mBlocks).ZeroGrad(), ...);
            }(std::make_index_sequence<NumBlocks>{});
        }

        // full train step: client provides x and dL/dA of final output
        // FIX THIS so they only have to give in X and Y...need to figure out way for dL/dA of final output to be done internally
        void TrainStep(const InputTensor &x, const OutputTensor &grad, float lr) {
            const auto A = ForwardAll(x);
            tick_adam();
            ZeroGrad();
            BackwardAll(A, grad);
            Update(lr);
        }

        // batched train step: X is Tensor<Batch, InSize>, grad is Tensor<Batch, OutSize>
        // uses einsum over batch dim; same weight matrices, no looping!
        template<size_t Batch>
        void BatchTrainStep(const Tensor<Batch, InSize> &X, const Tensor<Batch, OutSize> &grad,float lr)
        {
            const auto A = BatchedForwardAll<Batch>(X);
            tick_adam();
            ZeroGrad();
            // grad is accumulated fully...Blocks should scale lr or grad down by Batch internally!
            BatchedBackwardAll<Batch>(A, grad);
            Update(lr);
        }

        // build activations tuple type returned by BatchForwardAll
        template<size_t Batch>
        using BatchedActivationsTuple = typename BatchedTensorTupleBuilder<Batch, Blocks...>::type;

        template<size_t Batch>
        [[nodiscard]] BatchedActivationsTuple<Batch> BatchedForwardAll(const Tensor<Batch, InSize> &X) const 
        {
            BatchedActivationsTuple<Batch> A;
            std::get<0>(A) = X;
            batched_forward_impl<Batch>(A);
            return A;
        }

        template<size_t Batch>
        void BatchedBackwardAll(const BatchedActivationsTuple<Batch> &A, const Tensor<Batch, OutSize> &grad) 
        {
            batched_backward_impl<Batch, NumBlocks>(A, grad);
        }

        void Save(const std::string &path) const {
            std::ofstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot write: " + path);
            // for each block, call Block.Save()
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (std::get<Is>(mBlocks).Save(f), ...);
            }(std::make_index_sequence<NumBlocks>{});
        }

        void Load(const std::string &path) {
            std::ifstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot read: " + path);
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (std::get<Is>(mBlocks).Load(f), ...);
            }(std::make_index_sequence<NumBlocks>{});
        }

    private:
        // increment Adam ticker and update moment corrections
        void tick_adam() {
            ++mT;
            mCorr = 1.f / (1.f - std::pow(ADAM_BETA_1, static_cast<float>(mT)));
            vCorr = 1.f / (1.f - std::pow(ADAM_BETA_2, static_cast<float>(mT)));
        }

        // internal recursive implementation of forward pass
        // uses Block.Forward to populate ActivationsTuple with activation Tensors
        template<size_t I = 0>
        void forward_impl(ActivationsTuple &A) const {
            if constexpr (I < NumBlocks) {
                // next activation Tensor = Block[I].Forward(prev activation Tensor)
                // recall there are NumBlocks+1 activation Tensors but NumBlocks actual blocks
                std::get<I + 1>(A) = std::get<I>(mBlocks).Forward(std::get<I>(A));
                // recurse forward
                forward_impl<I + 1>(A);
            }
            // base case is just termination because we have no return
        }

        // recursively backpropagate gradient; each block stores its own dW/db
        // calls Block.Backward --> which peels off ActivatePrime and stores dW/db

        // 'I' starts as NumBlocks, so it starts by peeling off last block (that's how backprop works)
        // delta is dL/dA[I]: gradient wrt block I-1's output activation
        template<size_t I, size_t DeltaSize>
        void backward_impl(const ActivationsTuple &A, const Tensor<DeltaSize> &delta) {
            // block I-1 outputs A[I] and takes input A[I-1]
            // Backward peels off ActivatePrime, stores dW/db, returns dL/dA[I-1]
            const auto grad = std::get<I - 1>(mBlocks).Backward(delta, std::get<I>(A), std::get<I - 1>(A));
            if constexpr (I > 1) {
                backward_impl<I - 1>(A, grad);
            }
            // because we have an if-constexpr (compile time if), we must pair it with an else.
            // even when I > 1, this code (if not else-wrapped) would run, causing type errors!
        }

        // mirrors forward_impl but calls BatchedForward on each block
        template<size_t Batch, size_t I = 0>
        void batched_forward_impl(BatchedActivationsTuple<Batch> &A) const {
            if constexpr (I < NumBlocks) {
                // Blocks must implement BatchedForward<Batch>!!!
                std::get<I + 1>(A) = std::get<I>(mBlocks).template BatchedForward<Batch>(std::get<I>(A));
                batched_forward_impl<Batch, I + 1>(A);
            }
        }

        // batched backward impl: mirrors backward_impl but calls BatchedBackward on each block
        template<size_t Batch, size_t I, size_t... DeltaDims>
        void batched_backward_impl(const BatchedActivationsTuple<Batch> &A, const Tensor<DeltaDims...> &delta) {
            // Blocks must implement BatchedBackward<Batch>!!!
            const auto grad = std::get<I - 1>(mBlocks).template BatchedBackward<Batch>(
                delta, std::get<I>(A), std::get<I - 1>(A));
            if constexpr (I > 1) {
                batched_backward_impl<Batch, I - 1>(A, grad);
            }
        }
    };
}
