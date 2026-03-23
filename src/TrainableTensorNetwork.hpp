#pragma once
#include <stdexcept>
#include "NetworkUtil.hpp"
#include "TTTN_ML.hpp"

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

        // connectivity check: every Blocks[I]::OutputTensor must equal Blocks[I+1]::InputTensor
        static constexpr bool check_connected() {
            return []<size_t... Is>(std::index_sequence<Is...>) -> bool {
                return (
                    std::is_same_v<
                        typename std::tuple_element_t<Is,     BlockTuple>::OutputTensor,
                        typename std::tuple_element_t<Is + 1, BlockTuple>::InputTensor> &&
                    ...);
            }(std::make_index_sequence<NumBlocks - 1>{});
        }

        static_assert(check_connected(), "Block output/input sizes don't chain");


        // TENSOR TUPLE BUILDER
        // walks Blocks to produce
        //      std::tuple<B0::InputTensor, B0::OutputTensor, B1::OutputTensor, ...>
        // (consecutive OutputTensor/InputTensor are identical by the connectivity check, so no duplicates)

        template<typename... Bs>
        struct TensorTupleBuilder;

        // base case: single block --> (InputTensor, OutputTensor)
        template<typename Last>
        struct TensorTupleBuilder<Last> {
            // @doc: using type
            /** Result of splicing the block lists of `NetA` and `NetB`; a complete network supporting all single-sample and batched interfaces */
            using type = std::tuple<typename Last::InputTensor, typename Last::OutputTensor>;
        };

        // recursive case: emit First::InputTensor, then recurse on <Rest...>
        template<typename First, typename... Rest>
        struct TensorTupleBuilder<First, Rest...> {
            using type = decltype(std::tuple_cat(
                std::declval<std::tuple<typename First::InputTensor>>(),
                std::declval<typename TensorTupleBuilder<Rest...>::type>()
            ));
        };

        // BATCHED TENSOR TUPLE BUILDER
        // like TensorTupleBuilder but prepends Batch to every tensor's dims via PrependBatch
        // produces std::tuple<Tensor<Batch, B0InputDims...>, Tensor<Batch, B0OutputDims...>, ...>

        template<size_t Batch, typename... Bs>
        struct BatchedTensorTupleBuilder;

        template<size_t Batch, typename Last>
        struct BatchedTensorTupleBuilder<Batch, Last> {
            using type = std::tuple<
                typename PrependBatch<Batch, typename Last::InputTensor>::type,
                typename PrependBatch<Batch, typename Last::OutputTensor>::type>;
        };

        template<size_t Batch, typename First, typename... Rest>
        struct BatchedTensorTupleBuilder<Batch, First, Rest...> {
            using type = decltype(std::tuple_cat(
                std::declval<std::tuple<typename PrependBatch<Batch, typename First::InputTensor>::type>>(),
                std::declval<typename BatchedTensorTupleBuilder<Batch, Rest...>::type>()
            ));
        };

        BlockTuple mBlocks;
        int mT = 0;
        float mCorr = 1.f;
        float vCorr = 1.f;

    public:
        // tensor types flow directly from the first and last blocks
        using InputTensor  = typename std::tuple_element_t<0,             BlockTuple>::InputTensor;
        using OutputTensor = typename std::tuple_element_t<NumBlocks - 1, BlockTuple>::OutputTensor;

        // scalar convenience aliases derived from the tensor types
        static constexpr size_t InSize  = InputTensor::Size;
        static constexpr size_t OutSize = OutputTensor::Size;
        static constexpr size_t TotalParamCount = (Blocks::ParamCount + ...);

        // raw tuple types (internal / advanced use)
        using ActivationsTuple = TensorTupleBuilder<Blocks...>::type;
        template<size_t Batch>
        using BatchedActivationsTuple = typename BatchedTensorTupleBuilder<Batch, Blocks...>::type;

        // safe owning wrappers returned by ForwardAll / BatchedForwardAll
        using Activations = ActivationsWrap<ActivationsTuple>;
        template<size_t Batch>
        using BatchedActivations = ActivationsWrap<BatchedActivationsTuple<Batch>>;

        TrainableTensorNetwork() = default;

        // ForwardAll: returns an ActivationsWrap (input + one tensor per block).
        // Bind to a named variable — calling .get<N>() on a temporary is a compile error.
        // @doc: Activations ForwardAll(const InputTensor& x) const
        /** Run forward pass through entire network, returning `Activations` tuple of `Tensor`s from each layer */
        [[nodiscard]] Activations ForwardAll(const InputTensor& x) const {
            ActivationsTuple A;
            std::get<0>(A) = x;
            forward_impl(A);
            return Activations{std::move(A)};
        }

        // @doc: OutputTensor Forward(const InputTensor& x) const
        /** Run forward pass through entire network, returning a `Tensor` of type: `OutputTensor`, the final activation */
        [[nodiscard]] OutputTensor Forward(const InputTensor& x) const {
            auto A = ForwardAll(x);
            return A.template get<NumBlocks>();
        }

        // BackwardAll: takes the wrapper produced by ForwardAll.
        void BackwardAll(const Activations& A, const OutputTensor& grad) {
            backward_impl<NumBlocks>(A.tuple(), grad);
        }

        // apply Adam to every block's stored gradients
        void Update(float lr) {
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (std::get<Is>(mBlocks).Update(ADAM_BETA_1, ADAM_BETA_2, lr, mCorr, vCorr, EPS), ...);
            }(std::make_index_sequence<NumBlocks>{});
        }

        // zero all blocks' gradients
        void ZeroGrad() {
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (std::get<Is>(mBlocks).ZeroGrad(), ...);
            }(std::make_index_sequence<NumBlocks>{});
        }

        // TrainStep: raw gradient version (caller owns gradient computation).
        void TrainStep(const InputTensor& x, const OutputTensor& grad, float lr) {
            const auto A = ForwardAll(x);
            tick_adam();
            ZeroGrad();
            BackwardAll(A, grad);
            Update(lr);
        }

        // Fit: single-sample train step driven by a loss function.
        // Returns the loss value at the start of the step (before the weight update).
        // Usage: float loss = net.Fit<MSE>(x, target, lr);
        template<typename Loss>
        float Fit(const InputTensor& x, const OutputTensor& target, float lr) {
            static_assert(LossFunction<Loss, OutputTensor>,
                "Loss must expose static Loss(pred,target)->float and Grad(pred,target)->OutputTensor");
            const auto A = ForwardAll(x);
            const auto& pred = A.template get<NumBlocks>();
            const float loss_val = Loss::Loss(pred, target);
            const auto grad = Loss::Grad(pred, target);
            tick_adam();
            ZeroGrad();
            BackwardAll(A, grad);
            Update(lr);
            return loss_val;
        }

        // BatchedForwardAll: returns a BatchedActivations wrapper (same safety guarantee).
        // @doc: template<size_t Batch> BatchedActivations<Batch> BatchedForwardAll(const typename PrependBatch<Batch, InputTensor>::type& X) const
        /** Inference a batch and get a `Tensor` of type: `BatchedActivations<Batch>` */
        template<size_t Batch>
        [[nodiscard]] BatchedActivations<Batch> BatchedForwardAll(
                const typename PrependBatch<Batch, InputTensor>::type& X) const {
            BatchedActivationsTuple<Batch> A;
            std::get<0>(A) = X;
            batched_forward_impl<Batch>(A);
            return BatchedActivations<Batch>{std::move(A)};
        }

        // BatchedForward
        // @doc: template <size_t Batch> PrependBatch<Batch, OutputTensor>::type BatchedForward(const typename PrependBatch<Batch, InputTensor>::type& X)
        /** Inference the model with a batch dimension, getting in return a `Tensor` of type: `PrependBatch<Batch, OutputTensor>::type` */
        template <size_t Batch> 
        [[nodiscard]] PrependBatch<Batch, OutputTensor>::type BatchedForward(const typename PrependBatch<Batch, InputTensor>::type& X)
        {
            const auto A = BatchedForwardAll<Batch>(X);
            return A.template get<NumBlocks>();
        }

        // BatchedBackwardAll: takes the wrapper produced by BatchedForwardAll.
        template<size_t Batch>
        void BatchedBackwardAll(const BatchedActivations<Batch>& A,
                                const typename PrependBatch<Batch, OutputTensor>::type& grad) {
            batched_backward_impl<Batch, NumBlocks>(A.tuple(), grad);
        }

        // BatchTrainStep: raw batched gradient version.
        template<size_t Batch>
        void BatchTrainStep(const typename PrependBatch<Batch, InputTensor>::type& X,
                            const typename PrependBatch<Batch, OutputTensor>::type& grad, float lr) {
            const auto A = BatchedForwardAll<Batch>(X);
            tick_adam();
            ZeroGrad();
            BatchedBackwardAll<Batch>(A, grad);
            Update(lr);
        }

        // BatchFit: batched train step driven by a loss function.
        // Gradients and loss are averaged over Batch so lr is batch-size-invariant.
        // Returns the average loss over the batch at the start of the step.
        // Usage: float loss = net.BatchFit<CEL, 32>(X, Y, lr);
        template<typename Loss, size_t Batch>
        float BatchFit(const typename PrependBatch<Batch, InputTensor>::type& X,
                       const typename PrependBatch<Batch, OutputTensor>::type& Y, float lr) {
            static_assert(LossFunction<Loss, OutputTensor>,
                "Loss must expose static Loss(pred,target)->float and Grad(pred,target)->OutputTensor");
            const auto A = BatchedForwardAll<Batch>(X);
            const auto& A_out = A.template get<NumBlocks>();
            typename PrependBatch<Batch, OutputTensor>::type grad;
            float total_loss = 0.f;
            for (size_t b = 0; b < Batch; ++b) {
                OutputTensor pred_b, target_b;
                for (size_t i = 0; i < OutputTensor::Size; ++i) {
                    pred_b.flat(i)   = A_out.flat(b * OutputTensor::Size + i);
                    target_b.flat(i) = Y.flat(b * OutputTensor::Size + i);
                }
                total_loss += Loss::Loss(pred_b, target_b);
                const auto g = Loss::Grad(pred_b, target_b);
                for (size_t i = 0; i < OutputTensor::Size; ++i)
                    grad.flat(b * OutputTensor::Size + i) = g.flat(i) / static_cast<float>(Batch);
            }
            tick_adam();
            ZeroGrad();
            BatchedBackwardAll<Batch>(A, grad);
            Update(lr);
            return total_loss / static_cast<float>(Batch);
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
        // delta is dL/dA[I]: must be a Tensor and must match A[I]'s exact type
        template<size_t I, typename Delta>
            requires IsTensor<Delta> &&
                     std::is_same_v<Delta, std::tuple_element_t<I, ActivationsTuple>>
        void backward_impl(const ActivationsTuple &A, const Delta &delta) {
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
        // delta must be a Tensor and must match A[I]'s exact type
        template<size_t Batch, size_t I, typename Delta>
            requires IsTensor<Delta> &&
                     std::is_same_v<Delta, std::tuple_element_t<I, BatchedActivationsTuple<Batch>>>
        void batched_backward_impl(const BatchedActivationsTuple<Batch> &A, const Delta &delta) {
            // Blocks must implement BatchedBackward<Batch>!!!
            const auto grad = std::get<I - 1>(mBlocks).template BatchedBackward<Batch>(
                delta, std::get<I>(A), std::get<I - 1>(A));
            if constexpr (I > 1) {
                batched_backward_impl<Batch, I - 1>(A, grad);
            }
        }
    };


    // COMBINE NETWORKS
    // Concatenates the block lists of two networks into one TrainableTensorNetwork.
    // Compile-time check: OutTensor of A must equal InputTensor of B.
    //
    // Usage:
    //   using Encoder  = NetworkBuilder<Input<784>, Dense<128, ReLU>, Dense<32>>::type;
    //   using Decoder  = NetworkBuilder<Input<32>,  Dense<128, ReLU>, Dense<784>>::type;
    //   using Autoencoder = CombineNetworks<Encoder, Decoder>::type;
    //
    // All three types can be independently instantiated and trained.
    // CombineNetworks is a type-level operation — no shared weight state.

    template<typename NetA, typename NetB>
    struct CombineNetworks;

    template<ConcreteBlock... BlocksA, ConcreteBlock... BlocksB>
    struct CombineNetworks<TrainableTensorNetwork<BlocksA...>, TrainableTensorNetwork<BlocksB...>> {
        static_assert(
            std::is_same_v<
                typename TrainableTensorNetwork<BlocksA...>::OutputTensor,
                typename TrainableTensorNetwork<BlocksB...>::InputTensor>,
            "CombineNetworks: OutputTensor of first network must equal InputTensor of second");
        using type = TrainableTensorNetwork<BlocksA..., BlocksB...>;
    };


    // typename (not Block) because Input is not a Block
    template<typename In, Block... Blocks>
    struct NetworkBuilder {
        using BlockTuple = typename BuildChain<In, Blocks...>::type;

        template<typename Tuple>
        struct Apply;

        template<typename... Bs>
        struct Apply<std::tuple<Bs...>> {
            using type = TrainableTensorNetwork<Bs...>;
        };

        using type = typename Apply<BlockTuple>::type;
    };
}
