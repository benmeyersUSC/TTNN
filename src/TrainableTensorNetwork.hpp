#pragma once
#include <stdexcept>
#include "ChainBlock.hpp"
#include "TTTN_ML.hpp"

namespace TTTN {
    // @doc: template<ConcreteBlock... Blocks> class TrainableTensorNetwork
    /**
     * Capstone object of the library
     * Wrap `std::tuple` of shape-compliant `ConcreteBlock`s in a network API that enables and/or enforces:
     * `Forward` and `BatchedForward`
     * `Backward` and `BatchedBackward`
     * `Update`, `ZeroGrad`, `TrainStep` and `BatchedTrainStep`, `Fit` and `BatchFit`
     * `snap` (snapshot of activations)
     * `Save` and `Load` serialization
     */
    template<ConcreteBlock... Blocks>
    class TrainableTensorNetwork {
        static_assert(sizeof...(Blocks) >= 1, "Need at least one block");

        static constexpr size_t NumBlocks = sizeof...(Blocks);

        // @doc: TrainableTensorNetwork::BlockTuple
        /**
         * `using BlockTuple = std::tuple<Blocks...>`
         * NOTE: not a `std::tuple` of `Blocks...` *objects* but *types*
         */
        using BlockTuple = std::tuple<Blocks...>;

        // @doc: static constexpr bool TrainableTensorNetwork::check_connected()
        /** Immediate `static_assert` function to ensure that `ConcreteBlock... Blocks` have compliant shapes: - `std::is_same_v<typename std::tuple_element_t<Is, BlockTuple>::OutputTensor, typename std::tuple_element_t<Is + 1, BlockTuple>::InputTensor> && ...)` */
        static constexpr bool check_connected() {
            return []<size_t... Is>(std::index_sequence<Is...>) -> bool {
                return (
                    std::is_same_v<
                        typename std::tuple_element_t<Is, BlockTuple>::OutputTensor,
                        typename std::tuple_element_t<Is + 1, BlockTuple>::InputTensor> &&
                    ...);
            }(std::make_index_sequence<NumBlocks - 1>{});
        }

        static_assert(check_connected(), "Block output/input sizes don't chain");


        // @doc: TrainableTensorNetwork::mBlocks
        /** Default-constructed `BlockTuple` type containing actual `ConcreteBlock` values */
        BlockTuple mBlocks;
        AdamState mAdam_{};

    public:
        // @doc: TrainableTensorNetwork::InputTensor
        /** Extract `InputTensor` type from first element of `BlockTuple` */
        using InputTensor = std::tuple_element_t<0, BlockTuple>::InputTensor;

        // @doc: TrainableTensorNetwork::OutputTensor
        /** Extract `OutputTensor` type from last element of `BlockTuple` */
        using OutputTensor = std::tuple_element_t<NumBlocks - 1, BlockTuple>::OutputTensor;

        // @doc: TrainableTensorNetwork::InSize
        /** Convenience member for total size of `InputTensor` type */
        static constexpr size_t InSize = InputTensor::Size;
        // @doc: TrainableTensorNetwork::OutSize
        /** Convenience member for total size of `OutputTensor` type */
        static constexpr size_t OutSize = OutputTensor::Size;

        //
        static constexpr size_t TotalParamCount =
                (TupleParamCount<decltype(std::declval<Blocks &>().all_params())> + ...);

        template<size_t I>
        const auto &block() const { return std::get<I>(mBlocks); }

        // raw tuple types (internal / advanced use)
        using ActivationsTuple = TensorTupleBuilder<Blocks...>::type;
        template<size_t Batch>
        using BatchedActivationsTuple = BatchedTensorTupleBuilder<Batch, Blocks...>::type;

        // safe owning wrappers returned by ForwardAll / BatchedForwardAll
        using Activations = ActivationsWrap<ActivationsTuple>;
        template<size_t Batch>
        using BatchedActivations = ActivationsWrap<BatchedActivationsTuple<Batch> >;

        TrainableTensorNetwork() = default;

        // ForwardAll: returns an ActivationsWrap (input + one tensor per block).
        // Bind to a named variable — calling .get<N>() on a temporary is a compile error.
        [[nodiscard]] Activations ForwardAll(const InputTensor &x) const {
            ActivationsTuple A;
            std::get<0>(A) = x;
            forward_impl(A);
            return Activations{std::move(A)};
        }

        [[nodiscard]] OutputTensor Forward(const InputTensor &x) const {
            auto A = ForwardAll(x);
            return A.template get<NumBlocks>();
        }

        // BackwardAll: takes the wrapper produced by ForwardAll.
        void BackwardAll(const Activations &A, const OutputTensor &grad) {
            backward_impl<NumBlocks>(A.tuple(), grad);
        }


        void Update(float lr) {
            mAdam_.step();
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (UpdateAll(std::get<Is>(mBlocks).all_params(), mAdam_, lr), ...);
            }(std::make_index_sequence<NumBlocks>{});
        }


        void ZeroGrad() {
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (ZeroAllGrads(std::get<Is>(mBlocks).all_params()), ...);
            }(std::make_index_sequence<NumBlocks>{});
        }


        [[nodiscard]] SnapshotMap snap() const {
            SnapshotMap out;
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                ([&] {
                    if constexpr (const auto &blk = std::get<Is>(mBlocks); PeekableBlock<std::remove_cvref_t<decltype(
                        blk)> >)
                        blk.peek(out, "block_" + std::to_string(Is) + ".");
                }(), ...);
            }(std::make_index_sequence<NumBlocks>{});
            return out;
        }

        // TrainStep: raw gradient version (caller owns gradient computation).
        void TrainStep(const InputTensor &x, const OutputTensor &grad, float lr) {
            const auto A = ForwardAll(x);

            ZeroGrad();
            BackwardAll(A, grad);
            Update(lr);
        }

        // Fit: single-sample train step driven by a loss function.
        // Returns the loss value at the start of the step (before the weight update).
        // Usage: float loss = net.Fit<MSE>(x, target, lr);
        template<typename Loss>
        float Fit(const InputTensor &x, const OutputTensor &target, float lr) {
            static_assert(LossFunction<Loss, OutputTensor>,
                          "Loss must expose static Loss(pred,target)->float and Grad(pred,target)->OutputTensor");
            const auto A = ForwardAll(x);
            const auto &pred = A.template get<NumBlocks>();
            const float loss_val = Loss::Loss(pred, target);
            const auto grad = Loss::Grad(pred, target);

            ZeroGrad();
            BackwardAll(A, grad);
            Update(lr);
            return loss_val;
        }

        // BatchedForwardAll: returns a BatchedActivations wrapper (same safety guarantee).
        template<size_t Batch>
        [[nodiscard]] BatchedActivations<Batch> BatchedForwardAll(
            const PrependBatch<Batch, InputTensor>::type &X) const {
            BatchedActivationsTuple<Batch> A;
            std::get<0>(A) = X;
            batched_forward_impl<Batch>(A);
            return BatchedActivations<Batch>{std::move(A)};
        }

        // BatchedForward
        template<size_t Batch>
        [[nodiscard]] PrependBatch<Batch, OutputTensor>::type BatchedForward(
            const PrependBatch<Batch, InputTensor>::type &X) {
            const auto A = BatchedForwardAll<Batch>(X);
            return A.template get<NumBlocks>();
        }

        // BatchedBackwardAll: takes the wrapper produced by BatchedForwardAll.
        template<size_t Batch>
        void BatchedBackwardAll(const BatchedActivations<Batch> &A,
                                const PrependBatch<Batch, OutputTensor>::type &grad) {
            batched_backward_impl<Batch, NumBlocks>(A.tuple(), grad);
        }

        // BatchTrainStep: raw batched gradient version.
        template<size_t Batch>
        void BatchTrainStep(const PrependBatch<Batch, InputTensor>::type &X,
                            const PrependBatch<Batch, OutputTensor>::type &grad, float lr) {
            const auto A = BatchedForwardAll<Batch>(X);

            ZeroGrad();
            BatchedBackwardAll<Batch>(A, grad);
            Update(lr);
        }

        // BatchFit: batched train step driven by a loss function.
        // Gradients and loss are averaged over Batch so lr is batch-size-invariant.
        // Returns the average loss over the batch at the start of the step.
        // Usage: float loss = net.BatchFit<CEL, 32>(X, Y, lr);
        template<typename Loss, size_t Batch>
        float BatchFit(const PrependBatch<Batch, InputTensor>::type &X,
                       const PrependBatch<Batch, OutputTensor>::type &Y, float lr) {
            static_assert(LossFunction<Loss, OutputTensor>,
                          "Loss must expose static Loss(pred,target)->float and Grad(pred,target)->OutputTensor");
            const auto A = BatchedForwardAll<Batch>(X);
            const auto &A_out = A.template get<NumBlocks>();
            typename PrependBatch<Batch, OutputTensor>::type grad;
            float total_loss = 0.f;
            for (size_t b = 0; b < Batch; ++b) {
                OutputTensor pred_b, target_b;
                for (size_t i = 0; i < OutputTensor::Size; ++i) {
                    pred_b.flat(i) = A_out.flat(b * OutputTensor::Size + i);
                    target_b.flat(i) = Y.flat(b * OutputTensor::Size + i);
                }
                total_loss += Loss::Loss(pred_b, target_b);
                const auto g = Loss::Grad(pred_b, target_b);
                for (size_t i = 0; i < OutputTensor::Size; ++i)
                    grad.flat(b * OutputTensor::Size + i) = g.flat(i) / static_cast<float>(Batch);
            }

            ZeroGrad();
            BatchedBackwardAll<Batch>(A, grad);
            Update(lr);
            return total_loss / static_cast<float>(Batch);
        }

        void Save(const std::string &path) const {
            std::ofstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot write: " + path);
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (SaveAll(std::get<Is>(mBlocks).all_params(), f), ...);
            }(std::make_index_sequence<NumBlocks>{});
        }

        void Load(const std::string &path) {
            std::ifstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot read: " + path);
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (LoadAll(std::get<Is>(mBlocks).all_params(), f), ...);
            }(std::make_index_sequence<NumBlocks>{});
        }

    private:
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
                     std::is_same_v<Delta, std::tuple_element_t<I, ActivationsTuple> >
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
                     std::is_same_v<Delta, std::tuple_element_t<I, BatchedActivationsTuple<Batch> > >
        void batched_backward_impl(const BatchedActivationsTuple<Batch> &A, const Delta &delta) {
            // Blocks must implement BatchedBackward<Batch>!!!
            const auto grad = std::get<I - 1>(mBlocks).template BatchedBackward<Batch>(
                delta, std::get<I>(A), std::get<I - 1>(A));
            if constexpr (I > 1) {
                batched_backward_impl<Batch, I - 1>(A, grad);
            }
        }
    };


    template<typename NetA, typename NetB>
    struct CombineNetworks;

    template<ConcreteBlock... BlocksA, ConcreteBlock... BlocksB>
    struct CombineNetworks<TrainableTensorNetwork<BlocksA...>, TrainableTensorNetwork<BlocksB...> > {
        static_assert(
            std::is_same_v<
                typename TrainableTensorNetwork<BlocksA...>::OutputTensor,
                typename TrainableTensorNetwork<BlocksB...>::InputTensor>,
            "CombineNetworks: OutputTensor of first network must equal InputTensor of second");
        using type = TrainableTensorNetwork<BlocksA..., BlocksB...>;
    };


    template<typename Loss, size_t Batch,
        ConcreteBlock... Blocks, size_t N, size_t... DataDims, typename PrepFn>
    float RunEpoch(TrainableTensorNetwork<Blocks...> &net,
                   const Tensor<N, DataDims...> &dataset,
                   std::mt19937 &rng, float lr, PrepFn prep) {
        static constexpr size_t Steps = N / Batch;
        using Net = TrainableTensorNetwork<Blocks...>;
        using BatchX = PrependBatch<Batch, typename Net::InputTensor>::type;
        using BatchY = PrependBatch<Batch, typename Net::OutputTensor>::type;
        float total = 0.f;
        for (size_t s = 0; s < Steps; ++s) {
            auto batch = RandomBatch<Batch>(dataset, rng);
            BatchX X;
            BatchY Y;
            prep(batch, X, Y);
            total += net.template BatchFit<Loss, Batch>(X, Y, lr);
        }
        return total / static_cast<float>(Steps);
    }


    // NetworkBuilder: typename... (not Block...) so ComposeBlocks can appear in the list.
    // FlattenRecipes expands any ComposeBlocks before BuildChain sees the recipes.
    template<typename In, typename... Recipes>
    struct NetworkBuilder {
        using FlatRecipes = FlattenRecipes<Recipes...>::type;
        using BlockTuple = ApplyBuildChain<In, FlatRecipes>::type;

        template<typename Tuple>
        struct Apply;

        template<typename... Bs>
        struct Apply<std::tuple<Bs...> > {
            using type = TrainableTensorNetwork<Bs...>;
        };

        using type = Apply<BlockTuple>::type;
    };
}
