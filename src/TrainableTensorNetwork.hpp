#pragma once
#include <stdexcept>
#include <random>
#include "TTTN_ML.hpp"
#include "NetworkUtil.hpp"

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

        // @doc: static constexpr size_t TotalParamCount
        /** Sum of parameter counts of all elements of `Blocks...` */
        static constexpr size_t TotalParamCount =
                (TupleParamCount<decltype(std::declval<Blocks &>().all_params())> + ...);

        // @doc: template<size_t I> const auto &TrainableTensorNetwork::block() const
        /** Get a `const &` to the `I`-th `ConcreteBlock` in `TrainableTensorNetwork::mBlocks` */
        template<size_t I>
        const auto &block() const { return std::get<I>(mBlocks); }

        // @doc: using TrainableTensorNetwork::ActivationsTuple
        /** Type alias around `TensorTupleBuilder<Blocks...>::type` */
        using ActivationsTuple = TensorTupleBuilder<Blocks...>::type;

        // @doc: template<size_t Batch> using TrainableTensorNetwork::BatchedActivationsTuple
        /** Type alias around `BatchedTensorTupleBuilder<Blocks...>::type` */
        template<size_t Batch>
        using BatchedActivationsTuple = BatchedTensorTupleBuilder<Batch, Blocks...>::type;

        // @doc: using TrainableTensorNetwork::Activations
        /** Access-safe `ActivationsWrap` wrapper around `ActivationsTuple` */
        using Activations = ActivationsWrap<ActivationsTuple>;

        // @doc: template<size_t Batch> using TrainableTensorNetwork::BatchedActivations
        /** Access-safe `ActivationsWrap` wrapper around `BatchedActivationsTuple` */
        template<size_t Batch>
        using BatchedActivations = ActivationsWrap<BatchedActivationsTuple<Batch> >;

        // @doc: TrainableTensorNetwork::TrainableTensorNetwork
        /** Default constructor `= default` */
        TrainableTensorNetwork() = default;


        // @doc: [[nodiscard]] Activations TrainableTensorNetwork::ForwardAll(const InputTensor &x) const
        /**
         * Forward pass of the network, returning full `ActivationsWrap Activations` object
         * Calls `TrainableTensorNetwork::forward_impl` on `InputTensor &x`
         */
        [[nodiscard]] Activations ForwardAll(const InputTensor &x) const {
            ActivationsTuple A;
            std::get<0>(A) = x;
            forward_impl(A);
            return Activations{std::move(A)};
        }

        // @doc: [[nodiscard]] OutputTensor TrainableTensorNetwork::Forward(const InputTensor &x) const
        /** Forward pass of the network, returning `OutputTensor` extracted from the back of result of `TrainableTensorNetwork::ForwardAll` */
        [[nodiscard]] OutputTensor Forward(const InputTensor &x) const {
            auto A = ForwardAll(x);
            return A.template get<NumBlocks>();
        }

        // @doc: void TrainableTensorNetwork::BackwardAll(const Activations &A, const OutputTensor &grad)
        /**
         * Calls `TrainableTensorNetwork::backward_impl`, which backpropagates gradient via `ConcreteBlock::Backward` calls
         * Gradients are assumed stored and managed by `ConcreteBlock`s with `Param` members
         */
        void BackwardAll(const Activations &A, const OutputTensor &grad) {
            backward_impl<NumBlocks>(A.tuple(), grad);
        }


        // @doc: void TrainableTensorNetwork::Update(float lr)
        /** Calls `mAdam_.step()`, calls `UpdateAll` on each `ConcreteBlock`'s `all_params()`, passing `mAdam_` and `lr` */
        void Update(float lr) {
            mAdam_.step();
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (UpdateAll(std::get<Is>(mBlocks).all_params(), mAdam_, lr), ...);
            }(std::make_index_sequence<NumBlocks>{});
        }


        // @doc: void TrainableTensorNetwork::ZeroGrad()
        /** Calls `ZeroAllGrads` on each `ConcreteBlock`'s `all_params()` */
        void ZeroGrad() {
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (ZeroAllGrads(std::get<Is>(mBlocks).all_params()), ...);
            }(std::make_index_sequence<NumBlocks>{});
        }


        // @doc: [[nodiscard]] SnapshotMap TrainableTensorNetwork::Snap() const
        /** Create and fill `SnapshotMap` for each block, calling `peek()` for any `PeekableBlock`s */
        [[nodiscard]] SnapshotMap Snap() const {
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

        // @doc: void TrainableTensorNetwork::TrainStep(const InputTensor &x, const OutputTensor &grad, const float lr)
        /**
         * Inference -> ZeroGrad -> BackwardAll -> Update
         * Assumes `grad` is `dLoss/dOutputTensor`
         */
        void TrainStep(const InputTensor &x, const OutputTensor &grad, const float lr) {
            const auto A = ForwardAll(x);
            ZeroGrad();
            BackwardAll(A, grad);
            Update(lr);
        }


        // @doc: template<typename Loss> float TrainableTensorNetwork::Fit(const InputTensor &x, const OutputTensor &target, const float lr)
        /** Parameterized by `Loss` (satisfying `LossFunction<Loss, OutputTensor>`), runs Inference, calculates loss, then backpropagates and updates like `TrainStep` */
        template<typename Loss>
        float Fit(const InputTensor &x, const OutputTensor &target, const float lr) {
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

        // @doc: template<size_t Batch> [[nodiscard]] BatchedActivations<Batch> TrainableTensorNetwork::BatchedForwardAll(const PrependBatch<Batch, InputTensor>::type &X) const
        /**
         * Batched forward pass of the network, returning full `ActivationsWrap BatchedActivations` object
         * Calls `TrainableTensorNetwork::forward_impl` on `const PrependBatch<Batch, InputTensor>::type &X`
         */
        template<size_t Batch>
        [[nodiscard]] BatchedActivations<Batch>
        BatchedForwardAll(const PrependBatch<Batch, InputTensor>::type &X) const {
            BatchedActivationsTuple<Batch> A;
            std::get<0>(A) = X;
            batched_forward_impl<Batch>(A);
            return BatchedActivations<Batch>{std::move(A)};
        }

        // @doc: template<size_t Batch> [[nodiscard]] PrependBatch<Batch, OutputTensor>::type TrainableTensorNetwork::BatchedForward(const PrependBatch<Batch, InputTensor>::type &X)
        /** Batched forward pass of the network, returning `PrependBatch<Batch, OutputTensor>` extracted from the back of result of `TrainableTensorNetwork::BatchedForwardAll` */
        template<size_t Batch>
        [[nodiscard]] PrependBatch<Batch, OutputTensor>::type BatchedForward(
            const PrependBatch<Batch, InputTensor>::type &X) {
            const auto A = BatchedForwardAll<Batch>(X);
            return A.template get<NumBlocks>();
        }

        // @doc: template<size_t Batch> void TrainableTensorNetwork::BatchedBackwardAll(const BatchedActivations<Batch> &A, const PrependBatch<Batch, OutputTensor>::type &grad)
        /**
         * `BatchInnerContract` form is part conventional and part performance-informed:
         * `Batch` being left-aligned adopts common convention for `Tensor` shapes in ML
         * `Inner` being right-aligned departs from the original conventional configuration in which `Inner` means `A`'s rightmost and `B`'s leftmost axes. The reason for right-alignment of contracted axes is that `Tensor`s in `TTTN` are backed by ***row-major*** `float` arrays. This means that in the backing array, the only sets of values which are stored contiguously are those in the right-most axes. To maximize vectorization optimizations for `Reduce ∘ zipWith(Map)` operations, we want the loops over contracted indices to be traversing contiguous memory. Detailed comments on this subject are resident in the code.
         */
        template<size_t Batch>
        void BatchedBackwardAll(const BatchedActivations<Batch> &A,
                                const PrependBatch<Batch, OutputTensor>::type &grad) {
            batched_backward_impl<Batch, NumBlocks>(A.tuple(), grad);
        }

        // @doc: template<size_t Batch> void TrainableTensorNetwork::BatchTrainStep(const PrependBatch<Batch, InputTensor>::type &X, const PrependBatch<Batch, OutputTensor>::type &grad, const float lr)
        /**
         * Batched Inference -> ZeroGrad -> BatchedBackwardAll -> Update
         * Assumes `grad` is batched `dLoss/dOutputTensor`
         */
        template<size_t Batch>
        void BatchTrainStep(const PrependBatch<Batch, InputTensor>::type &X,
                            const PrependBatch<Batch, OutputTensor>::type &grad, const float lr) {
            const auto A = BatchedForwardAll<Batch>(X);

            ZeroGrad();
            BatchedBackwardAll<Batch>(A, grad);
            Update(lr);
        }

        // @doc: template<typename Loss, size_t Batch> float TrainableTensorNetwork::BatchFit(const PrependBatch<Batch, InputTensor>::type &X, const PrependBatch<Batch, OutputTensor>::type &Y, const float lr)
        /** Parameterized by `Loss` (satisfying `LossFunction<Loss, OutputTensor>`), runs Batched Inference, calculates loss, then batch backpropagates and updates like `TrainStep` */
        template<typename Loss, size_t Batch>
        float BatchFit(const PrependBatch<Batch, InputTensor>::type &X,
                       const PrependBatch<Batch, OutputTensor>::type &Y, const float lr) {
            static_assert(LossFunction<Loss, OutputTensor>,
                          "Loss must expose static Loss(pred,target)->Tensor<> and Grad(pred,target)->OutputTensor");
            const auto A = BatchedForwardAll<Batch>(X);
            const auto &A_out = A.template get<NumBlocks>();

            // get loss value for returning
            float loss_val = Reduce<0, Add>(
                BatchZip(
                    A_out, Y, [](const auto &p, const auto &t) {
                        return Loss::Loss(p, t);
                    }
                )
            );
            // grad becomes Tensor<Batch, OutputTensor>
            auto grad = BatchZip(A_out, Y, [](const auto &p, const auto &t) {
                return Loss::Grad(p, t);
            });


            // scale by Batch
            const auto inv = 1.f / static_cast<float>(Batch);
            loss_val *= inv;
            grad *= inv;

            // same as train step
            ZeroGrad();
            BatchedBackwardAll<Batch>(A, grad);
            Update(lr);
            return loss_val;
        }


        // @doc: void TrainableTensorNetwork::Save(const std::string &path) const
        /** Calls `SaveAll` on each `ConcreteBlock::all_params()`, which calls `Tensor` binary serialization function */
        void Save(const std::string &path) const {
            std::ofstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot write: " + path);
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (SaveAll(std::get<Is>(mBlocks).all_params(), f), ...);
            }(std::make_index_sequence<NumBlocks>{});
        }

        // @doc: void TrainableTensorNetwork::Load(const std::string &path)
        /** Calls `LoadAll` on each `ConcreteBlock::all_params()`, which calls `Tensor` binary serialization function */
        void Load(const std::string &path) {
            std::ifstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot read: " + path);
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (LoadAll(std::get<Is>(mBlocks).all_params(), f), ...);
            }(std::make_index_sequence<NumBlocks>{});
        }


        // @doc: template<typename Loss, size_t Batch, size_t N, size_t... InDims, size_t... OutDims> float TrainableTensorNetwork::RunEpoch(const Tensor<N, InDims...> &X_data, const Tensor<N, OutDims...> &Y_data, std::mt19937 &rng, const float lr)
        /**
         * Run one full epoch: `Steps = N / Batch` rounds of `BatchFit`, returning average loss per step
         * `X_data` and `Y_data` must already be in network shape (`Tensor<InDims...> == InputTensor`, `Tensor<OutDims...> == OutputTensor`); enforced by `static_assert`
         * Samples `Batch` indices per step from `[0, N)` using `rng`, applied to both `X_data` and `Y_data` in the same loop to keep them in sync
         */
        template<typename Loss, size_t Batch, size_t N, size_t... InDims, size_t... OutDims>
        float RunEpoch(const Tensor<N, InDims...> &X_data,
                       const Tensor<N, OutDims...> &Y_data,
                       std::mt19937 &rng, const float lr) {
            static_assert(LossFunction<Loss, OutputTensor>,
                          "Loss must expose static Loss(pred,target)->Tensor<> and Grad(pred,target)->OutputTensor");
            static_assert(std::is_same_v<Tensor<InDims...>, InputTensor>,
                          "X_data sample shape must match network InputTensor");
            static_assert(std::is_same_v<Tensor<OutDims...>, OutputTensor>,
                          "Y_data sample shape must match network OutputTensor");
            static constexpr size_t Steps = N / Batch;
            // to sample random rows for batches
            std::uniform_int_distribution<size_t> dist{0, N - 1};
            float total_loss = 0.f;
            for (size_t s = 0; s < Steps; ++s) {
                // fill a new batch tensor
                typename PrependBatch<Batch, InputTensor>::type X;
                typename PrependBatch<Batch, OutputTensor>::type Y;
                for (size_t b = 0; b < Batch; ++b) {
                    // get random (row) index
                    const size_t idx = dist(rng);
                    // grab subtensors at that index into batched tensor
                    TensorSet<0>(X, b, TensorGet<0>(X_data, idx));
                    TensorSet<0>(Y, b, TensorGet<0>(Y_data, idx));
                }
                total_loss += BatchFit<Loss, Batch>(X, Y, lr);
            }
            return total_loss / static_cast<float>(Steps);
        }

    private:
        // @doc: template<size_t I = 0> void TrainableTensorNetwork::forward_impl(ActivationsTuple &A) const
        /**
         * Private implementation of forward pass of network, fills `ActivationsTuple &A`
         * Recursively iterates through `TrainableTensorNetwork::mBlocks`, assigning results of mandated `ConcreteBlock::Forward` to corresponding entries of `ActivationsTuple &A`
         */
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


        // @doc: template<size_t I, typename Delta> requires IsTensor<Delta> && std::is_same_v<Delta, std::tuple_element_t<I, ActivationsTuple> > void TrainableTensorNetwork::backward_impl(const ActivationsTuple &A, const Delta &delta)
        /**
         * Starts with `Delta` `Tensor`, the derivative of the `Loss` with respect to the `OutputTensor`
         * `Delta` satisfies:
         * `IsTensor<Delta> && std::is_same_v<Delta, std::tuple_element_t<I, ActivationsTuple> >`
         * `I` starts at `NumBlocks` and recurses down until `I == 1`
         * At each `I`, the `I - 1`-th `ConcreteBlock`'s gradient takes into account that this block outputs the `I`-th activation in an `ActivationsTuple`, having taken the `I - 1`-th activation from `ActivationsTuple`
         */
        template<size_t I, typename Delta> requires
            IsTensor<Delta> && std::is_same_v<Delta, std::tuple_element_t<I, ActivationsTuple> >
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

        // @doc: template<size_t Batch, size_t I = 0> void TrainableTensorNetwork::batched_forward_impl(BatchedActivationsTuple<Batch> &A) const
        /**
         * Private implementation of batched forward pass of network, fills `BatchedActivationsTuple &A`
         * Recursively iterates through `TrainableTensorNetwork::mBlocks`, assigning results of mandated `ConcreteBlock::BatchedForward` to corresponding entries of `BatchedActivationsTuple &A`
         */
        template<size_t Batch, size_t I = 0>
        void batched_forward_impl(BatchedActivationsTuple<Batch> &A) const {
            if constexpr (I < NumBlocks) {
                // Blocks must implement BatchedForward<Batch>!!!
                std::get<I + 1>(A) = std::get<I>(mBlocks).template BatchedForward<Batch>(std::get<I>(A));
                batched_forward_impl<Batch, I + 1>(A);
            }
        }

        // @doc: template<size_t Batch, size_t I, typename Delta> requires IsTensor<Delta> && std::is_same_v<Delta, std::tuple_element_t<I, BatchedActivationsTuple<Batch> > > void TrainableTensorNetwork::batched_backward_impl(const BatchedActivationsTuple<Batch> &A, const Delta &delta)
        /**
         * Starts with `Delta` batched-prepended `Tensor`, the batched derivatives of the `Loss` with respect to the batch-prepended `OutputTensor`
         * Same logic as `TrainableTensorNetwork::backward_impl` but calling `ConcreteBlock::BatchedBackward` instead
         */
        template<size_t Batch, size_t I, typename Delta> requires
            IsTensor<Delta> && std::is_same_v<Delta, std::tuple_element_t<I, BatchedActivationsTuple<Batch> > >
        void batched_backward_impl(const BatchedActivationsTuple<Batch> &A, const Delta &delta) {
            // Blocks must implement BatchedBackward<Batch>!!!
            const auto grad = std::get<I - 1>(mBlocks).template BatchedBackward<Batch>(
                delta, std::get<I>(A), std::get<I - 1>(A));
            if constexpr (I > 1) {
                batched_backward_impl<Batch, I - 1>(A, grad);
            }
        }
    };
}
