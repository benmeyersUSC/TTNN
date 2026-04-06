#pragma once
#include <random>
#include "BlockSequence.hpp"
#include "TTTN_ML.hpp"

namespace TTTN {
    // @doc: template<Block... Blocks> class TrainableTensorNetwork::TrainableTensorNetwork
    /**
     * Capstone object of the library; owns a `BlockSequence<Blocks...> mSeq_` and an `AdamState mAdam_`
     * All type aliases (`InputTensor`, `OutputTensor`, `Activations`, etc.) and inference/backward/serialization methods delegate directly to `mSeq_`
     * Exclusively owns the optimizer state and loss-parameterized training entry points
     */
    template<Block... Blocks>
    class TrainableTensorNetwork {
        static_assert(sizeof...(Blocks) >= 1, "Need at least one block");

        // @doc: using TrainableTensorNetwork::Seq
        /** `using Seq = BlockSequence<Blocks...>` - internal shorthand for the inner sequence type */
        using Seq = BlockSequence<Blocks...>;

        // @doc: Seq TrainableTensorNetwork::mSeq_
        /** The inner `BlockSequence<Blocks...>` that owns all blocks and activation caches */
        Seq mSeq_;

        // @doc: AdamState TrainableTensorNetwork::mAdam_
        /** `AdamState` instance; stepped once per `Update` call */
        AdamState mAdam_{};

    public:
        // @doc: using TrainableTensorNetwork::InputTensor
        /** Delegates to `BlockSequence::InputTensor` */
        using InputTensor = Seq::InputTensor;
        // @doc: using TrainableTensorNetwork::OutputTensor
        /** Delegates to `BlockSequence::OutputTensor` */
        using OutputTensor = Seq::OutputTensor;
        // @doc: static constexpr size_t TrainableTensorNetwork::InSize
        /** Delegates to `BlockSequence::InSize` */
        static constexpr size_t InSize = Seq::InSize;
        // @doc: static constexpr size_t TrainableTensorNetwork::OutSize
        /** Delegates to `BlockSequence::OutSize` */
        static constexpr size_t OutSize = Seq::OutSize;
        // @doc: static constexpr size_t TrainableTensorNetwork::NumBlocks
        /** Delegates to `BlockSequence::NumBlocks` */
        static constexpr size_t NumBlocks = Seq::NumBlocks;
        // @doc: static constexpr size_t TrainableTensorNetwork::TotalParamCount
        /** Delegates to `BlockSequence::TotalParamCount` */
        static constexpr size_t TotalParamCount = Seq::TotalParamCount;

        // @doc: using TrainableTensorNetwork::Activations
        /** Delegates to `BlockSequence::Activations` */
        using Activations = Seq::Activations;

        // @doc: template<size_t Batch> using TrainableTensorNetwork::BatchedActivations
        /** Delegates to `BlockSequence::BatchedActivations<Batch>` */
        template<size_t Batch>
        using BatchedActivations = Seq::template BatchedActivations<Batch>;

        // @doc: TrainableTensorNetwork::TrainableTensorNetwork()
        /** Default constructor `= default` */
        TrainableTensorNetwork() = default;

        // @doc: template<size_t I> const auto &TrainableTensorNetwork::block() const
        /** Delegates to `BlockSequence::block<I>()` */
        template<size_t I>
        const auto &block() const { return mSeq_.template block<I>(); }


        // @doc: [[nodiscard]] Activations TrainableTensorNetwork::ForwardAll(const InputTensor &x) const
        /** Delegates to `BlockSequence::ForwardAll` */
        [[nodiscard]] Activations ForwardAll(const InputTensor &x) const { return mSeq_.ForwardAll(x); }

        // @doc: [[nodiscard]] OutputTensor TrainableTensorNetwork::Forward(const InputTensor &x) const
        /** Delegates to `BlockSequence::Forward` */
        [[nodiscard]] OutputTensor Forward(const InputTensor &x) const { return x >> mSeq_; }


        // @doc: template<size_t Batch> [[nodiscard]] BatchedActivations<Batch> TrainableTensorNetwork::BatchedForwardAll(const PrependBatch<Batch, InputTensor>::type &X) const
        /** Delegates to `BlockSequence::BatchedForwardAll<Batch>` */
        template<size_t Batch>
        [[nodiscard]] BatchedActivations<Batch>
        BatchedForwardAll(const PrependBatch<Batch, InputTensor>::type &X) const {
            return mSeq_.template BatchedForwardAll<Batch>(X);
        }

        // @doc: template<size_t Batch> [[nodiscard]] PrependBatch<Batch, OutputTensor>::type TrainableTensorNetwork::BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const
        /** Delegates to `BlockSequence::BatchedForward<Batch>` */
        template<size_t Batch>
        [[nodiscard]] PrependBatch<Batch, OutputTensor>::type BatchedForward(
            const PrependBatch<Batch, InputTensor>::type &X) const {
            return mSeq_.template BatchedForward<Batch>(X);
        }

        // @doc: template<size_t I, typename Delta> void TrainableTensorNetwork::BackwardFrom(const Activations &A, const Delta &grad)
        /** Delegates to `BlockSequence::BackwardFrom<I>` */
        template<size_t I, typename Delta>
        void BackwardFrom(const Activations &A, const Delta &grad) {
            mSeq_.template BackwardFrom<I>(A, grad);
        }

        // @doc: void TrainableTensorNetwork::BackwardAll(const Activations &A, const OutputTensor &grad)
        /** Delegates to `BlockSequence::BackwardAll` */
        void BackwardAll(const Activations &A, const OutputTensor &grad) { mSeq_.BackwardAll(A, grad); }


        // @doc: template<size_t Batch, size_t I, typename Delta> void TrainableTensorNetwork::BatchedBackwardFrom(const BatchedActivations<Batch> &A, const Delta &grad)
        /** Delegates to `BlockSequence::BatchedBackwardFrom<Batch, I>` */
        template<size_t Batch, size_t I, typename Delta>
        void BatchedBackwardFrom(const BatchedActivations<Batch> &A, const Delta &grad) {
            mSeq_.template BatchedBackwardFrom<Batch, I>(A, grad);
        }


        // @doc: template<size_t Batch> void TrainableTensorNetwork::BatchedBackwardAll(const BatchedActivations<Batch> &A, const PrependBatch<Batch, OutputTensor>::type &grad)
        /** Delegates to `BlockSequence::BatchedBackwardAll<Batch>` */
        template<size_t Batch>
        void BatchedBackwardAll(const BatchedActivations<Batch> &A,
                                const PrependBatch<Batch, OutputTensor>::type &grad) {
            mSeq_.template BatchedBackwardAll<Batch>(A, grad);
        }


        // @doc: template<size_t Lo, size_t Hi, typename GradT> auto TrainableTensorNetwork::BackwardRange(const Activations &A, const GradT &grad)
        /**
         * `BatchMinorContract` form is part conventional and part performance-informed:
         * `Batch` being left-aligned adopts common convention for `Tensor` shapes in ML
         * `Minor` (contracted) axes being right-aligned reflects that `Tensor`s in `TTTN` are backed by ***row-major*** `float` arrays. Only the rightmost (minor) axes are stored contiguously in memory. To maximize vectorization optimizations for `Reduce ∘ zipWith(Map)` operations, we want the loops over contracted indices to be traversing contiguous memory. Detailed comments on this subject are resident in the code.
         */
        template<size_t Lo, size_t Hi, typename GradT>
        auto BackwardRange(const Activations &A, const GradT &grad) {
            return mSeq_.template BackwardRange<Lo, Hi>(A, grad);
        }

        // @doc: template<size_t Batch, size_t Lo, size_t Hi, typename GradT> auto TrainableTensorNetwork::BatchedBackwardRange(const BatchedActivations<Batch> &A, const GradT &grad)
        /**
         * `BatchMinorContract` form is part conventional and part performance-informed:
         * `Batch` being left-aligned adopts common convention for `Tensor` shapes in ML
         * `Minor` (contracted) axes being right-aligned reflects that `Tensor`s in `TTTN` are backed by ***row-major*** `float` arrays. Only the rightmost (minor) axes are stored contiguously in memory. To maximize vectorization optimizations for `Reduce ∘ zipWith(Map)` operations, we want the loops over contracted indices to be traversing contiguous memory. Detailed comments on this subject are resident in the code.
         */
        template<size_t Batch, size_t Lo, size_t Hi, typename GradT>
        auto BatchedBackwardRange(const BatchedActivations<Batch> &A, const GradT &grad) {
            return mSeq_.template BatchedBackwardRange<Batch, Lo, Hi>(A, grad);
        }

        // @doc: void TrainableTensorNetwork::ZeroGrad()
        /** Delegates to `BlockSequence::ZeroGrad` */
        void ZeroGrad() { mSeq_.ZeroGrad(); }

        // @doc: void TrainableTensorNetwork::Update(const float lr)
        /**
         * `BatchMinorContract` form is part conventional and part performance-informed:
         * `Batch` being left-aligned adopts common convention for `Tensor` shapes in ML
         * `Minor` (contracted) axes being right-aligned reflects that `Tensor`s in `TTTN` are backed by ***row-major*** `float` arrays. Only the rightmost (minor) axes are stored contiguously in memory. To maximize vectorization optimizations for `Reduce ∘ zipWith(Map)` operations, we want the loops over contracted indices to be traversing contiguous memory. Detailed comments on this subject are resident in the code.
         */
        void Update(const float lr) {
            mAdam_.step();
            UpdateAll(mSeq_.all_params(), mAdam_, lr);
        }


        // @doc: [[nodiscard]] SnapshotMap TrainableTensorNetwork::Snap() const
        /** Delegates to `BlockSequence::Snap` */
        [[nodiscard]] SnapshotMap Snap() const { return mSeq_.Snap(); }

        // @doc: void TrainableTensorNetwork::Save(const std::string &path) const
        /** Delegates to `BlockSequence::Save` */
        void Save(const std::string &path) const { mSeq_.Save(path); }

        // @doc: void TrainableTensorNetwork::Load(const std::string &path)
        /** Delegates to `BlockSequence::Load` */
        void Load(const std::string &path) { mSeq_.Load(path); }


        // @doc: void TrainableTensorNetwork::TrainStep(const InputTensor &x, const OutputTensor &grad, const float lr)
        /**
         * `ForwardAll` -> `ZeroGrad` -> `BackwardAll` -> `Update`
         * `grad` is `dLoss/dOutputTensor`, computed externally
         */
        void TrainStep(const InputTensor &x, const OutputTensor &grad, const float lr) {
            const auto A = ForwardAll(x);
            ZeroGrad();
            BackwardAll(A, grad);
            Update(lr);
        }

        // @doc: template<size_t Batch> void TrainableTensorNetwork::BatchTrainStep(const PrependBatch<Batch, InputTensor>::type &X, const PrependBatch<Batch, OutputTensor>::type &grad, const float lr)
        /**
         * Batched `ForwardAll` -> `ZeroGrad` -> `BatchedBackwardAll` -> `Update`
         * `grad` is batched `dLoss/dOutputTensor`, computed externally
         */
        template<size_t Batch>
        void BatchTrainStep(const PrependBatch<Batch, InputTensor>::type &X,
                            const PrependBatch<Batch, OutputTensor>::type &grad, const float lr) {
            const auto A = BatchedForwardAll<Batch>(X);
            ZeroGrad();
            BatchedBackwardAll<Batch>(A, grad);
            Update(lr);
        }

        // @doc: template<typename Loss> float TrainableTensorNetwork::Fit(const InputTensor &x, const OutputTensor &target, const float lr)
        /**
         * Parameterized by `Loss` (satisfying `LossFunction<Loss, OutputTensor>`)
         * Computes loss and grad internally, then runs `TrainStep`-equivalent; returns loss value
         */
        template<typename Loss>
        float Fit(const InputTensor &x, const OutputTensor &target, const float lr) {
            static_assert(LossFunction<Loss, OutputTensor>,
                          "Loss must satisfy the LossFunction concept");
            // inference
            const auto A = ForwardAll(x);
            // grab final activation
            const auto &pred = A.template get<NumBlocks>();
            // compute loss
            const float loss_val = Loss::Loss(pred, target);
            // differentiate loss wrt final output and target, creating the gradient that will kick off BackwardAll
            const auto grad = Loss::Grad(pred, target);
            // zero grads
            ZeroGrad();
            // backprop
            BackwardAll(A, grad);
            // update
            Update(lr);
            // return loss to caller
            return loss_val;
        }

        // @doc: template<typename Loss, size_t Batch> float TrainableTensorNetwork::BatchFit(const PrependBatch<Batch, InputTensor>::type &X, const PrependBatch<Batch, OutputTensor>::type &Y, const float lr)
        /**
         * Batched counterpart to `Fit`
         * Averages per-sample gradients across the batch before backpropagating, returns mean loss
         */
        template<typename Loss, size_t Batch>
        float BatchFit(const PrependBatch<Batch, InputTensor>::type &X,
                       const PrependBatch<Batch, OutputTensor>::type &Y, const float lr) {
            static_assert(LossFunction<Loss, OutputTensor>,
                          "Loss must expose static Loss(pred,target)->Tensor<> and Grad(pred,target)->OutputTensor");
            const auto A = BatchedForwardAll<Batch>(X);
            const auto &A_out = A.template get<NumBlocks>();

            // pass Loss::Loss through BatchZip to turn Tensor<Batch, OutputTensor>s A_out and Y -> Tensor<Batch>,
            // then reduce that with addition to get net loss for entire batch
            float loss_val = Reduce<0, Add>(BatchZip(A_out, Y, [](const auto &p, const auto &t) {
                return Loss::Loss(p, t);
            }));
            // same logic to turn Tensor<Batch, OutputTensor>s A_out and Y -> Tensor<Batch, OutputTensor>,
            //      ie the last-layer gradients for each member of the batch
            auto grad = BatchZip(A_out, Y, [](const auto &p, const auto &t) {
                return Loss::Grad(p, t);
            });

            // normalize loss and gradients over batch size
            const auto inv = 1.f / static_cast<float>(Batch);
            loss_val *= inv;
            // even though gradients are independently computed, when we call BatchedBackwardAll, all Batch
            // gradients will be summed into the same Params tensors. This would distort lr relative to Batch size,
            // which we do not want. The gradients stored into the Params (and then used for updates) will be
            // mean gradients over the Batch.
            grad *= inv;

            ZeroGrad();
            BatchedBackwardAll<Batch>(A, grad);
            Update(lr);
            return loss_val;
        }

        // @doc: template<typename Loss, size_t Batch, size_t N, size_t... InDims, size_t... OutDims> float TrainableTensorNetwork::RunEpoch(const Tensor<N, InDims...> &X_data, const Tensor<N, OutDims...> &Y_data, std::mt19937 &rng, const float lr)
        /**
         * Run one full epoch: `Steps = N / Batch` rounds of `BatchFit`, returning average loss per step
         * `X_data` and `Y_data` must already be in network shape (`Tensor<InDims...> == InputTensor`, `Tensor<OutDims...> == OutputTensor`), enforced by `static_assert`
         * Creates temporary batch `Tensor<Batch, InDims...>`/`Tensor<Batch, OutDims...>` pairs where all `Batch` indices -
         * and samples `Batch` indices per step from `[0, N)` using `rng`, applied to both `Tensor`s in the same loop
         */
        template<typename Loss, size_t Batch, size_t N, size_t... InDims, size_t... OutDims>
        float RunEpoch(const Tensor<N, InDims...> &X_data, const Tensor<N, OutDims...> &Y_data, std::mt19937 &rng,
                       const float lr) {
            static_assert(LossFunction<Loss, OutputTensor>,
                          "Loss must expose static Loss(pred,target)->Tensor<> and Grad(pred,target)->OutputTensor");
            static_assert(std::is_same_v<Tensor<InDims...>, InputTensor>,
                          "X_data sample shape must match network InputTensor");
            static_assert(std::is_same_v<Tensor<OutDims...>, OutputTensor>,
                          "Y_data sample shape must match network OutputTensor");
            // steps = dataset size / batch size
            static constexpr size_t Steps = N / Batch;
            // for random batches
            std::uniform_int_distribution<size_t> dist{0, N - 1};
            float total_loss = 0.f;
            // for each step
            for (size_t s = 0; s < Steps; ++s) {
                // get a batch of Xs and Ys, where
                typename PrependBatch<Batch, InputTensor>::type batch_X;
                typename PrependBatch<Batch, OutputTensor>::type batch_Y;
                // for each member of the batch,
                for (size_t b = 0; b < Batch; ++b) {
                    // get a random (row) index from the dataset
                    const size_t idx = dist(rng);
                    // set the idx-th 'row' subTensors of batch_X and batch_Y to the respective subTensors from the dataset
                    TensorSet<0>(batch_X, b, TensorGet<0>(X_data, idx));
                    TensorSet<0>(batch_Y, b, TensorGet<0>(Y_data, idx));
                }
                // accumulate loss
                total_loss += BatchFit<Loss, Batch>(batch_X, batch_Y, lr);
            }
            // average loss out by num steps (loss per individual example, because batch losses are normalized too)
            return total_loss / static_cast<float>(Steps);
        }
    };

    // @doc: template<Block... Blocks, size_t... InDims> auto operator>>(const Tensor<InDims...> &x, const TrainableTensorNetwork<Blocks...> &net)
    /**
     * `BatchMinorContract` form is part conventional and part performance-informed:
     * `Batch` being left-aligned adopts common convention for `Tensor` shapes in ML
     * `Minor` (contracted) axes being right-aligned reflects that `Tensor`s in `TTTN` are backed by ***row-major*** `float` arrays. Only the rightmost (minor) axes are stored contiguously in memory. To maximize vectorization optimizations for `Reduce ∘ zipWith(Map)` operations, we want the loops over contracted indices to be traversing contiguous memory. Detailed comments on this subject are resident in the code.
     */
    template<Block... Blocks, size_t... InDims>
        requires std::same_as<typename TrainableTensorNetwork<Blocks...>::InputTensor, Tensor<InDims...>>
    auto operator>>(const Tensor<InDims...>& x, const TrainableTensorNetwork<Blocks...>& net) {
        return net.Forward(x);
    }

    // @doc: template<Block... Blocks, size_t Batch, size_t... InDims> auto operator>>(const Tensor<Batch, InDims...> &X, const TrainableTensorNetwork<Blocks...> &net)
    /**
     * `BatchMinorContract` form is part conventional and part performance-informed:
     * `Batch` being left-aligned adopts common convention for `Tensor` shapes in ML
     * `Minor` (contracted) axes being right-aligned reflects that `Tensor`s in `TTTN` are backed by ***row-major*** `float` arrays. Only the rightmost (minor) axes are stored contiguously in memory. To maximize vectorization optimizations for `Reduce ∘ zipWith(Map)` operations, we want the loops over contracted indices to be traversing contiguous memory. Detailed comments on this subject are resident in the code.
     */
    template<Block... Blocks, size_t Batch, size_t... InDims>
        requires std::same_as<typename TrainableTensorNetwork<Blocks...>::InputTensor, Tensor<InDims...>>
    auto operator>>(const Tensor<Batch, InDims...>& X, const TrainableTensorNetwork<Blocks...>& net) {
        return net.template BatchedForward<Batch>(X);
    }

    // @doc: template<typename T> inline constexpr bool is_trainable_network_v
    /** ###### */
    template<typename T>
    inline constexpr bool is_trainable_network_v = false;

    template<Block... Bs>
    inline constexpr bool is_trainable_network_v<TrainableTensorNetwork<Bs...>> = true;

    // @doc: template<typename T> concept IsTrainableNetwork
    /** ###### */
    template<typename T>
    concept IsTrainableNetwork = is_trainable_network_v<T>;

} // namespace TTTN
