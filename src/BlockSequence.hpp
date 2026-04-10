#pragma once
#include <stdexcept>
#include <string>
#include "NetworkUtil.hpp"

namespace TTTN {
    // @doc: template<Block... Blocks> class BlockSequence
    /** Unified sequential core: wraps a shape-compliant chain of `Block`s and provides both the `Block` interface (for nesting) and the explicit activation API (for top-level training) */
    template<Block... Blocks>
    class BlockSequence {
        static_assert(sizeof...(Blocks) >= 1, "BlockSequence needs at least one block");

    public:
        // @doc: BlockSequence::NumBlocks
        /** `static constexpr size_t NumBlocks = sizeof...(Blocks)` */
        static constexpr size_t NumBlocks = sizeof...(Blocks);

    private:
        static constexpr size_t N = NumBlocks;

        // @doc: using BlockSequence::BlockTuple
        /** Type alias for `std::tuple<Blocks...>` */
        using BlockTuple = std::tuple<Blocks...>;

        // @doc: static constexpr bool BlockSequence::check_connected()
        /** Immediate `static_assert` function to ensure that `Block... Blocks` have compliant shapes: `std::is_same_v<typename std::tuple_element_t<Is, BlockTuple>::OutputTensor, typename std::tuple_element_t<Is + 1, BlockTuple>::InputTensor> && ...)` */
        static constexpr bool check_connected() {
            if constexpr (N == 1) return true;
            else
                return []<size_t... Is>(std::index_sequence<Is...>) {
                    return (std::is_same_v<
                        typename std::tuple_element_t<Is, BlockTuple>::OutputTensor,
                        typename std::tuple_element_t<Is + 1, BlockTuple>::InputTensor> && ...);
                }(std::make_index_sequence<N - 1>{});
        }

        static_assert(check_connected(), "BlockSequence: block output/input types don't chain");

    public:
        // @doc: BlockSequence::InputTensor
        /** Extract `InputTensor` type from first element of `BlockTuple` */
        using InputTensor = std::tuple_element_t<0, BlockTuple>::InputTensor;
        // @doc: BlockSequence::OutputTensor
        /** Extract `OutputTensor` type from last element of `BlockTuple` */
        using OutputTensor = std::tuple_element_t<N - 1, BlockTuple>::OutputTensor;
        // @doc: BlockSequence::InSize
        /** Convenience member for total size of `InputTensor` type */
        static constexpr size_t InSize = InputTensor::Size;
        // @doc: BlockSequence::OutSize
        /** Convenience member for total size of `OutputTensor` type */
        static constexpr size_t OutSize = OutputTensor::Size;

        // @doc: using BlockSequence::ActivationsTuple
        /** Type alias around `TensorTupleBuilder<Blocks...>::type` */
        using ActivationsTuple = TensorTupleBuilder<Blocks...>::type;
        // @doc: template<size_t Batch> using BlockSequence::BatchedActivationsTuple
        /** Type alias around `BatchedTensorTupleBuilder<Batch, Blocks...>::type` */
        template<size_t Batch>
        using BatchedActivationsTuple = BatchedTensorTupleBuilder<Batch, Blocks...>::type;
        // @doc: using BlockSequence::Activations
        /** Access-safe `ActivationsWrap` wrapper around `ActivationsTuple` */
        using Activations = ActivationsWrap<ActivationsTuple>;
        // @doc: template<size_t Batch> using BlockSequence::BatchedActivations
        /** Access-safe `ActivationsWrap` wrapper around `BatchedActivationsTuple` */
        template<size_t Batch>
        using BatchedActivations = ActivationsWrap<BatchedActivationsTuple<Batch> >;

        // @doc: static constexpr size_t BlockSequence::TotalParamCount
        /** Sum of parameter counts of all elements of `Blocks...` */
        static constexpr size_t TotalParamCount =
                (TupleParamCount<decltype(std::declval<Blocks &>().all_params())> + ...);

    private:
        // @doc: BlockSequence::mBlocks
        /** Default-constructed `BlockTuple` containing actual `Block` objects */
        BlockTuple mBlocks;
        // @doc: mutable BlockSequence::mActs
        /** Mutable `ActivationsTuple` cache used by the `Block` interface (`Forward`/`Backward`) so that `BlockSequence` can be used as a nested block without the caller managing activations */
        mutable ActivationsTuple mActs{};


        // @doc: template<size_t I = 0> void BlockSequence::forward_impl(ActivationsTuple &A) const
        /** Private implementation; recursively fills `ActivationsTuple &A` by calling each `Block::Forward` in order and storing result */
        template<size_t I = 0>
        void forward_impl(ActivationsTuple &A) const {
            if constexpr (I < N) {
                std::get<I + 1>(A) = std::get<I>(A) >> std::get<I>(mBlocks);
                forward_impl<I + 1>(A);
            }
        }

        // @doc: template<size_t I, typename Delta> requires IsTensor<Delta> && std::is_same_v<Delta, std::tuple_element_t<I, ActivationsTuple>> auto BlockSequence::backward_impl(const ActivationsTuple &A, const Delta &delta)
        /**
         * Starts with `Delta` (derivative of loss w.r.t. activation `I`), recurses down to `I == 1`, returning `InputTensor` gradient
         * At each `I`, calls `Block::Backward(delta, A[I], A[I-1])` or `Block::Backward(gradient wrt this block's output, this block's output, this block's input / previous block's output)`
         */
        template<size_t I, typename Delta> requires
            IsTensor<Delta> && std::is_same_v<Delta, std::tuple_element_t<I, ActivationsTuple> >
        auto backward_impl(const ActivationsTuple &A, const Delta &delta) {
            // the I-1-th block's gradient is a function of: downstream gradient, the I-th activation (output of I-1-th block), and the I-1-th activation (input to I-1-th block)
            const auto grad = std::get<I - 1>(mBlocks) << BackwardArgs{delta, std::get<I>(A), std::get<I - 1>(A)};
            if constexpr (I > 1) {
                return backward_impl<I - 1>(A, grad);
            } else {
                return grad;
            }
        }

        // @doc: template<size_t Batch, size_t I = 0> void BlockSequence::batched_forward_impl(BatchedActivationsTuple<Batch> &A) const
        /** Private implementation; recursively fills `BatchedActivationsTuple &A` by calling each `Block::BatchedForward` in order */
        template<size_t Batch, size_t I = 0>
        void batched_forward_impl(BatchedActivationsTuple<Batch> &A) const {
            if constexpr (I < N) {
                std::get<I + 1>(A) = std::get<I>(A) >> std::get<I>(mBlocks);
                batched_forward_impl<Batch, I + 1>(A);
            }
        }

        // @doc: template<size_t Batch, size_t I, typename Delta> requires IsTensor<Delta> && std::is_same_v<Delta, std::tuple_element_t<I, BatchedActivationsTuple<Batch>>> auto BlockSequence::batched_backward_impl(const BatchedActivationsTuple<Batch> &A, const Delta &delta)
        /** Same logic as `backward_impl` but calls `Block::BatchedBackward` at each step */
        template<size_t Batch, size_t I, typename Delta> requires
            IsTensor<Delta> && std::is_same_v<Delta, std::tuple_element_t<I, BatchedActivationsTuple<Batch> > >
        auto batched_backward_impl(const BatchedActivationsTuple<Batch> &A, const Delta &delta) {
            const auto grad = std::get<I - 1>(mBlocks) << BackwardArgs{delta, std::get<I>(A), std::get<I - 1>(A)};
            if constexpr (I > 1) return batched_backward_impl<Batch, I - 1>(A, grad);
            else return grad;
        }

        // @doc: template<size_t I, size_t Lo> auto BlockSequence::backward_range_impl(const ActivationsTuple &A, const std::tuple_element_t<I, ActivationsTuple> &delta)
        /** Uses cached `mActs` to run `backward_impl` and returns `InputTensor` gradient for the first block */
        template<size_t I, size_t Lo>
        auto backward_range_impl(const ActivationsTuple &A,
                                 const std::tuple_element_t<I, ActivationsTuple> &delta) {
            const auto grad = std::get<I - 1>(mBlocks) << BackwardArgs{delta, std::get<I>(A), std::get<I - 1>(A)};
            if constexpr (I - 1 > Lo) return backward_range_impl<I - 1, Lo>(A, grad);
            else return grad;
        }

        // @doc: template<size_t Batch, size_t I, size_t Lo> auto BlockSequence::batched_backward_range_impl(const BatchedActivationsTuple<Batch> &A, const std::tuple_element_t<I, BatchedActivationsTuple<Batch>> &delta)
        /**
         * Starts with `Delta` (derivative of loss w.r.t. activation `I`), recurses down to `I == 1`, returning `InputTensor` gradient
         * At each `I`, calls `Block::Backward(delta, A[I], A[I-1])` or `Block::Backward(gradient wrt this block's output, this block's output, this block's input / previous block's output)`
         */
        template<size_t Batch, size_t I, size_t Lo>
        auto batched_backward_range_impl(const BatchedActivationsTuple<Batch> &A,
                                         const std::tuple_element_t<I, BatchedActivationsTuple<Batch> > &delta) {
            const auto grad = std::get<I - 1>(mBlocks) << BackwardArgs{delta, std::get<I>(A), std::get<I - 1>(A)};
            if constexpr (I - 1 > Lo) return batched_backward_range_impl<Batch, I - 1, Lo>(A, grad);
            else return grad;
        }

    public:
        // @doc: BlockSequence::BlockSequence()
        /** Default construct `mBlocks` and `mActs` */
        BlockSequence() = default;

        // @doc: template<size_t I> const auto &BlockSequence::block() const
        /** Get a `const &` to the `I`-th `Block` in `BlockSequence::mBlocks` */
        template<size_t I>
        const auto &block() const { return std::get<I>(mBlocks); }

        auto all_params() {
            return [&]<size_t... Is>(std::index_sequence<Is...>) {
                return std::tuple_cat(std::get<Is>(mBlocks).all_params()...);
            }(std::make_index_sequence<N>{});
        }

        auto all_params() const {
            return [&]<size_t... Is>(std::index_sequence<Is...>) {
                return std::tuple_cat(std::get<Is>(mBlocks).all_params()...);
            }(std::make_index_sequence<N>{});
        }


        // @doc: [[nodiscard]] OutputTensor BlockSequence::Forward(const InputTensor &x) const
        /**
         * Forward pass returning `OutputTensor`
         * Delegates to `ForwardAll` and extracts back element
         * Satisfies `Block` interface; uses `mActs` cache so a caller can follow with `Backward`
         */
        [[nodiscard]] OutputTensor Forward(const InputTensor &x) const {
            std::get<0>(mActs) = x;
            forward_impl(mActs);
            return std::get<N>(mActs);
        }

        // @doc: InputTensor BlockSequence::Backward(const OutputTensor &delta, const OutputTensor &, const InputTensor &)
        /** Uses cached `mActs` to run `backward_impl` and returns `InputTensor` gradient for the first block */
        InputTensor Backward(const OutputTensor &delta, const OutputTensor &, const InputTensor &) {
            return backward_impl<N>(mActs, delta);
        }

        // @doc: template<size_t Batch> [[nodiscard]] PrependBatch<Batch, OutputTensor>::type BlockSequence::BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const
        /**
         * Batched forward pass returning `PrependBatch<Batch, OutputTensor>`; extracted from back of `BatchedForwardAll`
         * Satisfies `Block` interface
         */
        template<size_t Batch>
        [[nodiscard]] PrependBatch<Batch, OutputTensor>::type
        BatchedForward(const PrependBatch<Batch, InputTensor>::type &X) const {
            const auto A = BatchedForwardAll<Batch>(X);
            return A.template get<N>();
        }

        // @doc: template<size_t Batch> PrependBatch<Batch, InputTensor>::type BlockSequence::BatchedBackward(const PrependBatch<Batch, OutputTensor>::type &delta, const PrependBatch<Batch, OutputTensor>::type &, const PrependBatch<Batch, InputTensor>::type &a_prev)
        /** Re-runs `BatchedForward` from `a_prev` to reconstruct activation cache, then runs `batched_backward_impl`, returning batched `InputTensor` gradient */
        template<size_t Batch>
        PrependBatch<Batch, InputTensor>::type
        BatchedBackward(const PrependBatch<Batch, OutputTensor>::type &delta,
                        const PrependBatch<Batch, OutputTensor>::type &,
                        const PrependBatch<Batch, InputTensor>::type &a_prev) {
            BatchedActivationsTuple<Batch> acts;
            std::get<0>(acts) = a_prev;
            batched_forward_impl<Batch>(acts);
            return batched_backward_impl<Batch, N>(acts, delta);
        }


        // @doc: [[nodiscard]] Activations BlockSequence::ForwardAll(const InputTensor &x) const
        /**
         * Forward pass returning full `ActivationsWrap Activations` object
         * Calls `BlockSequence::forward_impl` on `InputTensor &x`
         */
        [[nodiscard]] Activations ForwardAll(const InputTensor &x) const {
            // declare acts tuple
            ActivationsTuple A;
            // fill first activation with input
            std::get<0>(A) = x;
            // call recursive impl to fill activations tuple
            forward_impl(A);
            return Activations{std::move(A)};
        }

        // @doc: template<size_t I, typename Delta> void BlockSequence::BackwardFrom(const Activations &A, const Delta &grad)
        /**
         * Start the backward sweep at activation index `I` (rather than always `NumBlocks`)
         * Flexible primitive to backpropagate from anywhere, provided the correctly-shaped gradient coming in
         */
        template<size_t I, typename Delta>
        void BackwardFrom(const Activations &A, const Delta &grad) {
            backward_impl<I>(A.tuple(), grad);
        }

        // @doc: void BlockSequence::BackwardAll(const Activations &A, const OutputTensor &grad)
        /**
         * Delegates to `BackwardFrom<NumBlocks>` - full backward from output to input
         * Gradients are accumulated into `Block` `Param` members
         */
        void BackwardAll(const Activations &A, const OutputTensor &grad) {
            BackwardFrom<N>(A, grad);
        }

        // @doc: template<size_t Batch> [[nodiscard]] BatchedActivations<Batch> BlockSequence::BatchedForwardAll(const PrependBatch<Batch, InputTensor>::type &X) const
        /** Batched forward pass returning full `ActivationsWrap BatchedActivations` object */
        template<size_t Batch>
        [[nodiscard]] BatchedActivations<Batch>
        BatchedForwardAll(const PrependBatch<Batch, InputTensor>::type &X) const {
            BatchedActivationsTuple<Batch> A;
            std::get<0>(A) = X;
            batched_forward_impl<Batch>(A);
            return BatchedActivations<Batch>{std::move(A)};
        }

        // @doc: template<size_t Batch, size_t I, typename Delta> void BlockSequence::BatchedBackwardFrom(const BatchedActivations<Batch> &A, const Delta &grad)
        /** Batched counterpart to `BackwardFrom` - start batched backward sweep at activation index `I` */
        template<size_t Batch, size_t I, typename Delta>
        void BatchedBackwardFrom(const BatchedActivations<Batch> &A, const Delta &grad) {
            BatchedBackwardRange<Batch, 0, I>(A, grad);
        }

        // @doc: template<size_t Batch> void BlockSequence::BatchedBackwardAll(const BatchedActivations<Batch> &A, const PrependBatch<Batch, OutputTensor>::type &grad)
        /** Delegates to `BatchedBackwardFrom<Batch, NumBlocks>` - full batched backward from output to input */
        template<size_t Batch>
        void BatchedBackwardAll(const BatchedActivations<Batch> &A,
                                const PrependBatch<Batch, OutputTensor>::type &grad) {
            BatchedBackwardRange<Batch, 0, N>(A, grad);
        }


        // @doc: template<size_t Lo, size_t Hi> auto BlockSequence::BackwardRange(const Activations &A, const std::tuple_element_t<Hi, ActivationsTuple> &grad)
        /** Batched counterpart to `BackwardFrom` - start batched backward sweep at activation index `I` */
        template<size_t Lo, size_t Hi>
        auto BackwardRange(const Activations &A,
                           const std::tuple_element_t<Hi, ActivationsTuple> &grad) {
            static_assert(Lo <= Hi && Hi <= N, "BackwardRange: bounds out of range");
            if constexpr (Lo == Hi) return grad;
            else return backward_range_impl<Hi, Lo>(A.tuple(), grad);
        }

        // @doc: template<size_t Batch, size_t Lo, size_t Hi> auto BlockSequence::BatchedBackwardRange(const BatchedActivations<Batch> &A, const std::tuple_element_t<Hi, BatchedActivationsTuple<Batch>> &grad)
        /** Delegates to `BatchedBackwardFrom<Batch, NumBlocks>` - full batched backward from output to input */
        template<size_t Batch, size_t Lo, size_t Hi>
        auto BatchedBackwardRange(const BatchedActivations<Batch> &A,
                                  const std::tuple_element_t<Hi, BatchedActivationsTuple<Batch> > &grad) {
            static_assert(Lo <= Hi && Hi <= N, "BatchedBackwardRange: bounds out of range");
            if constexpr (Lo == Hi) return grad;
            else {
                auto normed = grad;
                normed *= (1.f / static_cast<float>(Batch));
                return batched_backward_range_impl<Batch, Hi, Lo>(A.tuple(), normed);
            }
        }

        // @doc: void BlockSequence::ZeroGrad()
        /** Calls `ZeroAllGrads` on each `Block`'s `all_params()` */
        void ZeroGrad() {
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (ZeroAllGrads(std::get<Is>(mBlocks).all_params()), ...);
            }(std::make_index_sequence<N>{});
        }


        // @doc: void BlockSequence::peek(SnapshotMap &out, const std::string &prefix) const
        /** Forwards peek calls from each `PeekableBlock` in `mBlocks` into `out`, keyed by `prefix + "block_N."` */
        void peek(SnapshotMap &out, const std::string &prefix) const {
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                ([&] {
                    if constexpr (const auto &blk = std::get<Is>(mBlocks); PeekableBlock<std::remove_cvref_t<decltype(
                        blk)> >) {
                        blk.peek(out, prefix + "block_" + std::to_string(Is) + ".");
                    }
                }(), ...);
            }(std::make_index_sequence<N>{});
        }

        // @doc: [[nodiscard]] SnapshotMap BlockSequence::Snap() const
        /** Calls `SaveAll` on each `Block::all_params()`, which calls `Tensor` binary serialization */
        [[nodiscard]] SnapshotMap Snap() const {
            SnapshotMap out;
            peek(out, "");
            return out;
        }


        // @doc: void BlockSequence::Save(const std::string &path) const
        /** Calls `SaveAll` on each `Block::all_params()`, which calls `Tensor` binary serialization */
        void Save(const std::string &path) const {
            std::ofstream f(path, std::ios::binary);
            if (!f) {
                throw std::runtime_error("Cannot write: " + path);
            }
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (SaveAll(std::get<Is>(mBlocks).all_params(), f), ...);
            }(std::make_index_sequence<N>{});
        }

        // @doc: void BlockSequence::Load(const std::string &path)
        /** Calls `LoadAll` on each `Block::all_params()`, which calls `Tensor` binary deserialization */
        void Load(const std::string &path) {
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                throw std::runtime_error("Cannot read: " + path);
            }
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (LoadAll(std::get<Is>(mBlocks).all_params(), f), ...);
            }(std::make_index_sequence<N>{});
        }
    };

    template<template<typename...> class Seq, typename B, typename Idx>
    struct RepeatSeqImpl;

    template<template<typename... Bs> class Seq, typename B, size_t... Is>
        requires Block<B>
    struct RepeatSeqImpl<Seq, B, std::index_sequence<Is...> > {
        template<size_t>
        using Same = B;
        using type = Seq<Same<Is>...>;
    };

    template<typename B, size_t N> requires Block<B>
    using RepeatedBlockSequence = RepeatSeqImpl<BlockSequence, B, std::make_index_sequence<N> >::type;
} // namespace TTTN
