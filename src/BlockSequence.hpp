#pragma once
#include <array>
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
        static constexpr size_t NumBlocks = sizeof...(Blocks);

    private:
        static constexpr size_t N = NumBlocks;
        using BlockTuple = std::tuple<Blocks...>;

        static constexpr bool check_connected() {
            if constexpr (N == 1) return true;
            else return []<size_t... Is>(std::index_sequence<Is...>) {
                return (std::is_same_v<
                    typename std::tuple_element_t<Is, BlockTuple>::OutputTensor,
                    typename std::tuple_element_t<Is + 1, BlockTuple>::InputTensor> && ...);
            }(std::make_index_sequence<N - 1>{});
        }
        static_assert(check_connected(), "BlockSequence: block output/input types don't chain");

    public:
        using InputTensor  = typename std::tuple_element_t<0,     BlockTuple>::InputTensor;
        using OutputTensor = typename std::tuple_element_t<N - 1, BlockTuple>::OutputTensor;
        static constexpr size_t InSize  = InputTensor::Size;
        static constexpr size_t OutSize = OutputTensor::Size;

        static constexpr size_t TotalParamCount =
            (TupleParamCount<decltype(std::declval<Blocks &>().all_params())> + ...);

        // Per-block param counts, in the order Blocks... appear.
        static constexpr std::array<size_t, NumBlocks> BlockParamCounts = {
            TupleParamCount<decltype(std::declval<Blocks &>().all_params())>...
        };

        // Metric I — Positional Leverage per block.
        // Block k's leverage = (# params strictly downstream of k) / TotalParamCount  ∈ [0, 1].
        // Early blocks (deep upstream) score highest; the final block scores 0.
        static constexpr std::array<float, NumBlocks> PositionalLeveragePerBlock = [] {
            std::array<float, NumBlocks> r{};
            size_t downstream = 0;
            for (size_t i = 0; i < NumBlocks; ++i) {
                const size_t k = NumBlocks - 1 - i;
                r[k] = TotalParamCount > 0
                     ? static_cast<float>(downstream) / static_cast<float>(TotalParamCount)
                     : 0.f;
                downstream += BlockParamCounts[k];
            }
            return r;
        }();

        // Metric I — Positional Leverage per parameter (flat, all_params() order).
        // Every parameter in block k receives PositionalLeveragePerBlock[k].
        static constexpr std::array<float, TotalParamCount> PositionalLeveragePerParam = [] {
            std::array<float, TotalParamCount> r{};
            size_t idx = 0;
            for (size_t k = 0; k < NumBlocks; ++k) {
                const float lev = PositionalLeveragePerBlock[k];
                for (size_t i = 0; i < BlockParamCounts[k]; ++i) r[idx++] = lev;
            }
            return r;
        }();

        // N+1 batched tensors: [X, A0, A1, ..., A_{N-1}]
        template<size_t Batch>
        using ActivationsTuple = typename TensorTupleBuilder<Batch, Blocks...>::type;

        template<size_t Batch>
        using Activations = ActivationsWrap<ActivationsTuple<Batch>>;

        // @doc: TrainingCache<Batch>
        /** Access-safe `ActivationsWrap` wrapper around `BatchedActivationsTuple` */
        template<size_t Batch>
        struct TrainingCacheData {
            ActivationsTuple<Batch> activations;
            std::tuple<typename Blocks::template TrainingCache<Batch>...> block_caches;
        };
        template<size_t Batch> using TrainingCache = TrainingCacheData<Batch>;

    private:
        BlockTuple mBlocks;

        // Pure forward: fills activations tuple using >> (inference Forward).
        template<size_t Batch, size_t I = 0>
        void forward_pure_impl(ActivationsTuple<Batch> &A) const {
            if constexpr (I < N) {
                std::get<I + 1>(A) = std::get<I>(A) >> std::get<I>(mBlocks);
                forward_pure_impl<Batch, I + 1>(A);
            }
        }

        // Training forward: calls Forward(X, cache) on each block; fills activations + block_caches.
        template<size_t Batch, size_t I = 0>
        void forward_training_impl(TrainingCache<Batch> &cache) const {
            if constexpr (I < N) {
                std::get<I + 1>(cache.activations) =
                    std::get<I>(mBlocks).template Forward<Batch>(
                        std::get<I>(cache.activations),
                        std::get<I>(cache.block_caches));
                forward_training_impl<Batch, I + 1>(cache);
            }
        }

        // Backward range: recurses from block I-1 down to block Lo.
        template<size_t Batch, size_t I, size_t Lo>
        auto backward_range_impl(const TrainingCache<Batch> &cache,
                                 const std::tuple_element_t<I, ActivationsTuple<Batch>> &delta) {
            const auto grad = std::get<I - 1>(mBlocks).template Backward<Batch>(
                delta,
                std::get<I>(cache.activations),
                std::get<I - 1>(cache.activations),
                std::get<I - 1>(cache.block_caches));
            if constexpr (I - 1 > Lo) return backward_range_impl<Batch, I - 1, Lo>(cache, grad);
            else                       return grad;
        }

    public:
        BlockSequence() = default;

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

        // ── Block interface (for nesting) ────────────────────────────────────

        // @doc: template<size_t Batch> Forward(X) -- pure inference; no cache.
        template<size_t Batch>
        auto Forward(const typename PrependBatch<Batch, InputTensor>::type &X) const
            -> typename PrependBatch<Batch, OutputTensor>::type {
            ActivationsTuple<Batch> A;
            std::get<0>(A) = X;
            forward_pure_impl<Batch>(A);
            return std::get<N>(A);
        }

        // @doc: template<size_t Batch> Forward(X, cache) -- training; populates cache.
        template<size_t Batch>
        auto Forward(const typename PrependBatch<Batch, InputTensor>::type &X,
                     TrainingCache<Batch> &cache) const
            -> typename PrependBatch<Batch, OutputTensor>::type {
            std::get<0>(cache.activations) = X;
            forward_training_impl<Batch>(cache);
            return std::get<N>(cache.activations);
        }

        // @doc: template<size_t Batch> Backward(dY, a, a_prev, cache)
        // a and a_prev ignored -- BlockSequence reads activations from cache.
        template<size_t Batch>
        auto Backward(const typename PrependBatch<Batch, OutputTensor>::type &delta_A,
                      const typename PrependBatch<Batch, OutputTensor>::type & /*a*/,
                      const typename PrependBatch<Batch, InputTensor>::type   & /*a_prev*/,
                      const TrainingCache<Batch> &cache)
            -> typename PrependBatch<Batch, InputTensor>::type {
            return backward_range_impl<Batch, N, 0>(cache, delta_A);
        }

        // ── Top-level API (used by NetworkTrainer) ───────────────────────────

        // @doc: template<size_t Batch> ForwardAll(X) -- pure inference; returns Activations wrap.
        template<size_t Batch>
        Activations<Batch> ForwardAll(const typename PrependBatch<Batch, InputTensor>::type &X) const {
            ActivationsTuple<Batch> A;
            std::get<0>(A) = X;
            forward_pure_impl<Batch>(A);
            return Activations<Batch>{std::move(A)};
        }

        // @doc: template<size_t Batch> ForwardAll(X, cache) -- training forward; returns Activations view.
        template<size_t Batch>
        Activations<Batch> ForwardAll(const typename PrependBatch<Batch, InputTensor>::type &X,
                                      TrainingCache<Batch> &cache) const {
            std::get<0>(cache.activations) = X;
            forward_training_impl<Batch>(cache);
            return Activations<Batch>{cache.activations};
        }

        // @doc: template<size_t Batch> BackwardAll(delta, cache) -- full backward from output to input.
        template<size_t Batch>
        auto BackwardAll(const typename PrependBatch<Batch, OutputTensor>::type &delta,
                         const TrainingCache<Batch> &cache)
            -> typename PrependBatch<Batch, InputTensor>::type {
            return backward_range_impl<Batch, N, 0>(cache, delta);
        }

        // @doc: template<size_t Batch, size_t Lo, size_t Hi> BackwardRange(cache, grad) -- partial backward.
        /**
         * Starts with `Delta` (derivative of loss w.r.t. activation `I`), recurses down to `I == 1`, returning `InputTensor` gradient
         * At each `I`, calls `Block::Backward(delta, A[I], A[I-1])` or `Block::Backward(gradient wrt this block's output, this block's output, this block's input / previous block's output)`
         */
        // Note: normalises by 1/Batch before entry (mirrors old BatchedBackwardRange behaviour).
        template<size_t Batch, size_t Lo, size_t Hi>
        auto BackwardRange(const TrainingCache<Batch> &cache,
                           const std::tuple_element_t<Hi, ActivationsTuple<Batch>> &grad)
            -> std::tuple_element_t<Lo, ActivationsTuple<Batch>> {
            static_assert(Lo <= Hi && Hi <= N, "BackwardRange: bounds out of range");
            if constexpr (Lo == Hi) return grad;
            else {
                auto normed = grad;
                normed *= (1.f / static_cast<float>(Batch));
                return backward_range_impl<Batch, Hi, Lo>(cache, normed);
            }
        }

        void ZeroGrad() {
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (ZeroAllGrads(std::get<Is>(mBlocks).all_params()), ...);
            }(std::make_index_sequence<N>{});
        }

        void peek(SnapshotMap &out, const std::string &prefix) const {
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                ([&] {
                    if constexpr (const auto &blk = std::get<Is>(mBlocks);
                                  PeekableBlock<std::remove_cvref_t<decltype(blk)>>) {
                        blk.peek(out, prefix + "block_" + std::to_string(Is) + ".");
                    }
                }(), ...);
            }(std::make_index_sequence<N>{});
        }

        [[nodiscard]] SnapshotMap Snap() const { SnapshotMap out; peek(out, ""); return out; }

        void Save(const std::string &path) const {
            std::ofstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot write: " + path);
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (SaveAllWeights(std::get<Is>(mBlocks).all_params(), f), ...);
            }(std::make_index_sequence<N>{});
        }

        void Load(const std::string &path) {
            std::ifstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot read: " + path);
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (LoadAllWeights(std::get<Is>(mBlocks).all_params(), f), ...);
            }(std::make_index_sequence<N>{});
        }

        void SaveForTraining(const std::string &path) const {
            std::ofstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot write: " + path);
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (SaveAll(std::get<Is>(mBlocks).all_params(), f), ...);
            }(std::make_index_sequence<N>{});
        }

        void LoadForTraining(const std::string &path) {
            std::ifstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot read: " + path);
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (LoadAll(std::get<Is>(mBlocks).all_params(), f), ...);
            }(std::make_index_sequence<N>{});
        }
    };


    template<template<typename...> class Seq, typename B, typename Idx>
    struct RepeatSeqImpl;

    template<template<typename... Bs> class Seq, typename B, size_t... Is>
        requires Block<B>
    struct RepeatSeqImpl<Seq, B, std::index_sequence<Is...>> {
        template<size_t> using Same = B;
        using type = Seq<Same<Is>...>;
    };

    template<typename B, size_t N> requires Block<B>
    using RepeatedBlockSequence = typename RepeatSeqImpl<BlockSequence, B, std::make_index_sequence<N>>::type;

} // namespace TTTN
