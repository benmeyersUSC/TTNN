#pragma once
#include <array>
#include <tuple>
#include <vector>
#include <cmath>
#include "TrainableTensorNetwork.hpp"
#include "TTTN_ML.hpp"

namespace TTTN {
    // @doc: struct NoLoss
    /** Sentinel type. Pass as `TrunkLoss` or any head loss to suppress that source's gradient. */
    struct NoLoss {};

    template<size_t TapIndex, typename HeadNet, typename HeadLoss>
        requires IsTrainableNetwork<HeadNet> &&
                 (std::is_same_v<HeadLoss, NoLoss> || LossFunction<HeadLoss, typename HeadNet::OutputTensor>)
    struct HeadBranch;

    template<typename T>
    inline constexpr bool is_head_branch_v = false;

    template<size_t TapIndex, typename HeadNet, typename HeadLoss>
    inline constexpr bool is_head_branch_v<HeadBranch<TapIndex, HeadNet, HeadLoss>> = true;

    template<typename T>
    concept IsHeadBranch = is_head_branch_v<T>;

    // @doc: template<size_t TapIndex, typename HeadNet, typename HeadLoss = NoLoss> struct HeadBranch
    /** Wraps a head network with its trunk tap index and loss. Use `NoLoss` to silence gradient. */
    template<size_t TapIndex, typename HeadNet, typename HeadLoss = NoLoss>
        requires IsTrainableNetwork<HeadNet> &&
                 (std::is_same_v<HeadLoss, NoLoss> || LossFunction<HeadLoss, typename HeadNet::OutputTensor>)
    struct HeadBranch {
        static constexpr size_t Tap = TapIndex;
        using Net  = HeadNet;
        using Loss = HeadLoss;
        HeadNet net;
    };


    // @doc: template<typename TrunkNet, typename... Heads> class BranchTrainer
    /**
     * Multi-head branched trainer.
     * Owns the trunk network, head networks, and per-source attribution storage.
     *
     * Sources are numbered:
     *   0           = trunk loss
     *   1..NumHeads = head I-1 (same order as the Heads... pack)
     * NoLoss sources contribute zero gradient and zero attribution every step.
     *
     * Two training paths:
     *   Fit<TrunkLoss, Batch>(...)            — standard combined backward, no attribution breakdown
     *   InstrumentedFit<TrunkLoss, Batch>(...)— per-source isolated backwards + attribution
     *
     * Query trajectory at any time:
     *   TrunkTrajectory()   — combined gross/net/efficiency from Param::metrics (always populated)
     *   SourceTrajectory()  — per-source breakdown (only meaningful after InstrumentedFit calls)
     *   ResetMetrics()      — zero both combined Param::metrics and per-source accumulators
     */
    template<typename TrunkNet, typename... Heads>
        requires IsTrainableNetwork<TrunkNet> && (IsHeadBranch<Heads> && ...)
    class BranchTrainer {
    public:
        static constexpr size_t NumHeads   = sizeof...(Heads);
        static constexpr size_t NumSources = NumHeads + 1; // trunk + one per head

    private:
        using InputTensor       = typename TrunkNet::InputTensor;
        using TrunkOutputTensor = typename TrunkNet::OutputTensor;

        static constexpr size_t TrunkParams = TrunkNet::TotalParamCount;

        static constexpr bool check_taps() {
            if constexpr (NumHeads <= 1) return true;
            std::array<size_t, NumHeads> t = {Heads::Tap...};
            for (size_t i = 1; i < NumHeads; ++i)
                if (t[i] < t[i - 1]) return false;
            return true;
        }
        static_assert(check_taps(), "HeadBranch tap indices must be non-decreasing");

        // Check: each head's InputTensor matches the trunk activation at its tap (using Batch=1).
        static constexpr bool check_head_types() {
            return []<size_t... I>(std::index_sequence<I...>) {
                return (std::is_same_v<
                    typename PrependBatch<1, typename std::tuple_element_t<I, std::tuple<Heads...>>::Net::InputTensor>::type,
                    std::remove_cvref_t<decltype(
                        std::declval<typename TrunkNet::template Activations<1>>().template get<
                            std::tuple_element_t<I, std::tuple<Heads...>>::Tap>()
                    )>
                > && ...);
            }(std::index_sequence_for<Heads...>{});
        }
        static_assert(check_head_types(),
                      "Each head's InputTensor must match the trunk activation tensor at its tap index");

        // ── Members ──────────────────────────────────────────────────────────

        TrunkNet trunk_;
        std::tuple<Heads...> heads_;

        AdamState trunk_adam_{};
        std::array<AdamState, NumHeads> head_adams_{};

        // Per-source flat grad capture buffers (temp; refilled each InstrumentedFit call).
        std::array<std::vector<float>, NumSources> g_s_;

        // Per-source trajectory accumulators (cumulative over InstrumentedFit calls).
        std::array<std::vector<float>, NumSources> source_gross_; // Σ|Δθ_s[i]| per element
        std::array<std::vector<float>, NumSources> source_net_;   // ΣΔθ_s[i]  per element

        // Temp buffer: trunk param values snapshot before each Update.
        std::vector<float> theta_buf_;

        // Metric III — StructuralPotential, populated by PrecomputeStructuralPotential().
        // Stays empty until first precompute call.
        std::vector<float> structural_potential_;

        // ── Flat param helpers ───────────────────────────────────────────────

        // Copy current trunk grad buffers → g_s_[s].
        void snap_grads(const size_t s) {
            size_t i = 0;
            std::apply([&](const auto&... ps) {
                ([&] {
                    for (size_t j = 0; j < ps.Size; ++j)
                        g_s_[s][i++] = ps.grad.flat(j);
                }(), ...);
            }, trunk_.all_params());
        }

        // Sum all g_s_ into trunk grad buffers (ZeroGrad first).
        void load_combined_grads() {
            trunk_.ZeroGrad();
            for (size_t s = 0; s < NumSources; ++s) {
                size_t i = 0;
                std::apply([&](auto&... ps) {
                    ([&] {
                        for (size_t j = 0; j < ps.Size; ++j)
                            ps.grad.flat(j) += g_s_[s][i++];
                    }(), ...);
                }, trunk_.all_params());
            }
        }

        // Copy current trunk param values → theta_buf_.
        void snap_theta() {
            size_t i = 0;
            std::apply([&](const auto&... ps) {
                ([&] {
                    for (size_t j = 0; j < ps.Size; ++j)
                        theta_buf_[i++] = ps.value.flat(j);
                }(), ...);
            }, trunk_.all_params());
        }

        // After Update: compute Δθ = θ_new − θ_old, attribute across sources proportionally.
        void attribute_step() {
            size_t i = 0;
            std::apply([&](const auto&... ps) {
                ([&] {
                    for (size_t j = 0; j < ps.Size; ++j) {
                        const float delta = ps.value.flat(j) - theta_buf_[i]; // actual Adam Δθ
                        float g_total = 0.f;
                        for (size_t s = 0; s < NumSources; ++s) g_total += g_s_[s][i];
                        const bool silent = std::abs(g_total) < 1e-10f;
                        for (size_t s = 0; s < NumSources; ++s) {
                            const float ds = silent
                                ? delta / static_cast<float>(NumSources)
                                : (g_s_[s][i] / g_total) * delta;
                            source_gross_[s][i] += std::abs(ds);
                            source_net_[s][i]   += ds;
                        }
                        ++i;
                    }
                }(), ...);
            }, trunk_.all_params());
        }

        // ── Backward chain helpers ────────────────────────────────────────────
        // Chains BackwardRange calls through trunk segments between tap sites,
        // summing in pre-computed per-head tap-input gradients along the way.
        // The initial `grad` should be the raw (pre-normalisation) per-sample sum.
        // BackwardRange applies 1/Batch internally; subsequent segments receive
        // already-scaled gradients — pass them through without re-normalising.

        template<size_t Batch, size_t I, size_t PrevTap, typename Cache>
        void backward_heads_chain(
            Cache &cache,
            const typename PrependBatch<Batch, TrunkOutputTensor>::type &grad,
            const std::tuple<typename PrependBatch<Batch, typename Heads::Net::InputTensor>::type...> &tap_grads)
        {
            constexpr size_t tapI = std::tuple_element_t<I, std::tuple<Heads...>>::Tap;
            auto g_at_tap = trunk_.template BackwardRange<Batch, tapI, PrevTap>(cache, grad);
            g_at_tap += std::get<I>(tap_grads);
            if constexpr (I > 0)
                backward_heads_chain<Batch, I - 1, tapI>(cache, g_at_tap, tap_grads);
            else
                trunk_.template BackwardRange<Batch, 0, tapI>(cache, g_at_tap);
        }

    public:
        BranchTrainer() {
            for (auto &v : g_s_)           v.assign(TrunkParams, 0.f);
            for (auto &v : source_gross_)  v.assign(TrunkParams, 0.f);
            for (auto &v : source_net_)    v.assign(TrunkParams, 0.f);
            theta_buf_.assign(TrunkParams, 0.f);
        }

        TrunkNet       &trunk()       { return trunk_; }
        const TrunkNet &trunk() const { return trunk_; }

        template<size_t I> auto       &head()       { return std::get<I>(heads_); }
        template<size_t I> const auto &head() const { return std::get<I>(heads_); }


        // ── Fit ──────────────────────────────────────────────────────────────
        // Standard combined backward — no per-source attribution.
        // Combined Param::metrics are still updated (Δθ accumulates every step).

        template<typename TrunkLoss = NoLoss, size_t Batch>
        float Fit(const typename PrependBatch<Batch, InputTensor>::type &X,
                  const typename PrependBatch<Batch, TrunkOutputTensor>::type &trunk_target,
                  const float trunk_lr,
                  const std::tuple<typename PrependBatch<Batch, typename Heads::Net::OutputTensor>::type...> &head_targets,
                  const std::array<float, NumHeads> &head_lrs)
        {
            // Forward trunk
            typename TrunkNet::template TrainingCache<Batch> trunk_cache;
            trunk_.template ForwardAll<Batch>(X, trunk_cache);
            const auto &trunk_pred = std::get<TrunkNet::NumBlocks>(trunk_cache.activations);

            // Forward heads and compute per-head tap grads
            using TapGradsTuple = std::tuple<typename PrependBatch<Batch, typename Heads::Net::InputTensor>::type...>;
            TapGradsTuple tap_grads{};
            float total_loss = 0.f;

            if constexpr (!std::is_same_v<TrunkLoss, NoLoss>)
                total_loss += TrunkLoss::Loss(trunk_pred, trunk_target).flat(0) * Batch;

            trunk_.ZeroGrad();
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                ([&] {
                    using HeadI = std::tuple_element_t<Is, std::tuple<Heads...>>;
                    using HL    = typename HeadI::Loss;
                    const auto &tap_act = std::get<HeadI::Tap>(trunk_cache.activations);
                    typename HeadI::Net::template TrainingCache<Batch> head_cache;
                    std::get<Is>(heads_).net.template ForwardAll<Batch>(tap_act, head_cache);
                    const auto &head_out = std::get<HeadI::Net::NumBlocks>(head_cache.activations);
                    if constexpr (!std::is_same_v<HL, NoLoss>) {
                        // per-sample head loss gradient
                        typename PrependBatch<Batch, typename HeadI::Net::OutputTensor>::type head_grad{};
                        for (size_t b = 0; b < Batch; ++b) {
                            typename HeadI::Net::OutputTensor pred_b, tgt_b;
                            for (size_t k = 0; k < HeadI::Net::OutputTensor::Size; ++k) {
                                pred_b.flat(k) = head_out.flat(b * HeadI::Net::OutputTensor::Size + k);
                                tgt_b.flat(k)  = std::get<Is>(head_targets).flat(b * HeadI::Net::OutputTensor::Size + k);
                            }
                            total_loss += HL::Loss(pred_b, tgt_b).flat(0);
                            const auto g = HL::Grad(pred_b, tgt_b);
                            for (size_t k = 0; k < HeadI::Net::OutputTensor::Size; ++k)
                                head_grad.flat(b * HeadI::Net::OutputTensor::Size + k) = g.flat(k);
                        }
                        std::get<Is>(heads_).net.ZeroGrad();
                        std::get<Is>(tap_grads) = std::get<Is>(heads_).net.template BackwardRange<Batch, 0, HeadI::Net::NumBlocks>(head_cache, head_grad);
                        head_adams_[Is].step();
                        std::get<Is>(heads_).net.Update(head_adams_[Is], head_lrs[Is]);
                    }
                }(), ...);
            }(std::index_sequence_for<Heads...>{});

            total_loss /= static_cast<float>(Batch);

            // Trunk backward: trunk loss + chained head contributions
            typename PrependBatch<Batch, TrunkOutputTensor>::type trunk_grad{};
            if constexpr (!std::is_same_v<TrunkLoss, NoLoss>) {
                for (size_t b = 0; b < Batch; ++b) {
                    TrunkOutputTensor pred_b, tgt_b;
                    for (size_t k = 0; k < TrunkOutputTensor::Size; ++k) {
                        pred_b.flat(k) = trunk_pred.flat(b * TrunkOutputTensor::Size + k);
                        tgt_b.flat(k)  = trunk_target.flat(b * TrunkOutputTensor::Size + k);
                    }
                    const auto g = TrunkLoss::Grad(pred_b, tgt_b);
                    for (size_t k = 0; k < TrunkOutputTensor::Size; ++k)
                        trunk_grad.flat(b * TrunkOutputTensor::Size + k) = g.flat(k);
                }
            }

            if constexpr (NumHeads > 0)
                backward_heads_chain<Batch, NumHeads - 1, TrunkNet::NumBlocks>(trunk_cache, trunk_grad, tap_grads);
            else
                trunk_.template BackwardRange<Batch, 0, TrunkNet::NumBlocks>(trunk_cache, trunk_grad);

            trunk_adam_.step();
            trunk_.Update(trunk_adam_, trunk_lr);
            return total_loss;
        }


        // ── InstrumentedFit ───────────────────────────────────────────────────
        // Per-source isolated backwards → proportional Δθ attribution.
        // Populates source_gross_ and source_net_ in addition to Param::metrics.

        template<typename TrunkLoss = NoLoss, size_t Batch>
        float InstrumentedFit(
            const typename PrependBatch<Batch, InputTensor>::type &X,
            const typename PrependBatch<Batch, TrunkOutputTensor>::type &trunk_target,
            const float trunk_lr,
            const std::tuple<typename PrependBatch<Batch, typename Heads::Net::OutputTensor>::type...> &head_targets,
            const std::array<float, NumHeads> &head_lrs)
        {
            // ── 1. Forward trunk ──────────────────────────────────────────────
            typename TrunkNet::template TrainingCache<Batch> trunk_cache;
            trunk_.template ForwardAll<Batch>(X, trunk_cache);
            const auto &trunk_pred = std::get<TrunkNet::NumBlocks>(trunk_cache.activations);

            // ── 2. Forward heads, compute losses and per-head tap grads ───────
            using TapGradsTuple = std::tuple<typename PrependBatch<Batch, typename Heads::Net::InputTensor>::type...>;
            TapGradsTuple tap_grads{};
            float total_loss = 0.f;

            if constexpr (!std::is_same_v<TrunkLoss, NoLoss>)
                total_loss += TrunkLoss::Loss(trunk_pred, trunk_target).flat(0) * Batch;

            // Forward each head and compute its tap grad. Store head caches for re-use in head update.
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                ([&] {
                    using HeadI = std::tuple_element_t<Is, std::tuple<Heads...>>;
                    using HL    = typename HeadI::Loss;
                    if constexpr (!std::is_same_v<HL, NoLoss>) {
                        const auto &tap_act = std::get<HeadI::Tap>(trunk_cache.activations);
                        typename HeadI::Net::template TrainingCache<Batch> head_cache;
                        std::get<Is>(heads_).net.template ForwardAll<Batch>(tap_act, head_cache);
                        const auto &head_out = std::get<HeadI::Net::NumBlocks>(head_cache.activations);
                        typename PrependBatch<Batch, typename HeadI::Net::OutputTensor>::type head_grad{};
                        for (size_t b = 0; b < Batch; ++b) {
                            typename HeadI::Net::OutputTensor pred_b, tgt_b;
                            for (size_t k = 0; k < HeadI::Net::OutputTensor::Size; ++k) {
                                pred_b.flat(k) = head_out.flat(b * HeadI::Net::OutputTensor::Size + k);
                                tgt_b.flat(k)  = std::get<Is>(head_targets).flat(b * HeadI::Net::OutputTensor::Size + k);
                            }
                            total_loss += HL::Loss(pred_b, tgt_b).flat(0);
                            const auto g = HL::Grad(pred_b, tgt_b);
                            for (size_t k = 0; k < HeadI::Net::OutputTensor::Size; ++k)
                                head_grad.flat(b * HeadI::Net::OutputTensor::Size + k) = g.flat(k);
                        }
                        // Backward through head net → tap grad for trunk
                        std::get<Is>(heads_).net.ZeroGrad();
                        std::get<Is>(tap_grads) = std::get<Is>(heads_).net.template BackwardRange<Batch, 0, HeadI::Net::NumBlocks>(head_cache, head_grad);
                        // Update head separately — not part of trunk attribution
                        head_adams_[Is].step();
                        std::get<Is>(heads_).net.Update(head_adams_[Is], head_lrs[Is]);
                    }
                }(), ...);
            }(std::index_sequence_for<Heads...>{});

            total_loss /= static_cast<float>(Batch);

            // ── 3. Snapshot θ_before ─────────────────────────────────────────
            snap_theta();

            // ── 4. Source 0: trunk loss only ──────────────────────────────────
            trunk_.ZeroGrad();
            if constexpr (!std::is_same_v<TrunkLoss, NoLoss>) {
                typename PrependBatch<Batch, TrunkOutputTensor>::type trunk_grad{};
                for (size_t b = 0; b < Batch; ++b) {
                    TrunkOutputTensor pred_b, tgt_b;
                    for (size_t k = 0; k < TrunkOutputTensor::Size; ++k) {
                        pred_b.flat(k) = trunk_pred.flat(b * TrunkOutputTensor::Size + k);
                        tgt_b.flat(k)  = trunk_target.flat(b * TrunkOutputTensor::Size + k);
                    }
                    const auto g = TrunkLoss::Grad(pred_b, tgt_b);
                    for (size_t k = 0; k < TrunkOutputTensor::Size; ++k)
                        trunk_grad.flat(b * TrunkOutputTensor::Size + k) = g.flat(k);
                }
                trunk_.template BackwardRange<Batch, 0, TrunkNet::NumBlocks>(trunk_cache, trunk_grad);
            }
            snap_grads(0);

            // ── 5. Sources 1..NumHeads: each head in isolation ────────────────
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                ([&] {
                    using HeadI = std::tuple_element_t<Is, std::tuple<Heads...>>;
                    using HL    = typename HeadI::Loss;
                    trunk_.ZeroGrad();
                    if constexpr (!std::is_same_v<HL, NoLoss>) {
                        // Head Is's contribution to the trunk: backward from tapIs to input
                        trunk_.template BackwardRange<Batch, 0, HeadI::Tap>(trunk_cache, std::get<Is>(tap_grads));
                    }
                    snap_grads(Is + 1);
                }(), ...);
            }(std::index_sequence_for<Heads...>{});

            // ── 6. Reconstruct combined grad and run Adam once ────────────────
            load_combined_grads();
            trunk_adam_.step();
            trunk_.Update(trunk_adam_, trunk_lr);

            // ── 7. Attribute the actual Δθ across sources ─────────────────────
            attribute_step();

            return total_loss;
        }


        // ── Trajectory queries ────────────────────────────────────────────────

        // Combined trajectory across all trunk params (from Param::metrics, always populated).
        struct TrunkTrajectorySnapshot {
            float gross;      // Σ_i gross_path[i]
            float net_norm;   // ||net_displacement||_2
            float efficiency; // net_norm / gross
        };

        TrunkTrajectorySnapshot TrunkTrajectory() const {
            float gross = 0.f, net_sq = 0.f;
            std::apply([&](const auto&... ps) {
                ([&] {
                    for (size_t i = 0; i < ps.metrics.gross_path.Size; ++i)
                        gross += ps.metrics.gross_path.flat(i);
                    for (size_t i = 0; i < ps.metrics.net_displacement.Size; ++i) {
                        const float d = ps.metrics.net_displacement.flat(i);
                        net_sq += d * d;
                    }
                }(), ...);
            }, trunk_.all_params());
            const float net = std::sqrt(net_sq);
            return {gross, net, gross > 0.f ? net / gross : 0.f};
        }

        // Per-source trajectory breakdown (populated by InstrumentedFit calls).
        struct SourceTrajectorySnapshot {
            std::array<float, NumSources> gross;      // Σ_i source_gross_[s][i]
            std::array<float, NumSources> net_norm;   // ||source_net_[s]||_2
            std::array<float, NumSources> efficiency; // net_norm[s] / gross[s]
        };

        SourceTrajectorySnapshot SourceTrajectory() const {
            SourceTrajectorySnapshot snap{};
            for (size_t s = 0; s < NumSources; ++s) {
                float net_sq = 0.f;
                for (size_t i = 0; i < TrunkParams; ++i) {
                    snap.gross[s] += source_gross_[s][i];
                    net_sq        += source_net_[s][i] * source_net_[s][i];
                }
                snap.net_norm[s]   = std::sqrt(net_sq);
                snap.efficiency[s] = snap.gross[s] > 0.f
                    ? snap.net_norm[s] / snap.gross[s] : 0.f;
            }
            return snap;
        }

        // Zero all trajectory accumulators (Param::metrics + per-source vectors).
        void ResetMetrics() {
            std::apply([](auto&... ps) { (ps.metrics.reset(), ...); }, trunk_.all_params());
            for (auto &v : source_gross_) std::fill(v.begin(), v.end(), 0.f);
            for (auto &v : source_net_)   std::fill(v.begin(), v.end(), 0.f);
        }


        // ── Leverage queries (Metric II + Metric III) ──────────────────────────

        // @doc: template<size_t Batch> std::vector<float> ComputeFunctionalInfluence(X)
        /**
         * Metric II — Functional Influence.
         * Per-parameter ||∂output/∂θ_i||_2 evaluated at the trunk's CURRENT weights.
         * Returns flat vector of size TrunkNet::TotalParamCount, in the order of trunk_.all_params().
         * Cost: Batch × OutSize backward passes (no Update — trained values, m, v, metrics untouched).
         * Snapshot quantity — call at checkpoints, not inside the training loop.
         */
        template<size_t Batch>
        std::vector<float> ComputeFunctionalInfluence(
            const typename PrependBatch<Batch, InputTensor>::type &X)
        {
            return TTTN::ComputeJacobianNorms<TrunkNet, Batch>(trunk_, X);
        }

        // @doc: template<size_t Batch, size_t KInits> void PrecomputeStructuralPotential(X_ref)
        /**
         * Metric III — Structural Potential under the Xavier init distribution.
         * Constructs `KInits` fresh trunk networks (each Xavier-initialised by its constructor),
         * computes per-parameter Jacobian norms on `X_ref` for each, averages the result.
         * Stores the result internally; access via `StructuralPotential()`.
         * Cost: KInits × Batch × OutSize backward passes — one-time precompute per architecture.
         * Idempotent: calling again recomputes from scratch.
         */
        template<size_t Batch, size_t KInits = 50>
        void PrecomputeStructuralPotential(
            const typename PrependBatch<Batch, InputTensor>::type &X_ref)
        {
            structural_potential_ = TTTN::ComputeStructuralPotentialOf<TrunkNet, Batch, KInits>(X_ref);
        }

        // @doc: const std::vector<float>& StructuralPotential() const
        /** Returns the precomputed StructuralPotential vector. Empty if PrecomputeStructuralPotential hasn't been called. */
        const std::vector<float> &StructuralPotential() const { return structural_potential_; }


        // ── Weighted trajectory queries ────────────────────────────────────────

        struct WeightedSourceSnapshot {
            std::array<float, NumSources> gross;      // Σ_i w_i · source_gross_[s][i]
            std::array<float, NumSources> net_norm;   // ||w_i · source_net_[s][i]||_2
            std::array<float, NumSources> efficiency; // net_norm[s] / gross[s]
        };

        // @doc: WeightedSourceSnapshot WeightedSourceTrajectory(weights) const
        /**
         * Per-source trajectory weighted by an arbitrary leverage vector (e.g. FunctionalInfluence
         * or StructuralPotential). `weights` must have size TrunkNet::TotalParamCount, in the same
         * flat order as `trunk_.all_params()`.
         *
         * gross[s]    = Σ_i  weights[i] · source_gross_[s][i]    (output-displacement attribution)
         * net_norm[s] = ||weights[i] · source_net_[s][i]||_2     (weighted directed displacement)
         * efficiency[s] = net_norm[s] / gross[s]
         */
        WeightedSourceSnapshot WeightedSourceTrajectory(const std::vector<float> &weights) const {
            WeightedSourceSnapshot snap{};
            for (size_t s = 0; s < NumSources; ++s) {
                float net_sq = 0.f;
                for (size_t i = 0; i < TrunkParams; ++i) {
                    snap.gross[s]   += weights[i] * source_gross_[s][i];
                    const float wn   = weights[i] * source_net_[s][i];
                    net_sq          += wn * wn;
                }
                snap.net_norm[s]   = std::sqrt(net_sq);
                snap.efficiency[s] = snap.gross[s] > 0.f
                    ? snap.net_norm[s] / snap.gross[s] : 0.f;
            }
            return snap;
        }

        // @doc: WeightedTrajectorySnapshot WeightedTrunkTrajectory(weights) const
        /** Combined-trunk trajectory (across all sources) weighted by the given leverage vector. */
        WeightedTrajectorySnapshot WeightedTrunkTrajectory(const std::vector<float> &weights) const {
            return TTTN::WeightedTrajectoryOf(trunk_, weights);
        }
    };

} // namespace TTTN
