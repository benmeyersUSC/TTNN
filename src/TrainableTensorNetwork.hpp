#pragma once
#include <random>
#include <memory>
#include <vector>
#include <cmath>
#include "BlockSequence.hpp"
#include "TTTN_ML.hpp"

namespace TTTN {
    // @doc: template<Block... Blocks> class TrainableTensorNetwork
    /**
     * Owns a BlockSequence and provides inference, serialization, and a MakeTrainer factory.
     * Training state (AdamState, forward caches) lives in NetworkTrainer, not here.
     */
    template<Block... Blocks>
    class TrainableTensorNetwork {
        static_assert(sizeof...(Blocks) >= 1, "Need at least one block");
        using Seq = BlockSequence<Blocks...>;
        Seq mSeq_;

    public:
        using InputTensor  = typename Seq::InputTensor;
        using OutputTensor = typename Seq::OutputTensor;
        static constexpr size_t InSize          = Seq::InSize;
        static constexpr size_t OutSize         = Seq::OutSize;
        static constexpr size_t NumBlocks       = Seq::NumBlocks;
        static constexpr size_t TotalParamCount = Seq::TotalParamCount;

        template<size_t Batch> using Activations    = typename Seq::template Activations<Batch>;
        template<size_t Batch> using TrainingCache  = typename Seq::template TrainingCache<Batch>;

        TrainableTensorNetwork() = default;

        template<size_t I>
        const auto &block() const { return mSeq_.template block<I>(); }

        // Inference
        template<size_t Batch>
        auto Forward(const typename PrependBatch<Batch, InputTensor>::type &X) const {
            return mSeq_.template Forward<Batch>(X);
        }

        template<size_t Batch>
        Activations<Batch> ForwardAll(const typename PrependBatch<Batch, InputTensor>::type &X) const {
            return mSeq_.template ForwardAll<Batch>(X);
        }

        // Training forward (populates cache)
        template<size_t Batch>
        auto Forward(const typename PrependBatch<Batch, InputTensor>::type &X, TrainingCache<Batch> &cache) const {
            return mSeq_.template Forward<Batch>(X, cache);
        }

        template<size_t Batch>
        Activations<Batch> ForwardAll(const typename PrependBatch<Batch, InputTensor>::type &X,
                                      TrainingCache<Batch> &cache) const {
            return mSeq_.template ForwardAll<Batch>(X, cache);
        }

        // Backward
        template<size_t Batch>
        auto BackwardAll(const typename PrependBatch<Batch, OutputTensor>::type &delta,
                         const TrainingCache<Batch> &cache) {
            return mSeq_.template BackwardAll<Batch>(delta, cache);
        }

        template<size_t Batch, size_t Lo, size_t Hi, typename GradT>
        auto BackwardRange(const TrainingCache<Batch> &cache, const GradT &grad) {
            return mSeq_.template BackwardRange<Batch, Lo, Hi>(cache, grad);
        }

        void ZeroGrad() { mSeq_.ZeroGrad(); }

        auto all_params()       { return mSeq_.all_params(); }
        auto all_params() const { return mSeq_.all_params(); }

        void Update(const AdamState &adam, float lr) {
            UpdateAll(mSeq_.all_params(), adam, lr);
        }

        [[nodiscard]] SnapshotMap Snap() const { return mSeq_.Snap(); }

        void Save(const std::string &path) const        { mSeq_.Save(path); }
        void Load(const std::string &path)              { mSeq_.Load(path); }
        void SaveForTraining(const std::string &path) const { mSeq_.SaveForTraining(path); }
        void LoadForTraining(const std::string &path)       { mSeq_.LoadForTraining(path); }
    };

    template<Block... Bs, size_t... InDims>
        requires std::same_as<typename TrainableTensorNetwork<Bs...>::InputTensor, Tensor<InDims...>>
    auto operator>>(const Tensor<InDims...>& x, const TrainableTensorNetwork<Bs...>& net) {
        // single-sample: wrap as Batch=1
        using BInT = typename PrependBatch<1, Tensor<InDims...>>::type;
        BInT X; for (size_t i = 0; i < Tensor<InDims...>::Size; ++i) X.flat(i) = x.flat(i);
        auto Y = net.template Forward<1>(X);
        typename TrainableTensorNetwork<Bs...>::OutputTensor y;
        for (size_t i = 0; i < y.Size; ++i) y.flat(i) = Y.flat(i);
        return y;
    }

    template<Block... Bs, size_t Batch, size_t... InDims>
        requires std::same_as<typename TrainableTensorNetwork<Bs...>::InputTensor, Tensor<InDims...>>
    auto operator>>(const Tensor<Batch, InDims...>& X, const TrainableTensorNetwork<Bs...>& net) {
        return net.template Forward<Batch>(X);
    }

    template<typename T>
    inline constexpr bool is_trainable_network_v = false;

    template<Block... Bs>
    inline constexpr bool is_trainable_network_v<TrainableTensorNetwork<Bs...>> = true;

    template<typename T>
    concept IsTrainableNetwork = is_trainable_network_v<T>;


    // ── Shared trajectory / leverage helpers (used by NetworkTrainer & BranchTrainer) ─────

    // @doc: struct WeightedTrajectorySnapshot
    /** Aggregate of leverage-weighted trajectory metrics. */
    struct WeightedTrajectorySnapshot {
        float gross;      // Σ_i w_i · gross_path[i]
        float net_norm;   // ||w_i · net_displacement[i]||_2
        float efficiency; // net_norm / gross
    };

    // @doc: template<typename Net> WeightedTrajectorySnapshot WeightedTrajectoryOf(net, weights)
    /**
     * Walks net.all_params(), aggregates Param::metrics weighted by `weights[i]` (per flat-element).
     * `weights.size()` must equal Net::TotalParamCount.
     */
    template<typename Net>
        requires IsTrainableNetwork<Net>
    WeightedTrajectorySnapshot WeightedTrajectoryOf(const Net &net, const std::vector<float> &weights) {
        float gross = 0.f, net_sq = 0.f;
        size_t i = 0;
        std::apply([&](const auto&... ps) {
            ([&] {
                for (size_t k = 0; k < ps.metrics.gross_path.Size; ++k) {
                    const float w  = weights[i];
                    gross         += w * ps.metrics.gross_path.flat(k);
                    const float wn = w * ps.metrics.net_displacement.flat(k);
                    net_sq        += wn * wn;
                    ++i;
                }
            }(), ...);
        }, net.all_params());
        const float n = std::sqrt(net_sq);
        return {gross, n, gross > 0.f ? n / gross : 0.f};
    }

    // @doc: template<typename Net, size_t Batch> std::vector<float> ComputeJacobianNorms(net, X)
    /**
     * Per-parameter L2 norm of ∂output/∂θ_i, averaged over `Batch` input samples.
     * Returns a flat vector of size Net::TotalParamCount, in the order of net.all_params().
     *
     * Computes E_x[||J(x)||] (per-sample norm averaged), not ||E_x[J(x)]|| (norm of average).
     * Modifies net.grad transiently and zeroes it before returning. Does NOT touch m, v, or
     * Param::metrics. Cost: Batch × OutSize backward passes.
     */
    template<typename Net, size_t Batch>
        requires IsTrainableNetwork<Net>
    std::vector<float> ComputeJacobianNorms(
        Net &net,
        const typename PrependBatch<Batch, typename Net::InputTensor>::type &X)
    {
        constexpr size_t P       = Net::TotalParamCount;
        constexpr size_t OutSize = Net::OutputTensor::Size;

        std::vector<float> avg_norm(P, 0.f);

        for (size_t b = 0; b < Batch; ++b) {
            typename PrependBatch<1, typename Net::InputTensor>::type Xb;
            for (size_t k = 0; k < Net::InputTensor::Size; ++k)
                Xb.flat(k) = X.flat(b * Net::InputTensor::Size + k);

            typename Net::template TrainingCache<1> cache;
            net.template ForwardAll<1>(Xb, cache);

            std::vector<float> norm_sq(P, 0.f);

            for (size_t j = 0; j < OutSize; ++j) {
                net.ZeroGrad();
                typename PrependBatch<1, typename Net::OutputTensor>::type ej{};
                ej.flat(j) = 1.f;
                net.template BackwardRange<1, 0, Net::NumBlocks>(cache, ej);

                size_t i = 0;
                std::apply([&](const auto&... ps) {
                    ([&] {
                        for (size_t k = 0; k < ps.Size; ++k) {
                            const float g = ps.grad.flat(k);
                            norm_sq[i++] += g * g;
                        }
                    }(), ...);
                }, net.all_params());
            }

            for (size_t i = 0; i < P; ++i)
                avg_norm[i] += std::sqrt(norm_sq[i]);
        }

        for (auto &v : avg_norm) v /= static_cast<float>(Batch);
        net.ZeroGrad();
        return avg_norm;
    }

    // @doc: template<typename Net, size_t Batch, size_t KInits> std::vector<float> ComputeStructuralPotentialOf(X_ref)
    /**
     * Heap-allocates `KInits` fresh Net instances (each Xavier-init via constructor),
     * computes ComputeJacobianNorms on each against `X_ref`, returns the average.
     * One-time precompute per architecture — Net is unchanged by this call.
     * Cost: KInits × Batch × OutSize backward passes.
     */
    template<typename Net, size_t Batch, size_t KInits = 50>
        requires IsTrainableNetwork<Net>
    std::vector<float> ComputeStructuralPotentialOf(
        const typename PrependBatch<Batch, typename Net::InputTensor>::type &X_ref)
    {
        constexpr size_t P = Net::TotalParamCount;
        std::vector<float> sp(P, 0.f);

        for (size_t k = 0; k < KInits; ++k) {
            auto temp = std::make_unique<Net>(); // heap-allocated; Xavier-init via constructor
            const auto influence = ComputeJacobianNorms<Net, Batch>(*temp, X_ref);
            for (size_t i = 0; i < P; ++i)
                sp[i] += influence[i];
        }
        for (auto &v : sp) v /= static_cast<float>(KInits);
        return sp;
    }


    // @doc: template<typename Net, size_t Batch_> class NetworkTrainer
    /**
     * Training wrapper around a TrainableTensorNetwork (held by reference).
     * Owns AdamState, TrainingCache<Batch_>, and all training entry points.
     * Shed it when training is complete; the Net retains only weights.
     */
    template<typename Net, size_t Batch_>
        requires IsTrainableNetwork<Net>
    class NetworkTrainer {
        Net &net_;
        AdamState adam_{};
        typename Net::template TrainingCache<Batch_> cache_{};
        std::vector<float> structural_potential_; // populated lazily by PrecomputeStructuralPotential

        using InputT  = typename Net::InputTensor;
        using OutputT = typename Net::OutputTensor;
        using BatchIn  = typename PrependBatch<Batch_, InputT>::type;
        using BatchOut = typename PrependBatch<Batch_, OutputT>::type;

    public:
        static constexpr size_t Batch = Batch_;

        explicit NetworkTrainer(Net &net) : net_(net) {}

        // Forward (training — populates cache)
        BatchOut Forward(const BatchIn &X) {
            return net_.template Forward<Batch_>(X, cache_);
        }

        // Backward — accumulates param grads
        BatchIn Backward(const BatchOut &dY) {
            return net_.template BackwardAll<Batch_>(dY, cache_);
        }

        // BackwardRange — partial backward for BranchTrainer tap routing
        template<size_t Lo, size_t Hi, typename GradT>
        auto BackwardRange(const GradT &grad) {
            return net_.template BackwardRange<Batch_, Lo, Hi>(cache_, grad);
        }

        void ZeroGrad()            { net_.ZeroGrad(); }
        void Update(float lr)      { adam_.step(); net_.Update(adam_, lr); }

        const auto &cache() const  { return cache_; }
        AdamState  &adam()         { return adam_; }

        // @doc: struct NetworkTrainer::TrajectorySnapshot
        /** Aggregate trajectory metrics across all parameters in the network. */
        struct TrajectorySnapshot {
            float gross_path;        // Σ_i gross_path[i] — total L1 distance walked
            float net_norm;          // ||net_displacement||_2 — L2 norm of displacement vector
            float efficiency_ratio;  // net_norm / gross_path; 0 = pure churn, 1 = geodesic
        };

        // @doc: TrajectorySnapshot NetworkTrainer::Trajectory() const
        /** Aggregate trajectory metrics from all Param::metrics fields in the network. */
        TrajectorySnapshot Trajectory() const {
            float gross = 0.f, net_sq = 0.f;
            std::apply([&](const auto &... ps) {
                ([&] {
                    for (size_t i = 0; i < ps.metrics.gross_path.Size; ++i)
                        gross += ps.metrics.gross_path.flat(i);
                    for (size_t i = 0; i < ps.metrics.net_displacement.Size; ++i) {
                        const float d = ps.metrics.net_displacement.flat(i);
                        net_sq += d * d;
                    }
                }(), ...);
            }, net_.all_params());
            const float net = std::sqrt(net_sq);
            return {gross, net, gross > 0.f ? net / gross : 0.f};
        }

        // @doc: void NetworkTrainer::ResetMetrics()
        /** Zero all Param::metrics fields — call at the start of a measurement window. */
        void ResetMetrics() {
            std::apply([](auto &... ps) { (ps.metrics.reset(), ...); }, net_.all_params());
        }

        // ── Leverage queries (Metric II + Metric III) ──────────────────────────

        // @doc: std::vector<float> NetworkTrainer::ComputeFunctionalInfluence(X)
        /** Per-parameter ||∂output/∂θ_i|| at the network's current weights, averaged over `X`. */
        std::vector<float> ComputeFunctionalInfluence(const BatchIn &X) {
            return TTTN::ComputeJacobianNorms<Net, Batch_>(net_, X);
        }

        // @doc: template<size_t KInits = 50> void NetworkTrainer::PrecomputeStructuralPotential(X_ref)
        /** Compute and cache StructuralPotential under the Net's Xavier init distribution. */
        template<size_t KInits = 50>
        void PrecomputeStructuralPotential(const BatchIn &X_ref) {
            structural_potential_ = TTTN::ComputeStructuralPotentialOf<Net, Batch_, KInits>(X_ref);
        }

        // @doc: const std::vector<float>& NetworkTrainer::StructuralPotential() const
        const std::vector<float> &StructuralPotential() const { return structural_potential_; }

        // @doc: WeightedTrajectorySnapshot NetworkTrainer::WeightedTrajectory(weights) const
        /** Aggregate Param::metrics weighted by an arbitrary leverage vector. */
        WeightedTrajectorySnapshot WeightedTrajectory(const std::vector<float> &weights) const {
            return TTTN::WeightedTrajectoryOf(net_, weights);
        }

        // @doc: template<typename Loss> float NetworkTrainer::Fit(X, Y, lr)
        template<typename Loss>
        float Fit(const BatchIn &X, const BatchOut &Y, float lr) {
            const BatchOut pred = Forward(X);
            float loss_val = 0.f;
            BatchOut grad{};
            for (size_t b = 0; b < Batch_; ++b) {
                OutputT pred_b, y_b;
                for (size_t i = 0; i < OutputT::Size; ++i) {
                    pred_b.flat(i) = pred.flat(b * OutputT::Size + i);
                    y_b.flat(i)    = Y.flat(b * OutputT::Size + i);
                }
                loss_val += Loss::Loss(pred_b, y_b).flat(0);
                const auto g = Loss::Grad(pred_b, y_b);
                for (size_t i = 0; i < OutputT::Size; ++i)
                    grad.flat(b * OutputT::Size + i) = g.flat(i);
            }
            loss_val /= static_cast<float>(Batch_);
            ZeroGrad();
            Backward(grad);
            Update(lr);
            return loss_val;
        }

        // @doc: template<typename Loss, size_t N> float NetworkTrainer::RunEpoch(X_data, Y_data, rng, lr)
        template<typename Loss, size_t N, size_t... InDims, size_t... OutDims>
        float RunEpoch(const Tensor<N, InDims...> &X_data, const Tensor<N, OutDims...> &Y_data,
                       std::mt19937 &rng, float lr) {
            static_assert(LossFunction<Loss, OutputT>, "Loss must satisfy LossFunction");
            static_assert(std::is_same_v<Tensor<InDims...>, InputT>,  "X_data sample shape mismatch");
            static_assert(std::is_same_v<Tensor<OutDims...>, OutputT>, "Y_data sample shape mismatch");
            static constexpr size_t Steps = N / Batch_;
            std::uniform_int_distribution<size_t> dist{0, N - 1};
            float total_loss = 0.f;
            for (size_t s = 0; s < Steps; ++s) {
                BatchIn  batch_X;
                BatchOut batch_Y;
                for (size_t b = 0; b < Batch_; ++b) {
                    const size_t idx = dist(rng);
                    TensorSet<0>(batch_X, b, TensorGet<0>(X_data, idx));
                    TensorSet<0>(batch_Y, b, TensorGet<0>(Y_data, idx));
                }
                total_loss += Fit<Loss>(batch_X, batch_Y, lr);
            }
            return total_loss / static_cast<float>(Steps);
        }

        void SaveTrainingState(const std::string &path) const {
            net_.SaveForTraining(path);
            std::ofstream f(path, std::ios::binary | std::ios::app);
            f.write(reinterpret_cast<const char*>(&adam_), sizeof(AdamState));
        }

        void LoadTrainingState(const std::string &path) {
            net_.LoadForTraining(path);
            std::ifstream f(path, std::ios::binary);
            f.seekg(0, std::ios::end);
            const auto end = f.tellg();
            f.seekg(end - static_cast<std::streamoff>(sizeof(AdamState)));
            f.read(reinterpret_cast<char*>(&adam_), sizeof(AdamState));
        }
    };

} // namespace TTTN
