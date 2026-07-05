#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <fstream>
#include "TrainableTensorNetwork.hpp"

namespace TTTN {

    // @doc: template<typename Net, size_t Batch> std::vector<typename Net::OutputTensor> ComputeJacobian(net, X)
    /** Runs `Batch × OutputTensor::Size` backward passes at the current weights; returns `Net::TotalParamCount` output-shape tensors, in `all_params()` flat order. Averages over `Batch` samples. Zeroes `Param::grad` on return; does not touch `m`, `v`, or `Param::metrics`. Snapshot quantity — call at checkpoints or (for small `Net`) every step. */
    template<typename Net, size_t Batch>
        requires IsTrainableNetwork<Net>
    std::vector<typename Net::OutputTensor> ComputeJacobian(
        Net &net,
        const typename PrependBatch<Batch, typename Net::InputTensor>::type &X)
    {
        using OutT               = typename Net::OutputTensor;
        constexpr size_t P       = Net::TotalParamCount;
        constexpr size_t OutSize = OutT::Size;

        std::vector<OutT> J(P);
        for (auto &t : J) t.fill(0.f);

        for (size_t b = 0; b < Batch; ++b) {
            typename PrependBatch<1, typename Net::InputTensor>::type Xb;
            for (size_t k = 0; k < Net::InputTensor::Size; ++k)
                Xb.flat(k) = X.flat(b * Net::InputTensor::Size + k);

            typename Net::template TrainingCache<1> cache;
            net.template ForwardAll<1>(Xb, cache);

            for (size_t j = 0; j < OutSize; ++j) {
                net.ZeroGrad();
                typename PrependBatch<1, OutT>::type ej{};
                ej.flat(j) = 1.f;
                net.template BackwardRange<1, 0, Net::NumBlocks>(cache, ej);

                size_t i = 0;
                std::apply([&](const auto &... ps) {
                    ([&] {
                        for (size_t k = 0; k < ps.Size; ++k)
                            J[i++].flat(j) += ps.grad.flat(k);
                    }(), ...);
                }, net.all_params());
            }
        }

        if constexpr (Batch > 1) {
            const float inv = 1.f / static_cast<float>(Batch);
            for (auto &t : J)
                for (size_t j = 0; j < OutSize; ++j) t.flat(j) *= inv;
        }

        net.ZeroGrad();
        return J;
    }


    // @doc: template<typename Net> void SnapshotFlatValues(net, out)
    /** Copies flat concatenation of every `Param::value` into `out` (resized to `Net::TotalParamCount`). Pair with `ComputeFlatDelta` to capture per-step `Δθ` across an Update. */
    template<typename Net>
        requires IsTrainableNetwork<Net>
    void SnapshotFlatValues(const Net &net, std::vector<float> &out) {
        out.resize(Net::TotalParamCount);
        size_t i = 0;
        std::apply([&](const auto &... ps) {
            ([&] {
                for (size_t k = 0; k < ps.Size; ++k) out[i++] = ps.value.flat(k);
            }(), ...);
        }, net.all_params());
    }

    // @doc: template<typename Net> void ComputeFlatDelta(net, before, delta_out)
    /** After Update: writes `Δθ_i = θ_after − before[i]` into `delta_out` (resized to `Net::TotalParamCount`). */
    template<typename Net>
        requires IsTrainableNetwork<Net>
    void ComputeFlatDelta(const Net &net, const std::vector<float> &before, std::vector<float> &delta_out) {
        delta_out.resize(Net::TotalParamCount);
        size_t i = 0;
        std::apply([&](const auto &... ps) {
            ([&] {
                for (size_t k = 0; k < ps.Size; ++k) {
                    delta_out[i] = ps.value.flat(k) - before[i];
                    ++i;
                }
            }(), ...);
        }, net.all_params());
    }

    // @doc: template<typename Net> void SnapshotFlatGrads(net, out)
    /** Copies flat concatenation of every `Param::grad` into `out` (resized to `Net::TotalParamCount`). Call between `Backward` and `Update` to capture the step's gradient. */
    template<typename Net>
        requires IsTrainableNetwork<Net>
    void SnapshotFlatGrads(const Net &net, std::vector<float> &out) {
        out.resize(Net::TotalParamCount);
        size_t i = 0;
        std::apply([&](const auto &... ps) {
            ([&] {
                for (size_t k = 0; k < ps.Size; ++k) out[i++] = ps.grad.flat(k);
            }(), ...);
        }, net.all_params());
    }


    // @doc: template<typename Net> class OutputSpaceTrajectory
    /** Owns two per-parameter and two network-level output-shape accumulators (GROSS and NET). Optional per-parameter storage (~`P × OutSize` floats) via constructor flag. Everything else is derived from these — scalar L2 norms, per-output-dim accord, dumps/loads to a single binary stream. */
    template<typename Net>
        requires IsTrainableNetwork<Net>
    class OutputSpaceTrajectory {
    public:
        using OutT = typename Net::OutputTensor;
        static constexpr size_t P = Net::TotalParamCount;

    private:
        std::vector<OutT> gross_per_param_; // Σ_t |Δθ_i(t)| · |J_i(t)|  elementwise, one OutT per param
        std::vector<OutT> net_per_param_;   // Σ_t Δθ_i(t) · J_i(t)      elementwise, one OutT per param
        OutT gross_net_{};                  // Σ_t Σ_i |Δθ_i · J_ij|     across params — trivial storage
        OutT net_net_{};                    // Σ_t Σ_i Δθ_i · J_ij       across params — trivial storage
        bool per_param_enabled_;
        size_t step_count_ = 0;

    public:
        explicit OutputSpaceTrajectory(const bool per_param = true)
            : per_param_enabled_(per_param) {
            gross_net_.fill(0.f);
            net_net_.fill(0.f);
            if (per_param_enabled_) {
                gross_per_param_.resize(P);
                net_per_param_.resize(P);
                for (auto &t : gross_per_param_) t.fill(0.f);
                for (auto &t : net_per_param_) t.fill(0.f);
            }
        }

        // @doc: struct OutputSpaceTrajectory::StepContribution
        /** One step's network-level output-space movement: `net_j = Σ_i Δθ_i · J_ij` (signed), `gross_j = Σ_i |Δθ_i · J_ij|`. Returned by `Accumulate` so callers can maintain instantaneous and rolling-window accords on top of the cumulative ones. `AccordL2()` on a single step measures pure cross-parameter agreement (no time dimension). */
        struct StepContribution {
            OutT net{};
            OutT gross{};

            float NetL2() const {
                float s = 0.f;
                for (size_t j = 0; j < OutT::Size; ++j) s += net.flat(j) * net.flat(j);
                return std::sqrt(s);
            }
            float GrossL2() const {
                float s = 0.f;
                for (size_t j = 0; j < OutT::Size; ++j) s += gross.flat(j) * gross.flat(j);
                return std::sqrt(s);
            }
            float AccordL2() const {
                const float g = GrossL2();
                return g > 0.f ? NetL2() / g : 0.f;
            }
        };

        // @doc: StepContribution OutputSpaceTrajectory::Accumulate(const std::vector<float> &delta_theta, const std::vector<OutT> &J)
        /** Adds one training step's contribution: `net += Δθ_i · J_i`, `gross += |Δθ_i · J_i|` elementwise per output dim, both per-param and aggregated across params. Returns the step's own network-level contribution for instantaneous / windowed accords. */
        StepContribution Accumulate(const std::vector<float> &delta_theta,
                                    const std::vector<OutT> &J) {
            StepContribution step{};
            step.net.fill(0.f);
            step.gross.fill(0.f);
            for (size_t i = 0; i < P; ++i) {
                const float dth  = delta_theta[i];
                const float adth = std::abs(dth);
                const OutT &Ji   = J[i];
                for (size_t j = 0; j < OutT::Size; ++j) {
                    const float g = Ji.flat(j);
                    const float contrib_signed = dth * g;
                    const float contrib_gross  = adth * std::abs(g);
                    if (per_param_enabled_) {
                        net_per_param_[i].flat(j)   += contrib_signed;
                        gross_per_param_[i].flat(j) += contrib_gross;
                    }
                    step.net.flat(j)   += contrib_signed;
                    step.gross.flat(j) += contrib_gross;
                }
            }
            for (size_t j = 0; j < OutT::Size; ++j) {
                net_net_.flat(j)   += step.net.flat(j);
                gross_net_.flat(j) += step.gross.flat(j);
            }
            ++step_count_;
            return step;
        }

        // @doc: template<size_t W> class OutputSpaceTrajectory::RollingAccord
        /** Fixed-window rolling accord over the last `W` steps' `StepContribution`s. Ring-buffered; `add` is O(OutT::Size). `accord()` = ‖Σ_window net‖₂ / ‖Σ_window gross‖₂ — cross-parameter *and* cross-time coherence within the window. Not serialized — warms up over the first `W` steps after any (re)start. */
        template<size_t W>
        class RollingAccord {
            std::array<StepContribution, W> ring_{};
            OutT net_sum_{};
            OutT gross_sum_{};
            size_t count_ = 0;

        public:
            RollingAccord() {
                net_sum_.fill(0.f);
                gross_sum_.fill(0.f);
            }

            void add(const StepContribution &s) {
                const size_t slot = count_ % W;
                if (count_ >= W) {
                    const StepContribution &old = ring_[slot];
                    for (size_t j = 0; j < OutT::Size; ++j) {
                        net_sum_.flat(j)   -= old.net.flat(j);
                        gross_sum_.flat(j) -= old.gross.flat(j);
                    }
                }
                ring_[slot] = s;
                for (size_t j = 0; j < OutT::Size; ++j) {
                    net_sum_.flat(j)   += s.net.flat(j);
                    gross_sum_.flat(j) += s.gross.flat(j);
                }
                ++count_;
            }

            float accord() const {
                float ns = 0.f, gs = 0.f;
                for (size_t j = 0; j < OutT::Size; ++j) {
                    ns += net_sum_.flat(j)   * net_sum_.flat(j);
                    gs += gross_sum_.flat(j) * gross_sum_.flat(j);
                }
                return gs > 0.f ? std::sqrt(ns) / std::sqrt(gs) : 0.f;
            }

            size_t count() const { return count_; }
        };

        // @doc: void OutputSpaceTrajectory::Reset()
        /** Zeroes all accumulators. Call at phase boundaries. */
        void Reset() {
            gross_net_.fill(0.f);
            net_net_.fill(0.f);
            if (per_param_enabled_) {
                for (auto &t : gross_per_param_) t.fill(0.f);
                for (auto &t : net_per_param_) t.fill(0.f);
            }
            step_count_ = 0;
        }

        // Network-level accessors
        const OutT &NetworkGrossOutput() const { return gross_net_; }
        const OutT &NetworkNetOutput()   const { return net_net_; }

        float NetworkGrossL2() const {
            float s = 0.f;
            for (size_t j = 0; j < OutT::Size; ++j)
                s += gross_net_.flat(j) * gross_net_.flat(j);
            return std::sqrt(s);
        }

        float NetworkNetL2() const {
            float s = 0.f;
            for (size_t j = 0; j < OutT::Size; ++j)
                s += net_net_.flat(j) * net_net_.flat(j);
            return std::sqrt(s);
        }

        float NetworkGrossL1() const {
            float s = 0.f;
            for (size_t j = 0; j < OutT::Size; ++j) s += gross_net_.flat(j);
            return s;
        }

        // @doc: float OutputSpaceTrajectory::AccordRatioL2() const
        /** `‖NetworkNetOutput‖_2 / ‖NetworkGrossOutput‖_2 ∈ [0, 1]`. Cross-parameter/time collaboration signal in output space — `= 1` iff every per-step per-param output contribution pointed the same way; `< 1` measures destructive interference. */
        float AccordRatioL2() const {
            const float g = NetworkGrossL2();
            return g > 0.f ? NetworkNetL2() / g : 0.f;
        }

        // @doc: OutT OutputSpaceTrajectory::PerDimAccord() const
        /** Per-output-dim accord tensor: entry `j` is `|NET_j| / GROSS_j`. Reveals which output directions received collaborative updates versus cancellation. */
        OutT PerDimAccord() const {
            OutT r{};
            for (size_t j = 0; j < OutT::Size; ++j) {
                const float g = gross_net_.flat(j);
                r.flat(j) = g > 1e-30f ? std::abs(net_net_.flat(j)) / g : 0.f;
            }
            return r;
        }

        const std::vector<OutT> &PerParamGross() const { return gross_per_param_; }
        const std::vector<OutT> &PerParamNet()   const { return net_per_param_; }

        bool   per_param_enabled() const { return per_param_enabled_; }
        size_t step_count()        const { return step_count_; }

        // @doc: void OutputSpaceTrajectory::SaveTo(std::ofstream &f) const
        /** Serialises step count, per-param flag, network-level tensors, then per-param tensors if enabled. */
        void SaveTo(std::ofstream &f) const {
            f.write(reinterpret_cast<const char *>(&step_count_), sizeof(step_count_));
            const std::uint8_t pp = per_param_enabled_ ? 1u : 0u;
            f.write(reinterpret_cast<const char *>(&pp), sizeof(pp));
            gross_net_.Save(f);
            net_net_.Save(f);
            if (per_param_enabled_) {
                for (const auto &t : gross_per_param_) t.Save(f);
                for (const auto &t : net_per_param_)   t.Save(f);
            }
        }

        // @doc: void OutputSpaceTrajectory::LoadFrom(std::ifstream &f)
        /** Reads back the format written by `SaveTo`. */
        void LoadFrom(std::ifstream &f) {
            f.read(reinterpret_cast<char *>(&step_count_), sizeof(step_count_));
            std::uint8_t pp = 0;
            f.read(reinterpret_cast<char *>(&pp), sizeof(pp));
            per_param_enabled_ = (pp != 0);
            gross_net_.Load(f);
            net_net_.Load(f);
            if (per_param_enabled_) {
                if (gross_per_param_.size() != P) gross_per_param_.resize(P);
                if (net_per_param_.size()   != P) net_per_param_.resize(P);
                for (auto &t : gross_per_param_) t.Load(f);
                for (auto &t : net_per_param_)   t.Load(f);
            }
        }
    };

} // namespace TTTN
