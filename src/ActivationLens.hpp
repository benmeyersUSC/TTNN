#pragma once
#include <cmath>
#include <cstddef>
#include "TrainableTensorNetwork.hpp"

namespace TTTN {

    // @doc: inline constexpr size_t LensToOutput
    /** Sentinel for `TargetIdx` meaning "the network output" — resolved to `Net::NumBlocks` at instantiation. The default everywhere; pass an interior boundary instead for a residual→residual `J` in the paper's original form. */
    inline constexpr size_t LensToOutput = static_cast<size_t>(-1);

    template<typename T>
    struct StripBatchDim;

    // @doc: template<size_t B, size_t... Dims> struct StripBatchDim<Tensor<B, Dims...> >
    /** Removes the leading batch axis from a batched activation type: `Tensor<B, Dims...> → Tensor<Dims...>`. */
    template<size_t B, size_t... Dims>
    struct StripBatchDim<Tensor<B, Dims...> > {
        using type = Tensor<Dims...>;
    };

    template<typename A, typename B>
    struct JoinTensorDims;

    // @doc: template<size_t... As, size_t... Bs> struct JoinTensorDims<Tensor<As...>, Tensor<Bs...> >
    /** Concatenates two tensor types' axes into one: `(Tensor<As...>, Tensor<Bs...>) → Tensor<As..., Bs...>`. Builds the lens type as target-axes-then-activation-axes. */
    template<size_t... As, size_t... Bs>
    struct JoinTensorDims<Tensor<As...>, Tensor<Bs...> > {
        using type = Tensor<As..., Bs...>;
    };

    // @doc: template<typename Net, size_t I> using BoundaryActivation
    /** The *unbatched* activation type at boundary `I` of `Net`, deduced from the net's own `TrainingCache` activations tuple. `I = 0` is the input; `I = NumBlocks` is the output. */
    template<typename Net, size_t I>
    using BoundaryActivation = typename StripBatchDim<std::remove_cvref_t<
        std::tuple_element_t<I, decltype(std::declval<typename Net::template TrainingCache<1> &>().activations)>
    >>::type;

    // @doc: template<typename Net, size_t TargetIdx> inline constexpr size_t ResolvedLensTarget
    /** `TargetIdx` unless it is `LensToOutput`, in which case `Net::NumBlocks`. */
    template<typename Net, size_t TargetIdx>
    inline constexpr size_t ResolvedLensTarget = (TargetIdx == LensToOutput) ? Net::NumBlocks : TargetIdx;

    // @doc: template<typename Net, size_t BoundaryIdx, size_t TargetIdx = LensToOutput> using LensTensor
    /** The fully-typed lens: `Tensor<TargetDims..., ActDims...>`. Row `j` (over the leading target axes) is `∂target_j/∂h` in the activation's own shape. Sequence activations keep their per-position axes — position averaging is a downstream reduction, never baked in. */
    template<typename Net, size_t BoundaryIdx, size_t TargetIdx = LensToOutput>
    using LensTensor = typename JoinTensorDims<
        BoundaryActivation<Net, ResolvedLensTarget<Net, TargetIdx>>,
        BoundaryActivation<Net, BoundaryIdx>
    >::type;


    // @doc: template<size_t BoundaryIdx, size_t Batch, size_t TargetIdx = LensToOutput, typename Net> LensTensor<Net, BoundaryIdx, TargetIdx> FitActivationLens(net, X)
    /** Fits the lens at `BoundaryIdx` over one prompt batch: per sample, one forward with cache, then one `BackwardRange` per flat target dimension with a one-hot cotangent, capturing the returned interior gradient; averages over `Batch`. With `Batch = 1` this is the *per-context* (exact, unaveraged) lens. `Param::grad`s are written as a side effect and zeroed on return; `m`, `v`, and `Param::metrics` are untouched. Cost: `Batch × TargetSize` partial backward passes. `BoundaryIdx == TargetIdx` returns the identity. */
    template<size_t BoundaryIdx, size_t Batch, size_t TargetIdx = LensToOutput, typename Net>
        requires IsTrainableNetwork<Net>
    LensTensor<Net, BoundaryIdx, TargetIdx> FitActivationLens(
        Net &net,
        const typename PrependBatch<Batch, typename Net::InputTensor>::type &X)
    {
        constexpr size_t Tgt = ResolvedLensTarget<Net, TargetIdx>;
        static_assert(BoundaryIdx <= Tgt && Tgt <= Net::NumBlocks,
                      "FitActivationLens: need BoundaryIdx <= TargetIdx <= NumBlocks");

        using LensT = LensTensor<Net, BoundaryIdx, TargetIdx>;
        using TgtT  = BoundaryActivation<Net, Tgt>;
        using ActT  = BoundaryActivation<Net, BoundaryIdx>;
        constexpr size_t TgtSize = TgtT::Size;
        constexpr size_t ActSize = ActT::Size;

        LensT lens{};

        for (size_t b = 0; b < Batch; ++b) {
            typename PrependBatch<1, typename Net::InputTensor>::type Xb;
            for (size_t k = 0; k < Net::InputTensor::Size; ++k)
                Xb.flat(k) = X.flat(b * Net::InputTensor::Size + k);

            typename Net::template TrainingCache<1> cache;
            net.template ForwardAll<1>(Xb, cache);

            for (size_t j = 0; j < TgtSize; ++j) {
                typename PrependBatch<1, TgtT>::type ej{};
                ej.flat(j) = 1.f;
                const auto grad_at = net.template BackwardRange<1, BoundaryIdx, Tgt>(cache, ej);
                for (size_t k = 0; k < ActSize; ++k)
                    lens.flat(j * ActSize + k) += grad_at.flat(k);
            }
        }

        if constexpr (Batch > 1) {
            constexpr float inv = 1.f / static_cast<float>(Batch);
            for (size_t i = 0; i < LensT::Size; ++i) lens.flat(i) *= inv;
        }

        net.ZeroGrad();
        return lens;
    }


    // @doc: template<size_t... LensDims, size_t... ActDims> auto ApplyLens(const Tensor<LensDims...> &lens, const Tensor<ActDims...> &h)
    /** Contracts the lens's trailing activation axes against `h` (a `ΣΠ`), yielding the target-shaped **linearized output** — what the network would emit if everything downstream of the boundary were replaced by its fitted linear map. Rows are sensitivities; contracted with an actual `h` they become a prediction. Misses downstream bias constants (affine caveat): compare rankings, not raw values. */
    template<size_t... LensDims, size_t... ActDims>
    auto ApplyLens(const Tensor<LensDims...> &lens, const Tensor<ActDims...> &h) {
        static_assert(sizeof...(LensDims) > sizeof...(ActDims),
                      "ApplyLens: lens must carry target axes ahead of the activation axes");
        return SigmaPi<sizeof...(ActDims)>(lens, h);
    }

    // @doc: template<size_t TgtRank = 1, size_t... LensDims> auto LensVector(const Tensor<LensDims...> &lens, const size_t target_flat)
    /** Extracts one target row as an activation-shaped tensor: `∂target_t/∂h`. Dual-read as a functional (dot with any `h` gives that target's lens logit) and as a steering direction for activation-space interventions. `TgtRank` is how many leading axes index the target (1 for a vocab of logits). */
    template<size_t TgtRank = 1, size_t... LensDims>
    auto LensVector(const Tensor<LensDims...> &lens, const size_t target_flat) {
        static_assert(TgtRank < sizeof...(LensDims), "LensVector: TgtRank must leave activation axes");
        using ActT = typename SeqToTensor<typename SplitAt<TgtRank, LensDims...>::tail>::type;
        ActT v;
        for (size_t k = 0; k < ActT::Size; ++k)
            v.flat(k) = lens.flat(target_flat * ActT::Size + k);
        return v;
    }


    // @doc: template<typename Net, size_t BoundaryIdx, size_t TargetIdx = LensToOutput> class ActivationLensAccumulator
    /** The E-over-contexts estimator *and* its dispersion instrument. Feed it single contexts; it maintains the running mean lens plus per-row second moments, exposing how much the per-context Jacobians agree with their average — the accord-ratio idea applied across *contexts* instead of across parameters/time. Where the downstream map is linear, per-context lenses are identical and coherence is exactly 1. */
    template<typename Net, size_t BoundaryIdx, size_t TargetIdx = LensToOutput>
        requires IsTrainableNetwork<Net>
    class ActivationLensAccumulator {
    public:
        using LensT = LensTensor<Net, BoundaryIdx, TargetIdx>;
        using TgtT  = BoundaryActivation<Net, ResolvedLensTarget<Net, TargetIdx>>;
        using ActT  = BoundaryActivation<Net, BoundaryIdx>;
        static constexpr size_t TgtSize = TgtT::Size;
        static constexpr size_t ActSize = ActT::Size;

    private:
        LensT  sum_{};
        TgtT   row_sqnorm_sum_{};
        size_t count_ = 0;

    public:
        // @doc: LensT ActivationLensAccumulator::Add(Net &net, const typename PrependBatch<1, typename Net::InputTensor>::type &x)
        /** Fits the exact per-context lens for `x` (one `FitActivationLens<BoundaryIdx, 1>` call), folds it into the running sums, and returns it so callers can stream their own per-context analyses without a second fit. */
        LensT Add(Net &net, const typename PrependBatch<1, typename Net::InputTensor>::type &x) {
            LensT Lc = FitActivationLens<BoundaryIdx, 1, TargetIdx>(net, x);
            for (size_t j = 0; j < TgtSize; ++j) {
                float sq = 0.f;
                for (size_t k = 0; k < ActSize; ++k) {
                    const float g = Lc.flat(j * ActSize + k);
                    sum_.flat(j * ActSize + k) += g;
                    sq += g * g;
                }
                row_sqnorm_sum_.flat(j) += sq;
            }
            ++count_;
            return Lc;
        }

        // @doc: LensT ActivationLensAccumulator::Mean() const
        /** The mean lens over all contexts seen so far — the paper's `E[J]` estimator. */
        LensT Mean() const {
            LensT m = sum_;
            if (count_ > 0) {
                const float inv = 1.f / static_cast<float>(count_);
                for (size_t i = 0; i < LensT::Size; ++i) m.flat(i) *= inv;
            }
            return m;
        }

        // @doc: TgtT ActivationLensAccumulator::RowCoherence() const
        /** Per target row `j`: `‖E_c[row_j]‖ / sqrt(E_c[‖row_j‖²]) ∈ [0, 1]` — 1 iff row `j`'s per-context Jacobians are identical (variance decomposition: the ratio is `sqrt(1 − Var/E[‖·‖²])`). Which targets' sensitivities are context-uniform, which are context-specific. */
        TgtT RowCoherence() const {
            TgtT r{};
            if (count_ == 0) return r;
            const float n = static_cast<float>(count_);
            for (size_t j = 0; j < TgtSize; ++j) {
                float mean_sq = 0.f;
                for (size_t k = 0; k < ActSize; ++k) {
                    const float m = sum_.flat(j * ActSize + k) / n;
                    mean_sq += m * m;
                }
                const float e_sq = row_sqnorm_sum_.flat(j) / n;
                r.flat(j) = e_sq > 0.f ? std::sqrt(mean_sq / e_sq) : 0.f;
            }
            return r;
        }

        // @doc: float ActivationLensAccumulator::Coherence() const
        /** Whole-lens Frobenius version of `RowCoherence`: `‖E_c[L]‖_F / sqrt(E_c[‖L‖_F²])`. How honestly the mean lens speaks for every individual context. */
        float Coherence() const {
            if (count_ == 0) return 0.f;
            const float n = static_cast<float>(count_);
            float mean_sq = 0.f, e_sq = 0.f;
            for (size_t i = 0; i < LensT::Size; ++i) {
                const float m = sum_.flat(i) / n;
                mean_sq += m * m;
            }
            for (size_t j = 0; j < TgtSize; ++j) e_sq += row_sqnorm_sum_.flat(j);
            e_sq /= n;
            return e_sq > 0.f ? std::sqrt(mean_sq / e_sq) : 0.f;
        }

        // @doc: float ActivationLensAccumulator::Dispersion() const
        /** `1 − Coherence²`: the fraction of per-context Jacobian energy that is context-specific — exactly the linearization-infidelity of `E[J]`. 0 where downstream is linear; hypothesized to collapse at circuit formation. */
        float Dispersion() const {
            const float c = Coherence();
            return 1.f - c * c;
        }

        size_t count() const { return count_; }
    };

} // namespace TTTN
