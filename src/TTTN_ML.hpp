#pragma once
#include <random>
#include "TensorContract.hpp"
#include "TensorReduce.hpp"

namespace TTTN {
    // @doc: static constexpr float EPS
    /** Error constant used throughout ML file to allow divisions by `0` */
    static constexpr float EPS = 1e-8f;

    // @doc: struct ReLU
    /**
     * `ActivationOp` for ***Rectified Linear Unit*** (`ReLU`)
     * `operator()` -> `[0, infinity)`
     * `prime` -> `1.0f || 0.0f`
     */
    struct ReLU {
        constexpr float operator()(const float x) const { return x > 0.f ? x : 0.f; }
        static constexpr float prime(const float a) { return a > 0.f ? 1.f : 0.f; }
    };

    // @doc: Sigmoid
    /**
     * `ActivationOp` for `Sigmoid`
     * `operator()` -> `[0, 1.0f]`
     * `prime` -> `(0.0f, 0.25f]`
     */
    struct Sigmoid {
        constexpr float operator()(const float x) const { return 1.f / (1.f + std::exp(-x)); }
        static constexpr float prime(const float a) { return a * (1.f - a); }
    };

    // @doc: Tanh
    /**
     * `ActivationOp` for ***Hyperbolic Tangent*** (`Tanh`)
     * `operator()` -> `[-1.0f, 1.0f]`
     * `prime` -> `(0.0f, 1.0f]`
     */
    struct Tanh {
        constexpr float operator()(const float x) const { return std::tanh(x); }
        static constexpr float prime(const float a) { return 1.f - a * a; }
    };

    // @doc: Linear
    /**
     * `ActivationOp` for `Linear` (no activation)
     * `operator()` -> `(-infinity, infinity)`
     * `prime` -> `(-infinity, infinity)`
     */
    struct Linear {
        constexpr float operator()(const float x) const { return x; }
        static constexpr float prime(float) { return 1.f; }
    };

    // @doc: template<typename T> concept ActivationOp = FloatUnaryOp<T> && requires(float a)
    /**
     * `concept` requiring:
     * `constexpr float operator()(float x)`
     * `constexpr float prime(float a)`
     */
    template<typename T> concept ActivationOp = FloatUnaryOp<T> && requires(float a)
    {
        { T::prime(a) } -> std::convertible_to<float>;
    };


    // @doc: template<size_t N> float CrossEntropyLoss(const Tensor<N> &output, const Tensor<N> &target)
    /**
     * Computes ***Cross Entropy*** between two `Tensor`s, `output` and `target`, and returns `float`
     * Calls `Collapse<Mul, Add>(target, Map<Compose<Log, Clamp<EPS>>>(output)) * -1.f`
     */
    template<size_t N>
    float CrossEntropyLoss(const Tensor<N> &output, const Tensor<N> &target) {
        return Collapse<Mul, Add>(target, Map<Compose<Log, Clamp<EPS> > >(output)) * -1.f;
    }


    // @doc: template<size_t... Dims> void XavierInitMD(Tensor<Dims...> &W, const size_t fan_in, const size_t fan_out)
    /** ***Xavier Initializes*** a `Tensor` inplace, given `fan_in` and `fan_out` values denoting net size of input and output to a neural network layer */
    template<size_t... Dims>
    void XavierInitMD(Tensor<Dims...> &W, const size_t fan_in, const size_t fan_out) {
        static std::mt19937 rng{std::random_device{}()};
        const float limit = std::sqrt(6.f / static_cast<float>(fan_in + fan_out));
        std::uniform_real_distribution<float> dist{-limit, limit};
        for (size_t i = 0; i < Tensor<Dims...>::Size; ++i)
            W.flat(i) = dist(rng);
    }


    // @doc: template<size_t Axis, float Temp = 1.f, size_t... Dims> Tensor<Dims...> Softmax(const Tensor<Dims...> &x)
    /**
     * Given an `Axis` on which to normalize, perform `Softmax` normalization with optional temperature `Temp` (default `1.f`)
     * When `Temp != 1`, input is scaled by `1/Temp` before the standard softmax: `softmax(x / Temp)`
     * Elegantly calls `BroadcastReduceMove<Axis, Div, Add>(BroadcastReduce<Axis, Compose<Exp, Sub>, Max>(scaled))` to first map to `a = e^(x - max)` and then to `b = a / sum(a)`
     * Shape-preserving
     */
    template<size_t Axis, float Temp = 1.f, size_t... Dims>
    Tensor<Dims...> Softmax(const Tensor<Dims...> &x) {
        if constexpr (Temp == 1.f) {
            return BroadcastReduceMove<Axis, Div, Add>(BroadcastReduce<Axis, Compose<Exp, Sub>, Max>(x));
        } else {
            return BroadcastReduceMove<Axis, Div, Add>(BroadcastReduce<Axis, Compose<Exp, Sub>, Max>(x * (1.f / Temp)));
        }
    }

    // @doc: template<size_t Axis, float Temp = 1.f, size_t... Dims> Tensor<Dims...> SoftmaxPrime(const Tensor<Dims...> &grad, const Tensor<Dims...> &a)
    /**
     * Computes derivative of `Softmax` with optional temperature `Temp` (default `1.f`)
     * By the chain rule, `d softmax(x/T)/dx = (1/T) * J_softmax(x/T)`, so the result is scaled by `1/Temp`
     * Calls (efficient equivalent of) `a * BroadcastMap<Axis, Sub>(grad, BroadcastReduce<Axis, Add, Mul>(a, grad))`
     * Generalization of `a * (g - (g . a))`, scaled by `1/Temp`
     * Shape-preserving
     */
    template<size_t Axis, float Temp = 1.f, size_t... Dims>
    Tensor<Dims...> SoftmaxPrime(const Tensor<Dims...> &grad, const Tensor<Dims...> &a) {
        // sum of elementwise products along Axis
        auto axisDots = Reduce<Axis, Add>(a * grad);
        // copy grad in
        auto g = grad;
        // subtract dot inplace
        BroadcastApply<Axis, Sub>(g, axisDots);
        // multiply by activation inplace
        g *= a;
        if constexpr (Temp != 1.f) g *= (1.f / Temp);
        return g;
    }


    template<size_t Axis, float Temp, typename TensorT>
    class SoftmaxBlock;

    // @doc: template<size_t Axis, float Temp, size_t... Dims> class SoftmaxBlock<Axis, Temp, Tensor<Dims...> >
    /** Class representing the concrete block of a `Softmax` layer in a `TrainableTensorNetwork`, satisfying the `Block` `concept`; `Temp` is the softmax temperature (default `1.f` via `SoftmaxLayer`) */
    template<size_t Axis, float Temp, size_t... Dims>
    class SoftmaxBlock<Axis, Temp, Tensor<Dims...> > {
    public:
        using InputTensor = Tensor<Dims...>;
        using OutputTensor = Tensor<Dims...>;

        // @doc: auto all_params()
        /** Returns `std::tuple<>{}` (no parameters) */
        auto all_params() { return std::tuple<>{}; }

        // @doc: auto all_params()
        /** Returns `std::tuple<>{}` (no parameters) */
        auto all_params() const { return std::tuple<>{}; }

        // @doc: OutputTensor SoftmaxBlock::Forward(const InputTensor &x) const
        /** Calls `Softmax<Axis, Temp>(x)` */
        OutputTensor Forward(const InputTensor &x) const {
            return Softmax<Axis, Temp>(x);
        }

        // @doc: InputTensor SoftmaxBlock::Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor & /*a_prev*/)
        /** Calls `SoftmaxPrime<Axis, Temp>(delta_A, a)` */
        InputTensor Backward(const OutputTensor &delta_A, const OutputTensor &a, const InputTensor & /*a_prev*/) {
            return SoftmaxPrime<Axis, Temp>(delta_A, a);
        }

        // @doc: template<size_t Batch> Tensor<Batch, Dims...> SoftmaxBlock::BatchedForward(const Tensor<Batch, Dims...> &X) const
        /**
         * Calls `Softmax<Axis + 1, Temp>(X)`
         * NOTE: assumes first axis is `Batch` axis
         */
        template<size_t Batch>
        Tensor<Batch, Dims...> BatchedForward(const Tensor<Batch, Dims...> &X) const {
            return Softmax<Axis + 1, Temp>(X);
        }

        // @doc: template<size_t Batch> Tensor<Batch, Dims...> SoftmaxBlock::BatchedBackward(const Tensor<Batch, Dims...> &delta_A, const Tensor<Batch, Dims...> &a, const Tensor<Batch, Dims...> & /*a_prev*/)
        /**
         * Calls `SoftmaxPrime<Axis + 1, Temp>(delta_A, a)`
         * NOTE: assumes first axis is `Batch` axis
         */
        template<size_t Batch>
        Tensor<Batch, Dims...> BatchedBackward(const Tensor<Batch, Dims...> &delta_A, const Tensor<Batch, Dims...> &a,
                                               const Tensor<Batch, Dims...> & /*a_prev*/) {
            return SoftmaxPrime<Axis + 1, Temp>(delta_A, a);
        }
    };


    // @doc: template<size_t Axis, float Temp = 1.f> struct SoftmaxLayer
    /**
     * `BlockRecipe`-compliant recipe struct to create `Block SoftmaxBlock`
     * Pass in `Axis` of normalization and optional `Temp` (softmax temperature, default `1.f`); tensor shape is deduced from `InputT`
     */
    template<size_t Axis, float Temp = 1.f>
    struct SoftmaxLayer {
        using OutputTensor = Tensor<1>; // this is just for the concepts in TTN...InputT == OutputT
        template<typename InputT>
        using Resolve = SoftmaxBlock<Axis, Temp, InputT>;
    };


    // @doc: template<typename L, typename TensorT> concept LossFunction
    /**
     * `concept` to define `LossFunction` structs
     * Requires:
     * `Loss(Tensor<Dims...>, Tensor<Dims...>) -> Tensor<>` (rank-0 scalar; implicitly converts to `float`)
     * `Grad(Tensor<Dims...>, Tensor<Dims...>) -> Tensor<Dims...>`
     */
    template<typename L, typename TensorT> concept LossFunction =
            IsTensor<TensorT> &&
            requires
            {
                {
                    L::Loss(std::declval<const TensorT &>(), std::declval<const TensorT &>())
                } -> std::same_as<Tensor<> >;
                { L::Grad(std::declval<const TensorT &>(), std::declval<const TensorT &>()) } -> std::same_as<TensorT>;
            };


    // @doc: struct MSE
    /** `LossFunction` struct for ***Mean Squared Error*** (`MSE`) */
    struct MSE {
        // @doc: template<size_t... Dims> static Tensor<> MSE::Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)
        /**
         * Sum of squares of difference between `target` and `pred`
         * Calls `Collapse<Compose<Sq, Sub>, Add>(pred, target) * Inv`
         */
        template<size_t... Dims>
        static Tensor<> Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            static constexpr float Inv = 1.f / Tensor<Dims...>::Size;
            Tensor<> s;
            s.flat(0) = Collapse<Compose<Sq, Sub>, Add>(pred, target) * Inv;
            return s;
        }

        // @doc: template<size_t... Dims> static Tensor<Dims...> MSE::Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)
        /** Derivative of `MSE` loss - `2(pred - target) / Tensor<Dims...>::Size` (standard power rule derivative, scaled by how many elements composed the original sum) */
        template<size_t... Dims>
        static Tensor<Dims...> Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            constexpr float inv = 2.f / static_cast<float>(Tensor<Dims...>::Size);
            return (pred - target) * inv;
        }
    };


    // @doc: struct BinaryCEL
    /**
     * `LossFunction` struct for ***Binary Cross Entropy Loss*** (`BinaryCEL`)
     * Helper for binary cases, but is just a specialization of `struct CEL`
     */
    struct BinaryCEL {
        // @doc: template<size_t... Dims> static Tensor<> BinaryCEL::Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)
        /** `-log(pred[true])` (negative log of the predicted value for `true` answer, whose target value is `1.0f`) */
        template<size_t... Dims>
        static Tensor<> Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            const auto p_c = Map<Clamp<EPS, 1.f - EPS> >(pred);
            Tensor<> s;
            s.flat(0) = -(Collapse<Mul, Add>(target, Map<Log>(p_c)) +
                          Collapse<Mul, Add>(Map<OneMinus>(target), Map<Compose<Log, OneMinus> >(p_c)));
            return s;
        }

        // @doc: template<size_t... Dims> static Tensor<Dims...> BinaryCEL::Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)
        /** `(p - t) / (p * (1.f - p) + EPS)` */
        template<size_t... Dims>
        static Tensor<Dims...> Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            return pred.zip(target, [](float p, float t) {
                return (p - t) / (p * (1.f - p) + EPS);
            });
        }
    };


    // @doc: struct CEL
    /** `LossFunction` struct for ***Cross Entropy Loss*** (`CEL`) */
    struct CEL {
        // @doc: template<size_t... Dims> static Tensor<> CEL::Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)
        /**
         * `-log(pred[true])` (negative log of the predicted value for `true` answer, whose target value is `1.0f`)
         * Elegantly calls `Collapse<Mul, Add>(target, Map<Compose<Log, Clamp<EPS> > >(pred)) * -1.f`
         */
        template<size_t... Dims>
        static Tensor<> Loss(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            Tensor<> s;
            s.flat(0) = Collapse<Mul, Add>(target, Map<Compose<Log, Clamp<EPS> > >(pred)) * -1.f;
            return s;
        }

        // @doc: template<size_t... Dims> static Tensor<Dims...> CEL::Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target)
        /** Elegantly calls `Zip<Compose<Neg, Div> >(target, Map<Clamp<EPS> >(pred))` */
        template<size_t... Dims>
        static Tensor<Dims...> Grad(const Tensor<Dims...> &pred, const Tensor<Dims...> &target) {
            return Zip<Compose<Neg, Div> >(target, Map<Clamp<EPS> >(pred));
        }
    };


    template<size_t PadId>
    struct SequenceCEL {
        template<size_t SeqLen, size_t Vocab>
        static Tensor<> Loss(const Tensor<SeqLen, Vocab> &pred, const Tensor<SeqLen, Vocab> &target) {
            static_assert(PadId < Vocab, "PadId out of Vocab range");
            // mask[t] = 1 - target[t, PadId]; 1 at non-PAD positions, 0 at PAD positions
            Tensor<SeqLen> mask;
            for (size_t t = 0; t < SeqLen; ++t) mask.flat(t) = 1.f - target.flat(t * Vocab + PadId);
            const float n_nonpad = std::max(Reduce<0, Add>(mask).flat(0), 1.f);
            // per-position CE: -sum_v target[t,v] * log(clamp(pred[t,v]))  ->  Tensor<SeqLen>
            const auto logp = Map<Compose<Log, Clamp<EPS> > >(pred);
            const auto per_pos = Reduce<1, Add>(target * logp) * -1.f;
            Tensor<> s;
            s.flat(0) = Reduce<0, Add>(per_pos * mask) / n_nonpad;
            return s;
        }

        template<size_t SeqLen, size_t Vocab>
        static Tensor<SeqLen, Vocab> Grad(const Tensor<SeqLen, Vocab> &pred, const Tensor<SeqLen, Vocab> &target) {
            static_assert(PadId < Vocab, "PadId out of Vocab range");
            Tensor<SeqLen> mask;
            for (size_t t = 0; t < SeqLen; ++t) mask.flat(t) = 1.f - target.flat(t * Vocab + PadId);
            const float n_nonpad = std::max(Reduce<0, Add>(mask).flat(0), 1.f);
            // base grad: -target / clamp(pred), normalized by non-PAD count
            auto g = Zip<Compose<Neg, Div> >(target, Map<Clamp<EPS> >(pred));
            g *= (1.f / n_nonpad);
            // zero out PAD positions: broadcast mask along Vocab axis (axis 1)
            BroadcastApply<1, Mul>(g, mask);
            return g;
        }
    };


    // Sequence loss for networks whose output is raw logits [SeqLen, Vocab].
    // Applies Softmax<1> per row internally, then token-wise CEL with PAD masking.
    // Grad uses the combined softmax+CEL identity: (probs - target) / n_nonpad.
    // Satisfies LossFunction<SequenceSoftmaxCEL<PadId>, Tensor<SeqLen,Vocab>>.
    // Use with TrainableTensorNetwork::Fit when the last block outputs logits.
    template<size_t PadId = 0>
    struct SequenceSoftmaxCEL {
        template<size_t SeqLen, size_t Vocab>
        static Tensor<> Loss(const Tensor<SeqLen, Vocab> &logits, const Tensor<SeqLen, Vocab> &target) {
            static_assert(PadId < Vocab, "PadId out of Vocab range");
            return SequenceCEL<PadId>::Loss(Softmax<1>(logits), target);
        }

        template<size_t SeqLen, size_t Vocab>
        static Tensor<SeqLen, Vocab> Grad(const Tensor<SeqLen, Vocab> &logits, const Tensor<SeqLen, Vocab> &target) {
            static_assert(PadId < Vocab, "PadId out of Vocab range");
            const auto probs = Softmax<1>(logits);
            Tensor<SeqLen> mask;
            for (size_t t = 0; t < SeqLen; ++t) mask.flat(t) = 1.f - target.flat(t * Vocab + PadId);
            const float n_nonpad = std::max(Reduce<0, Add>(mask).flat(0), 1.f);
            auto grad = (probs - target) * (1.f / n_nonpad);
            BroadcastApply<1, Mul>(grad, mask);
            return grad;
        }
    };


    // General token-wise averaged loss wrapping any LossFunction.
    // Applies L per-token position, averages over non-PAD positions.
    // PAD detection: target[t, PadId] == 1 (one-hot convention).
    // Satisfies LossFunction<TokenWiseLoss<L,PadId>, Tensor<SeqLen,Vocab>>.
    template<typename L, size_t PadId = 0>
    struct TokenWiseLoss {
        template<size_t SeqLen, size_t Vocab>
            requires LossFunction<L, Tensor<Vocab> >
        static Tensor<> Loss(const Tensor<SeqLen, Vocab> &pred, const Tensor<SeqLen, Vocab> &target) {
            float n_nonpad = 0.f;
            for (size_t t = 0; t < SeqLen; ++t)
                n_nonpad += 1.f - target.flat(t * Vocab + PadId);
            n_nonpad = std::max(n_nonpad, 1.f);

            float total = 0.f;
            for (size_t t = 0; t < SeqLen; ++t) {
                if (target.flat(t * Vocab + PadId) > 0.5f) continue;
                Tensor<Vocab> pred_t, tgt_t;
                for (size_t v = 0; v < Vocab; ++v) {
                    pred_t.flat(v) = pred.flat(t * Vocab + v);
                    tgt_t.flat(v) = target.flat(t * Vocab + v);
                }
                total += L::Loss(pred_t, tgt_t).flat(0);
            }
            Tensor<> s;
            s.flat(0) = total / n_nonpad;
            return s;
        }

        template<size_t SeqLen, size_t Vocab>
            requires LossFunction<L, Tensor<Vocab> >
        static Tensor<SeqLen, Vocab> Grad(const Tensor<SeqLen, Vocab> &pred, const Tensor<SeqLen, Vocab> &target) {
            float n_nonpad = 0.f;
            for (size_t t = 0; t < SeqLen; ++t)
                n_nonpad += 1.f - target.flat(t * Vocab + PadId);
            n_nonpad = std::max(n_nonpad, 1.f);

            Tensor<SeqLen, Vocab> grad;
            grad.fill(0.f);
            for (size_t t = 0; t < SeqLen; ++t) {
                if (target.flat(t * Vocab + PadId) > 0.5f) continue;
                Tensor<Vocab> pred_t, tgt_t;
                for (size_t v = 0; v < Vocab; ++v) {
                    pred_t.flat(v) = pred.flat(t * Vocab + v);
                    tgt_t.flat(v) = target.flat(t * Vocab + v);
                }
                const auto g_t = L::Grad(pred_t, tgt_t);
                for (size_t v = 0; v < Vocab; ++v)
                    grad.flat(t * Vocab + v) = g_t.flat(v) / n_nonpad;
            }
            return grad;
        }
    };


    // @doc: template<size_t Batch, size_t N> float OneHotAccuracy(const Tensor<Batch, N> &pred, const Tensor<Batch, N> &labels)
    /**
     * Computes accuracy for `Tensor`s organized in a batched `Tensor`
     * Takes any `Tensor<Batch, Dims...>` and flattens `Dims...` internally
     * NOTE: assumes ***one-hot encoding*** for labels
     */
    template<size_t Batch, size_t... Dims>
    float OneHotAccuracy(const Tensor<Batch, Dims...> &pred, const Tensor<Batch, Dims...> &labels) {
        constexpr size_t InnerSize = (Dims * ... * 1);
        // flatten to become [Batch x FlatPredVector]
        const auto p = Reshape<Batch, InnerSize>(pred);
        //                   [Batch x OneHotVector]
        const auto l = Reshape<Batch, InnerSize>(labels);
        // collect predicted decimal @ target index
        const auto p_correct = Reduce<1, Add>(p * l);
        // collect max prediction decimal
        const auto p_max = Reduce<1, Max>(p);
        // global sum of (p_correct - p_max) < EPS ? 1 : 0 gives total count of correct predictions
        const float n = Reduce<0, Add>(Map<Step<1e-5f> >(p_max - p_correct));
        // correct n out of total
        return 100.f * n / static_cast<float>(Batch);
    }

    // @doc: template<typename TupleT> float OneHotAccuracy(const ActivationsWrap<TupleT> &A, const std::tuple_element_t<std::tuple_size_v<TupleT> - 1, TupleT> &labels)
    /**
     * Overload that takes a full `ActivationsWrap` and automatically uses the final activation
     * Eliminates the need to manually index with `.get<N>()`
     */
    template<typename TupleT>
    float OneHotAccuracy(const ActivationsWrap<TupleT> &A,
                         const std::tuple_element_t<std::tuple_size_v<TupleT> - 1, TupleT> &labels) {
        constexpr size_t Last = std::tuple_size_v<TupleT> - 1;
        return OneHotAccuracy(A.template get<Last>(), labels);
    }


    // ── Inference utilities ───────────────────────────────────────────────────

    // Index of the maximum element in a 1D tensor.
    template<size_t N>
    size_t Argmax(const Tensor<N> &t) {
        size_t best = 0;
        for (size_t i = 1; i < N; ++i)
            if (t.flat(i) > t.flat(best)) best = i;
        return best;
    }

    // Index of the maximum element in row `row` of a 2D tensor.
    template<size_t Rows, size_t Cols>
    size_t ArgmaxAt(const Tensor<Rows, Cols> &t, size_t row) {
        size_t best = 0;
        for (size_t v = 1; v < Cols; ++v)
            if (t(row, v) > t(row, best)) best = v;
        return best;
    }

    // {argmax, sum_exp} for a 1D tensor — numerically stable softmax decomposition.
    // Any softmax prob: exp(t[i] - t[argmax]) / sum_exp.
    template<size_t N>
    std::pair<size_t, float> SoftmaxStats(const Tensor<N> &t) {
        const size_t best = Argmax(t);
        const float max_val = t.flat(best);
        float sum_exp = 0.f;
        for (size_t v = 0; v < N; ++v)
            sum_exp += std::exp(t.flat(v) - max_val);
        return {best, sum_exp};
    }

    // {argmax, sum_exp} for row `row` of a 2D tensor.
    template<size_t Rows, size_t Cols>
    std::pair<size_t, float> SoftmaxStatsAt(const Tensor<Rows, Cols> &t, size_t row) {
        const size_t best = ArgmaxAt(t, row);
        const float max_val = t(row, best);
        float sum_exp = 0.f;
        for (size_t v = 0; v < Cols; ++v)
            sum_exp += std::exp(t(row, v) - max_val);
        return {best, sum_exp};
    }
};
