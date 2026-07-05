#pragma once
#include <concepts>
#include <random>
#include <cmath>
#include <fstream>
#include <tuple>
#include "Tensor.hpp"
#include "Snapshot.hpp"
#include "TensorContract.hpp"  // provides SplitAt, PrependBatch, ConcatSeqs, Contract, AxisList, etc.


namespace TTTN {
    struct AdamState {
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float wd = 0.f;    // decoupled AdamW weight decay coefficient; 0 = plain Adam
        float mCorr = 1.f; // 1 / (1 - b1^t)
        float vCorr = 1.f; // 1 / (1 - b2^t)
        int t = 0;

        // @doc: void AdamState::step()
        /**
         * Increment `t`
         * Recompute `mCorr`, `vCorr`
         */
        void step() {
            ++t;
            mCorr = 1.f / (1.f - std::pow(beta1, static_cast<float>(t)));
            vCorr = 1.f / (1.f - std::pow(beta2, static_cast<float>(t)));
        }
    };


    // @doc: template<typename TensorT> struct TrajectoryMetrics
    template<typename TensorT>
    struct TrajectoryMetrics {
        TensorT gross_path{};       // running Σ|Δθ| per element
        TensorT net_displacement{}; // running ΣΔθ per element

        void reset() { gross_path.fill(0.f); net_displacement.fill(0.f); }

        void save(std::ofstream &f) const { gross_path.Save(f); net_displacement.Save(f); }
        void load(std::ifstream &f)       { gross_path.Load(f); net_displacement.Load(f); }
    };


    // @doc: template<typename TensorT> struct Param
    /** `struct` layer around a `Block` to abstract away management, Adam updates */
    template<typename TensorT>
    struct Param {
        TensorT value{};
        TensorT grad{};
        TensorT m{};
        TensorT v{};
        TrajectoryMetrics<TensorT> metrics{};

        static constexpr size_t Size = TensorT::Size;

        void zero_grad() { grad.fill(0.f); }

        // @doc: void Param::update(const AdamState &adam, float lr)
        /** For each `float` parameter in `value`, use Adam moments and gradient to update */
        void update(const AdamState &adam, float lr) {
            ParForEach(Size, [&](const size_t i) {
                const float before = value.flat(i);
                const float g = grad.flat(i);
                m.flat(i) = adam.beta1 * m.flat(i) + (1.f - adam.beta1) * g;
                v.flat(i) = adam.beta2 * v.flat(i) + (1.f - adam.beta2) * g * g;
                const float adam_delta = -lr * (m.flat(i) * adam.mCorr) /
                                          (std::sqrt(v.flat(i) * adam.vCorr) + adam.eps);
                const float decay_delta = (adam.wd != 0.f) ? -lr * adam.wd * before : 0.f;
                value.flat(i) = before + adam_delta + decay_delta;
                const float delta = value.flat(i) - before;
                metrics.gross_path.flat(i)       += std::abs(delta);
                metrics.net_displacement.flat(i) += delta;
            });
        }

        void save_weights(std::ofstream &f) const { value.Save(f); }
        void load_weights(std::ifstream &f)       { value.Load(f); }

        void save(std::ofstream &f) const { value.Save(f); m.Save(f); v.Save(f); metrics.save(f); }
        void load(std::ifstream &f)       { value.Load(f); m.Load(f); v.Load(f); metrics.load(f); }
    };

    template<typename T>
    struct is_param : std::false_type {};

    template<typename TensorT>
    struct is_param<Param<TensorT>> : std::true_type {};

    // @doc: template<typename T> concept IsParam
    /** Concept to verify that a type `T` is a `Param` */
    template<typename T>
    concept IsParam = is_param<std::remove_cvref_t<T> >::value;

    template<typename Tuple>
    struct all_params_check : std::false_type {
    };

    template<typename... Ts>
    struct all_params_check<std::tuple<Ts...> > : std::bool_constant<(IsParam<Ts> && ...)> {
    };

    // @doc: template<typename Tuple> concept IsParamTuple
    /** Concept to verify that a type `Tuple` is a `std::tuple` of `Param` objects */
    template<typename Tuple>
    concept IsParamTuple = all_params_check<std::remove_cvref_t<Tuple> >::value;


    // @doc: template<IsParamTuple Tuple> void ZeroAllGrads(Tuple &&params)
    /** Calls `Param::zero_grad` on each `Param` in the `std::tuple` of `Param`s */
    template<IsParamTuple Tuple>
    void ZeroAllGrads(Tuple &&params) {
        std::apply([](auto &... p) { (p.zero_grad(), ...); }, params);
    }

    // @doc: template<IsParamTuple Tuple> void UpdateAll(Tuple &&params)
    /** Calls `Param::update` on each `Param` in the `std::tuple` of `Param`s */
    template<IsParamTuple Tuple>
    void UpdateAll(Tuple &&params, const AdamState &adam, float lr) {
        std::apply([&](auto &... p) { (p.update(adam, lr), ...); }, params);
    }

    // @doc: template<IsParamTuple Tuple> void SaveAll(Tuple &&params)
    /** Calls `Param::save` on each `Param` in the `std::tuple` of `Param`s */
    template<IsParamTuple Tuple>
    void SaveAll(Tuple &&params, std::ofstream &f) {
        std::apply([&](const auto &... p) { (p.save(f), ...); }, params);
    }

    // @doc: template<IsParamTuple Tuple> void LoadAll(Tuple &&params)
    /** Calls `Param::load` on each `Param` in the `std::tuple` of `Param`s */
    template<IsParamTuple Tuple>
    void LoadAll(Tuple &&params, std::ifstream &f) {
        std::apply([&](auto &... p) { (p.load(f), ...); }, params);
    }

    template<IsParamTuple Tuple>
    void SaveAllWeights(Tuple &&params, std::ofstream &f) {
        std::apply([&](const auto &... p) { (p.save_weights(f), ...); }, params);
    }

    template<IsParamTuple Tuple>
    void LoadAllWeights(Tuple &&params, std::ifstream &f) {
        std::apply([&](auto &... p) { (p.load_weights(f), ...); }, params);
    }

    // ---- free size queries ----

    template<IsTensor T>
    constexpr size_t bytes(const T&)  { return T::Size * sizeof(float); }
    template<IsTensor T>
    constexpr size_t floats(const T&) { return T::Size; }

    template<IsParam T>
    constexpr size_t bytes(const T&)  { return 4 * T::Size * sizeof(float); }
    template<IsParam T>
    constexpr size_t floats(const T&) { return 4 * T::Size; }

    template<typename T>
        requires (!IsTensor<T> && !IsParam<T> &&
                  requires(const T& t) { { t.all_params() } -> IsParamTuple; })
    size_t bytes(const T& obj) {
        return std::apply(
            [](auto&&... ps) -> size_t { return (bytes(ps) + ... + size_t{0}); },
            obj.all_params());
    }
    template<typename T>
        requires (!IsTensor<T> && !IsParam<T> &&
                  requires(const T& t) { { t.all_params() } -> IsParamTuple; })
    size_t floats(const T& obj) {
        return std::apply(
            [](auto&&... ps) -> size_t { return (floats(ps) + ... + size_t{0}); },
            obj.all_params());
    }


    // @doc: template<IsParam... Params> constexpr size_t TotalParamSize
    /**
     * Sum of all `Param` sizes in variadic list of `Param`s
     * `(Params::Size + ...)`
     */
    template<IsParam... Params>
    constexpr size_t TotalParamSize = (Params::Size + ...);


    // @doc: template<IsParamTuple Tuple, size_t... Is> constexpr size_t tuple_param_count_impl(std::index_sequence<Is...>)
    /** Unpacks `IsParamTuple` and sums each `Param::Size`, giving the net size of a `std::tuple` of `Param`s */
    template<IsParamTuple Tuple, size_t... Is>
    constexpr size_t tuple_param_count_impl(std::index_sequence<Is...>) {
        return (static_cast<size_t>(0) + ... + std::remove_reference_t<std::tuple_element_t<Is, Tuple> >::Size);
    }


    // @doc: template<IsParamTuple Tuple> constexpr size_t TupleParamCount
    /**
     * Sum of all `Param` sizes in a `std::tuple` of `Param`s
     * Calls `tuple_param_count_impl`
     */
    template<IsParamTuple Tuple>
    constexpr size_t TupleParamCount =
            tuple_param_count_impl<std::remove_cvref_t<Tuple> >(
                std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<Tuple> > >{});


    // @doc: template<size_t... Dims> void XavierInitMD(Tensor<Dims...> &W, size_t fan_in, size_t fan_out)
    template<size_t... Dims>
    void XavierInitMD(Tensor<Dims...> &W, const size_t fan_in, const size_t fan_out) {
        static std::mt19937 rng{std::random_device{}()};
        const float limit = std::sqrt(6.f / static_cast<float>(fan_in + fan_out));
        std::uniform_real_distribution<float> dist{-limit, limit};
        for (size_t i = 0; i < Tensor<Dims...>::Size; ++i)
            W.flat(i) = dist(rng);
    }


    template<typename InDims, typename OutDims, size_t NumFree>
    struct LearnedContraction {
    };

    // @doc: template<size_t... InDims, size_t... OutDims, size_t NumFree> struct LearnedContraction
    /**
     * Abstraction of learned weight `Tensor` component to any `Block`, handling forward and backward pass internally
     * Templated by `Tensor<InDims...>`, `Tensor<OutDims...>`, and the number of leading (from the left) free axes you want to pass through the transformation, the internal `WeightTensor` type is deduced, initialized, stored, and updated internally
     * Learned weight `Tensor`s are as easy as simply declaring what your desired input and output shapes are.
     */
    template<size_t... InDims, size_t... OutDims, size_t NumFree>
    struct LearnedContraction<Tensor<InDims...>, Tensor<OutDims...>, NumFree> {
        // @doc: using LearnedContraction::InputTensor
        /** Alias for input type: `Tensor<InDims...>` */
        using InputTensor = Tensor<InDims...>;
        // @doc: using LearnedContraction::OutputTensor
        /** Alias for output type: `Tensor<OutDims...>` */
        using OutputTensor = Tensor<OutDims...>;

        static constexpr size_t N_contract_in  = InputTensor::Rank  - NumFree;
        static constexpr size_t N_contract_out = OutputTensor::Rank - NumFree;

        using InSplit   = SplitAt<NumFree, InDims...>;
        using OutSplit  = SplitAt<NumFree, OutDims...>;

        static_assert(std::is_same_v<typename InSplit::head, typename OutSplit::head>,
                      "Free dims of InputTensor and OutputTensor must match");

        using WeightSeq    = ConcatSeqs<typename OutSplit::tail, typename InSplit::tail>::type;
        using WeightTensor = SeqToTensor<WeightSeq>::type;

        // @doc: Param<WeightTensor> LearnedContraction::W_
        /** `Param` type, holding a `WeightTensor` */
        Param<WeightTensor> W_;

        auto all_params()       { return std::tie(W_); }
        auto all_params() const { return std::tie(W_); }

        LearnedContraction() {
            XavierInitMD(W_.value, SeqProduct<typename InSplit::tail>::value,
                         SeqProduct<typename OutSplit::tail>::value);
        }

        // @doc: template<size_t Batch> Tensor<Batch, OutDims...> LearnedContraction::forward(const Tensor<Batch, InDims...> &X) const
        /** Single-sample backward pass: routes to `b.Backward(args.delta_A, args.a, args.a_prev)` */
        template<size_t Batch>
        Tensor<Batch, OutDims...> forward(const Tensor<Batch, InDims...> &X) const {
            return [&]<size_t... I>(std::index_sequence<I...>) {
                return Contract<AxisList<(NumFree + 1 + I)...>{}, AxisList<(N_contract_out + I)...>{}, Mul, Add>(
                    X, W_.value);
            }(std::make_index_sequence<N_contract_in>{});
        }

        // @doc: template<size_t Batch> Tensor<Batch, InDims...> LearnedContraction::backward(const Tensor<Batch, OutDims...> &deltaO, const Tensor<Batch, InDims...> &X)
        /**
         * Batched forward pass implementation, called by `>>`
         * Executes general pattern, implementation heavily documented in code
         */
        template<size_t Batch>
        Tensor<Batch, InDims...> backward(const Tensor<Batch, OutDims...> &deltaO,
                                          const Tensor<Batch, InDims...>  &X) {
            [&]<size_t... I>(std::index_sequence<I...>) {
                W_.grad += Contract<AxisList<I...>{}, AxisList<I...>{}, Mul, Add>(deltaO, X);
            }(std::make_index_sequence<NumFree + 1>{});

            return [&]<size_t... I>(std::index_sequence<I...>) {
                return Contract<AxisList<(NumFree + 1 + I)...>{}, AxisList<I...>{}, Mul, Add>(deltaO, W_.value);
            }(std::make_index_sequence<N_contract_out>{});
        }
    };

    // @doc: template<size_t Batch, size_t... InDims, size_t... OutDims, size_t NumFree> auto operator>>(...)
    /**
     * Forward pass: `InDims...` deduced from `x`, `requires B::InputTensor == Tensor<InDims...>`; routes to `b.Forward(x)`
     * Single-sample overload — batched overload selected automatically when input has a prepended batch dim
     */
    template<size_t Batch, size_t... InDims, size_t... OutDims, size_t NumFree>
    auto operator>>(const Tensor<Batch, InDims...> &X,
                    const LearnedContraction<Tensor<InDims...>, Tensor<OutDims...>, NumFree> &lc) {
        return lc.template forward<Batch>(X);
    }


    // @doc: template<typename T> concept PeekableBlock
    /**
     * Opt-in `concept` for `Block`s to be able to expose their internal activations to an owning `TrainableTensorNetwork`
     * Compliant `Block`s must implement `void peek(SnapshotMap& m, const std::string& s)`
     */
    template<typename T>
    concept PeekableBlock = requires(const T &t, SnapshotMap &m, const std::string &s)
    {
        { t.peek(m, s) } -> std::same_as<void>;
    };


    // @doc: template<typename T> concept Block
    /**
     * Any block in a `TrainableTensorNetwork` must satisfy `Block`:
     * Defined `InputTensor` and `OutputTensor` types which are `Tensor` objects
     * `OutputTensor Forward(InputTensor)`
     * `InputTensor Backward(OutputTensor, OutputTensor, InputTensor)`
     * `auto all_params()` and `auto all_params() const`
     * `TrainableTensorNetwork` blocks need not belong to a specific hierarchy; just satisfy this `concept`
     */
    template<typename T> concept Block =
        requires { typename T::InputTensor; } &&
        requires { typename T::OutputTensor; } &&
        IsTensor<typename T::InputTensor> &&
        IsTensor<typename T::OutputTensor> &&
        requires { typename T::template TrainingCache<1>; } &&
        requires(T t, const T ct,
                 typename PrependBatch<1, typename T::InputTensor>::type  in1,
                 typename PrependBatch<1, typename T::OutputTensor>::type out1,
                 typename T::template TrainingCache<1>       cache,
                 const typename T::template TrainingCache<1> const_cache)
        {
            { ct.template Forward<1>(in1) }
                -> std::same_as<typename PrependBatch<1, typename T::OutputTensor>::type>;
            { ct.template Forward<1>(in1, cache) }
                -> std::same_as<typename PrependBatch<1, typename T::OutputTensor>::type>;
            { t.template Backward<1>(out1, out1, in1, const_cache) }
                -> std::same_as<typename PrependBatch<1, typename T::InputTensor>::type>;
            { t.all_params() };
            { ct.all_params() };
        };


    // @doc: template<Block B, size_t Batch, size_t... InDims> auto operator>>(const Tensor<Batch, InDims...> &X, const B &b)
    /** Batched forward pass: `Batch` and `InDims...` deduced directly from `X`, `requires B::InputTensor == Tensor<InDims...>`; routes to `b.BatchedForward<Batch>(X)` */
    template<Block B, size_t Batch, size_t... InDims>
        requires std::same_as<typename B::InputTensor, Tensor<InDims...>>
    auto operator>>(const Tensor<Batch, InDims...>& X, const B& b) {
        return b.template Forward<Batch>(X);
    }


    // @doc: template<typename TupleT> class ActivationsWrap
    /**
     * Wrapper around a `std::tuple` of `Tensor`s representing intermediate activations of a `TrainableTensorNetwork`
     * Internally stores `std::tuple` and provides safe access to elements and entire `std::tuple` via overloaded `get` methods
     */
    template<typename TupleT>
    class ActivationsWrap {
        TupleT data_;

    public:
        explicit ActivationsWrap(TupleT t) : data_(std::move(t)) {}

        template<size_t N>
        auto get() const & -> const std::tuple_element_t<N, TupleT> & { return std::get<N>(data_); }

        template<size_t N>
        auto get() & -> std::tuple_element_t<N, TupleT> & { return std::get<N>(data_); }

        template<size_t N>
        auto get() && -> std::tuple_element_t<N, TupleT> && = delete;

        const TupleT &tuple() const { return data_; }
    };


    // @doc: template<size_t Batch, typename... Bs> struct TensorTupleBuilder
    /**
     * Recursively build `std::tuple` of `Tensor` objects representing intermediate activations of the network, wrapped by `ActivationsWrap`
     * Base case: one single `Block` left, whose `InputTensor` and `OutputTensor` are wrapped in a `std::tuple`
     */
    template<size_t Batch, typename... Bs>
    struct TensorTupleBuilder;

    template<size_t Batch, typename Last>
    struct TensorTupleBuilder<Batch, Last> {
        using type = std::tuple<
            typename PrependBatch<Batch, typename Last::InputTensor>::type,
            typename PrependBatch<Batch, typename Last::OutputTensor>::type>;
    };

    template<size_t Batch, typename First, typename... Rest>
    struct TensorTupleBuilder<Batch, First, Rest...> {
        using type = decltype(std::tuple_cat(
            std::declval<std::tuple<typename PrependBatch<Batch, typename First::InputTensor>::type>>(),
            std::declval<typename TensorTupleBuilder<Batch, Rest...>::type>()
        ));
    };

} // namespace TTTN
