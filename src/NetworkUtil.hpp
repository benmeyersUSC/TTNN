#pragma once
#include <concepts>
#include "Tensor.hpp"
#include "Snapshot.hpp"
#include <cmath>
#include <fstream>
#include <tuple>


namespace TTTN {
    struct AdamState {
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
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


    // @doc: template<typename TensorT> struct Param
    /** `struct` layer around a `Block` to abstract away management, Adam updates */
    template<typename TensorT>
    struct Param {
        TensorT value{};
        TensorT grad{};
        TensorT m{};
        TensorT v{};

        // @doc: static constexpr size_t Param::Size
        /** Size of parameter `Tensor` */
        static constexpr size_t Size = TensorT::Size;


        // @doc: void Param::zero_grad()
        /** Fill `grad` with `0.f` */
        void zero_grad() { grad.fill(0.f); }

        // @doc: void Param::update(const AdamState &adam, float lr)
        /** For each `float` parameter in `value`, use Adam moments and gradient to update */
        void update(const AdamState &adam, float lr) {
            ParForEach(Size, [&](const size_t i) {
                const float g = grad.flat(i);
                m.flat(i) = adam.beta1 * m.flat(i) + (1.f - adam.beta1) * g;
                v.flat(i) = adam.beta2 * v.flat(i) + (1.f - adam.beta2) * g * g;
                value.flat(i) -= lr * (m.flat(i) * adam.mCorr) / (std::sqrt(v.flat(i) * adam.vCorr) + adam.eps);
            });
        }

        // @doc: void Param::save(std::ofstream &f) const
        /** Call `Tensor::Save` on `value` */
        void save(std::ofstream &f) const { value.Save(f); }

        // @doc: void Param::save(std::ifstream &f)
        /** Call `Tensor::Load` on `value` */
        void load(std::ifstream &f) { value.Load(f); }
    };

    template<typename T>
    struct is_param : std::false_type {
    };

    template<typename TensorT>
    struct is_param<Param<TensorT> > : std::true_type {
    };

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


    template<typename InDims, typename OutDims, size_t NumFree>
    struct LearnedContraction {
    };

    // @doc: template<size_t... InDims, size_t... OutDims, size_t NumFree> struct LearnedContraction<Tensor<InDims...>, Tensor<OutDims...>, NumFree>
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

        // @doc: static constexpr size_t LearnedContraction::N_contract_in
        /**
         * Deduced number of contracted axes of `InputTensor`
         * `InputTensor::Rank - NumFree`
         */
        static constexpr size_t N_contract_in = InputTensor::Rank - NumFree;
        // @doc: static constexpr size_t LearnedContraction::N_contract_out
        /**
         * Deduced number of contracted axes of `OutputTensor`
         * `OutputTensor::Rank - NumFree`
         */
        static constexpr size_t N_contract_out = OutputTensor::Rank - NumFree;
        // @doc: using LearnedContraction::InSplit
        /** Split `InDims...` into first `NumFree` free axes (`InSplit::head`) and latter `N_contract_in` contracted axes of `InputTensor` (`InSplit::tail`) */
        using InSplit = SplitAt<NumFree, InDims...>;
        // @doc: using LearnedContraction::OutSplit
        /** Split `OutDims...` into first `NumFree` free axes (`OutSplit::head`) and latter `N_contract_out` contracted axes of `OutputTensor` (`OutSplit::tail`) */
        using OutSplit = SplitAt<NumFree, OutDims...>;

        static_assert(std::is_same_v<typename InSplit::head, typename OutSplit::head>,
                      "Free dims of InputTensor and OutputTensor must match");
        // @doc: using LearnedContraction::WeightSeq
        /**
         * `std::integer_sequence` of two splits fused together:
         * `[OutSplit::tail, InSplit::tail]`
         * `WeightTensor` needs to minor-contract with `InSplit::tail` (the contracted axes of `InputTensor`)
         */
        using WeightSeq = ConcatSeqs<typename OutSplit::tail, typename InSplit::tail>::type;
        // @doc: using LearnedContraction::WeightTensor
        /** `Tensor` type created from `WeightSeq` using `SeqToTensor` */
        using WeightTensor = SeqToTensor<WeightSeq>::type;

        // @doc: Param<WeightTensor> LearnedContraction::W_
        /** `Param` type, holding a `WeightTensor` */
        Param<WeightTensor> W_;
        // @doc: mutable InputTensor LearnedContraction::X_cache_
        /** `mutable` cache for `InputTensor`, used for efficient backward pass */
        mutable InputTensor X_cache_{};
        // @doc: mutable std::vector<float> LearnedContraction::bX_buf_
        /**
         * `mutable` cache for batched input `Tensor`, used for efficient backward pass
         * Uses `std::vector<float>` instead of `Tensor` type because `Batch` is only templated in `>>` and `<<` functions (i.e. `LearnedContraction` and its `WeightTensor` are oblivious to `Batch` dimensions)
         */
        mutable std::vector<float> bX_buf_;


        // @doc: auto LearnedContraction::all_params()
        /** Return `std::tuple` of `Param&` to weight parameter */
        auto all_params() { return std::tie(W_); }
        // @doc: auto LearnedContraction::all_params() const
        /** Return `std::tuple` of `const Param&` to weight parameter */
        auto all_params() const { return std::tie(W_); }

        // @doc: LearnedContraction::LearnedContraction()
        /** Default construct, call `XavierInitMD` on `W_.value` */
        LearnedContraction() {
            XavierInitMD(W_.value, SeqProduct<typename InSplit::tail>::value,
                         SeqProduct<typename OutSplit::tail>::value);
        }

        // @doc: OutputTensor LearnedContraction::forward(const InputTensor &X) const
        /**
         * Forward pass implementation, called by `>>`
         * Executes general pattern, implementation heavily documented in code
         */
        OutputTensor forward(const InputTensor &X) const {
            // cache input
            X_cache_ = X;

            // for I in N_contract_in
            return [&]<size_t... I>(std::index_sequence<I...>) {
                // X: [FX..., CX...]
                // W_: [CY..., CX...]
                // contract:
                //      X[NumFree + I...] = [CX[0], ..., CX[N_contract_in]]
                //      W_[N_contract_out + I...] = [CX[0], ..., CX[N_contract_out + N_contract_in]]
                //
                //      -> OutputTensor: Tensor<FX..., CY...>
                return Contract<AxisList<(NumFree + I)...>{}, AxisList<(N_contract_out + I)...>{}, Mul, Add>(
                    X, W_.value);
            }(std::make_index_sequence<N_contract_in>{});
        }

        // @doc: InputTensor LearnedContraction::backward(const OutputTensor &deltaO)
        /**
         * Backward pass implementation, called by `<<`
         * Executes general pattern, implementation heavily documented in code
         */
        InputTensor backward(const OutputTensor &deltaO) {
            // for I in NumFree
            [&]<size_t... I>(std::index_sequence<I...>) {
                // deltaO: [FX..., CY...]
                // X_cache_: [FX..., CX...]
                // contract:
                //      deltaO[I...] = [FX[0], ..., FX[NumFree]]
                //      X_cache_[I...] = [FX[0], ..., FX[NumFree]]
                //
                //      -> WeightTensor<CY..., CX...>
                W_.grad += Contract<AxisList<I...>{}, AxisList<I...>{}, Mul, Add>(deltaO, X_cache_);
            }(std::make_index_sequence<NumFree>{});

            // for I in N_contract_out
            return [&]<size_t... I>(std::index_sequence<I...>) {
                // deltaO: [FX..., CY...]
                // W_: [CY..., CX...]
                // contract:
                //      deltaO[NumFree + I...] = [CY[0], ..., CY[N_contract_out]]
                //      W_[I...] = [CY[0], ..., CY[N_contract_out]]
                //
                //      -> InputTensor<FX..., CX...>
                return Contract<AxisList<(NumFree + I)...>{}, AxisList<I...>{}, Mul, Add>(deltaO, W_.value);
            }(std::make_index_sequence<N_contract_out>{});
        }

        // @doc: template<size_t Batch> Tensor<Batch, OutDims...> LearnedContraction::batched_forward(const Tensor<Batch, InDims...> &X) const
        /**
         * Batched forward pass implementation, called by `>>`
         * Executes general pattern, implementation heavily documented in code
         */
        template<size_t Batch>
        Tensor<Batch, OutDims...> batched_forward(const Tensor<Batch, InDims...> &X) const {
            // efficiently copy batched X into buffer
            bX_buf_.assign(X.data(), X.data() + Tensor<Batch, InDims...>::Size);

            // for I in N_contract_in
            return [&]<size_t... I>(std::index_sequence<I...>) {
                // X: [B, FX..., CX...]
                // W_: [CY..., CX...]
                // contract:
                //      X[NumFree + 1 + I...] = [CX[1], ..., CX[N_contract_in]]
                //      W_[N_contract_out + I...] = [CX[0], ..., CX[N_contract_out + N_contract_in]]
                //
                //      -> OutputTensor: Tensor<B, FX..., CY...>
                return Contract<AxisList<(NumFree + 1 + I)...>{}, AxisList<(N_contract_out + I)...>{}, Mul, Add>(
                    X, W_.value);
            }(std::make_index_sequence<N_contract_in>{});
        }

        // @doc: template<size_t Batch> Tensor<Batch, InDims...> LearnedContraction::batched_backward(const Tensor<Batch, OutDims...> &deltaO)
        /**
         * Batched backward pass implementation, called by `<<`
         * Executes general pattern, implementation heavily documented in code
         */
        template<size_t Batch>
        Tensor<Batch, InDims...> batched_backward(const Tensor<Batch, OutDims...> &deltaO) {
            // precompute batch mean correction factor
            const float inv_batch = 1.f / static_cast<float>(Batch);
            // load in cached batched X
            Tensor<Batch, InDims...> bX;
            std::copy(bX_buf_.begin(), bX_buf_.begin() + Tensor<Batch, InDims...>::Size, bX.data());

            // for I in NumFree+1
            [&]<size_t... I>(std::index_sequence<I...>) {
                // deltaO: [B, FX..., CY...]
                // X_cache_: [B, FX..., CX...]
                // contract:
                //      deltaO[I...] = [B, FX[0], ..., FX[NumFree + 1]]
                //      X_cache_[I...] = [B, FX[0], ..., FX[NumFree + 1]]
                //
                //      -> WeightTensor<CY..., CX...>
                auto localW = Contract<AxisList<I...>{}, AxisList<I...>{}, Mul, Add>(deltaO, bX);
                localW *= inv_batch;
                W_.grad += localW;
            }(std::make_index_sequence<NumFree + 1>{});

            // for I in N_contract_out
            return [&]<size_t... I>(std::index_sequence<I...>) {
                // deltaO: [B, FX..., CY...]
                // W_: [CY..., CX...]
                // contract:
                //      deltaO[NumFree + 1 + I...] = [CY[0], ..., CY[N_contract_out]]
                //      W_[I...] = [CY[0], ..., CY[N_contract_out]]
                //
                //      -> InputTensor<B, FX..., CX...>
                return Contract<AxisList<(NumFree + 1 + I)...>{}, AxisList<I...>{}, Mul, Add>(deltaO, W_.value);
            }(std::make_index_sequence<N_contract_out>{});
        }
    };

    // @doc: template<size_t...InDims, size_t... OutDims, size_t NumFree> auto operator>>(const Tensor<InDims...> &X, const LearnedContraction<Tensor<InDims...>, Tensor<OutDims...>, NumFree> &lc)
    /**
     * Terse operator for forward pass (takes `const Tensor<InDims...> &X` on the left of the operator, `LearnedContraction` on the right)
     * *Propagate signal forward through `W_`*
     */
    template<size_t... InDims, size_t... OutDims, size_t NumFree>
    auto operator>>(const Tensor<InDims...> &X,
                    const LearnedContraction<Tensor<InDims...>, Tensor<OutDims...>, NumFree> &lc) {
        return lc.forward(X);
    }


    // @doc: template<size_t...InDims, size_t... OutDims, size_t NumFree> auto operator<<(LearnedContraction<Tensor<InDims...>, Tensor<OutDims...>, NumFree> &lc, const Tensor<OutDims...> &deltaO)
    /**
     * Terse operator for backward pass (takes `LearnedContraction` on the left of the operator, `const Tensor<OutDims...> &deltaO` on the right)
     * *Propagate gradient backward through `W_`*
     */
    template<size_t... InDims, size_t... OutDims, size_t NumFree>
    auto operator<<(LearnedContraction<Tensor<InDims...>, Tensor<OutDims...>, NumFree> &lc,
                    const Tensor<OutDims...> &deltaO) {
        return lc.backward(deltaO);
    }

    // @doc: template<size_t Batch, size_t... InDims, size_t... OutDims, size_t NumFree> auto operator>>(const Tensor<Batch, InDims...> &X, const LearnedContraction<Tensor<InDims...>, Tensor<OutDims...>, NumFree> &lc)
    /**
     * Terse operator for batched forward pass (takes `const Tensor<Batch, InDims...> &X` on the left of the operator, `LearnedContraction` on the right)
     * *Propagate signal forward through `W_`*
     */
    template<size_t Batch, size_t... InDims, size_t... OutDims, size_t NumFree>
    auto operator>>(const Tensor<Batch, InDims...> &X,
                    const LearnedContraction<Tensor<InDims...>, Tensor<OutDims...>, NumFree> &lc) {
        return lc.template batched_forward<Batch>(X);
    }

    // @doc: template<size_t Batch, size_t... InDims, size_t... OutDims, size_t NumFree> auto operator<<(LearnedContraction<Tensor<InDims...>, Tensor<OutDims...>, NumFree> &lc, const Tensor<Batch, OutDims...> &deltaO)
    /**
     * Terse operator for backward pass (takes `LearnedContraction` on the left of the operator, `const Tensor<Batch, OutDims...> &deltaO` on the right)
     * *Propagate gradient backward through `W_`*
     */
    template<size_t Batch, size_t... InDims, size_t... OutDims, size_t NumFree>
    auto operator<<(LearnedContraction<Tensor<InDims...>, Tensor<OutDims...>, NumFree> &lc,
                    const Tensor<Batch, OutDims...> &deltaO) {
        return lc.template batched_backward<Batch>(deltaO);
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
            requires(T t, const T ct,
                     typename T::InputTensor in,
                     typename T::OutputTensor out)
            {
                { ct.Forward(in) } -> std::same_as<typename T::OutputTensor>;
                { t.Backward(out, out, in) } -> std::same_as<typename T::InputTensor>;
                { t.all_params() }; // non-const: ZeroGrad + Update
                { ct.all_params() }; // const: Save
            };


    // @doc: template<typename TupleT> class ActivationsWrap
    /**
     * Wrapper around a `std::tuple` of `Tensor`s representing intermediate activations of a `TrainableTensorNetwork`
     * Internally stores `std::tuple` and provides safe access to elements and entire `std::tuple` via overloaded `get` methods
     */
    template<typename TupleT>
    class ActivationsWrap {
        TupleT data_;

    public:
        // @doc: explicit ActivationsWrap::ActivationsWrap(TupleT t)
        /** `explicit` constructor which `move`s incoming `std::tuple` into member `data_` */
        explicit ActivationsWrap(TupleT t) : data_(std::move(t)) {
        }

        // @doc: template<size_t N> auto ActivationsWrap::get() const & -> const std::tuple_element_t<N, TupleT> &
        /** `const &` getter, valid as long as `ActivationsWrap` object exists */
        template<size_t N>
        auto get() const & -> const std::tuple_element_t<N, TupleT> & {
            return std::get<N>(data_);
        }

        // @doc: template<size_t N> auto ActivationsWrap::get() & -> std::tuple_element_t<N, TupleT> &
        /** `&` getter, valid as long as `ActivationsWrap` object exists */
        template<size_t N>
        auto get() & -> std::tuple_element_t<N, TupleT> & {
            return std::get<N>(data_);
        }

        // @doc: template<size_t N> auto ActivationsWrap::get() && -> std::tuple_element_t<N, TupleT> &&
        /**
         * Explicitly `delete`d function!
         * Getting temporary activation `Tensor` from temporary `ActivationsWrap` is a compile error, you must bind the `ActivationsWrap` object to a variable and get a reference
         * This is because it would return a dangling reference to a soon-deleted `ActivationsWrap` object
         * Instead of:
         * `auto& act = net.BatchedForwardAll(X).get<2>();`
         * You must do:
         * `auto& wrap = net.BatchedForwardAll(X);`
         * `auto& act = wrap.get<2>();`
         */
        template<size_t N>
        auto get() && -> std::tuple_element_t<N, TupleT> && = delete;

        // @doc: const TupleT &ActivationsWrap::tuple() const
        /** `const &` to raw `std::tuple` */
        const TupleT &tuple() const { return data_; }
    };


    template<typename... Bs>
    struct TensorTupleBuilder;

    // @doc: template<typename Last> struct TensorTupleBuilder<Last>
    /**
     * Recursively build `std::tuple` of `Tensor` objects representing intermediate activations of the network, wrapped by `ActivationsWrap`
     * Base case: one single `Block` left, whose `InputTensor` and `OutputTensor` are wrapped in a `std::tuple`
     */
    template<typename Last>
    struct TensorTupleBuilder<Last> {
        using type = std::tuple<typename Last::InputTensor, typename Last::OutputTensor>;
    };

    // @doc: template<typename First, typename... Rest> struct TensorTupleBuilder<First, Rest...>
    /**
     * Recursively build `std::tuple` of `Tensor` objects representing intermediate activations of the network, wrapped by `ActivationsWrap`
     * Recursive case: `std::tuple_cat` of `First` `InputTensor` object and `TensorTupleBuilder<Rest...>`
     */
    template<typename First, typename... Rest>
    struct TensorTupleBuilder<First, Rest...> {
        using type = decltype(std::tuple_cat(
            std::declval<std::tuple<typename First::InputTensor> >(),
            std::declval<typename TensorTupleBuilder<Rest...>::type>()
        ));
    };


    template<size_t Batch, typename... Bs>
    struct BatchedTensorTupleBuilder;

    // @doc: template<size_t Batch, typename Last> struct BatchedTensorTupleBuilder<Batch, Last>
    /**
     * For `Batched` functions and use-cases, create a `Batched` version of a `std::tuple` of activations by passing `PrependBatch<Batch, ...>` on all `Tensor`s that `TensorTupleBuilder` adds raw
     * Base case: one single `Block` left, whose `InputTensor` and `OutputTensor` are wrapped in `PrependBatch<Batch, ...>` and then in a `std::tuple`
     */
    template<size_t Batch, typename Last>
    struct BatchedTensorTupleBuilder<Batch, Last> {
        using type = std::tuple<
            typename PrependBatch<Batch, typename Last::InputTensor>::type,
            typename PrependBatch<Batch, typename Last::OutputTensor>::type>;
    };

    // @doc: template<size_t Batch, typename First, typename... Rest> struct BatchedTensorTupleBuilder<Batch, First, Rest...>
    /**
     * For `Batched` functions and use-cases, create a `Batched` version of a `std::tuple` of activations by passing `PrependBatch<Batch, ...>` on all `Tensor`s that `TensorTupleBuilder` adds raw
     * Recursive case: `std::tuple_cat` of `First` `PrependBatch<Batch, InputTensor>` object and `BatchedTensorTupleBuilder<Rest...>`
     */
    template<size_t Batch, typename First, typename... Rest>
    struct BatchedTensorTupleBuilder<Batch, First, Rest...> {
        using type = decltype(std::tuple_cat(
            std::declval<std::tuple<typename PrependBatch<Batch, typename First::InputTensor>::type> >(),
            std::declval<typename BatchedTensorTupleBuilder<Batch, Rest...>::type>()
        ));
    };
};
