#pragma once
#include "TensorContract.hpp"
#include "TensorShapeOps.hpp"
#include "TTTN_ML.hpp"
#include "NetworkUtil.hpp"

namespace TTTN {

    template<typename InputT, typename OutputT, size_t NFree>
    class ContractionBlock;

    // InputT  = Tensor<FreeDims..., ContractInDims...>
    // OutputT = Tensor<FreeDims..., ContractOutDims...>
    // NFree   = number of leading axes that pass through (must match between Input and Output)
    //
    // WeightTensor = Tensor<ContractOutDims..., ContractInDims...>
    // Forward:  Contract last N_contract_in axes of X against last N_contract_in axes of W
    // Backward: W grad = contract free axes of delta_O and X
    //           dX     = contract contract-out axes of delta_O against weight's contract-out axes
    template<size_t... InDims, size_t... OutDims, size_t NFree>
    class ContractionBlock<Tensor<InDims...>, Tensor<OutDims...>, NFree> {
    public:
        using InputTensor  = Tensor<InDims...>;
        using OutputTensor = Tensor<OutDims...>;

        static constexpr size_t InRank         = sizeof...(InDims);
        static constexpr size_t OutRank        = sizeof...(OutDims);
        static constexpr size_t N_contract_in  = InRank  - NFree;
        static constexpr size_t N_contract_out = OutRank - NFree;

    private:
        using InSplit  = SplitAt<NFree, InDims...>;
        using OutSplit = SplitAt<NFree, OutDims...>;

        static_assert(std::is_same_v<typename InSplit::head, typename OutSplit::head>,
                      "Free dims of InputTensor and OutputTensor must match");

        using WeightSeq = ConcatSeqs<typename OutSplit::tail, typename InSplit::tail>::type;

        static constexpr size_t fan_in  = SeqProduct<typename InSplit::tail>::value;
        static constexpr size_t fan_out = SeqProduct<typename OutSplit::tail>::value;

    public:
        using WeightTensor = SeqToTensor<WeightSeq>::type;

    private:
        Param<WeightTensor> W_;

        mutable InputTensor      X_cache_{};
        mutable std::vector<float> bX_buf_;

    public:
        auto all_params()       { return std::tie(W_); }
        auto all_params() const { return std::tie(W_); }

        ContractionBlock() { XavierInitMD(W_.value, fan_in, fan_out); }

        OutputTensor Forward(const InputTensor &X) const {
            X_cache_ = X;
            return [&]<size_t... Is>(std::index_sequence<Is...>) {
                return Contract<AxisList<(NFree + Is)...>{}, AxisList<(N_contract_out + Is)...>{}, Mul, Add>(X, W_.value);
            }(std::make_index_sequence<N_contract_in>{});
        }

        InputTensor Backward(const OutputTensor &delta_O,
                             const OutputTensor & /*a*/,
                             const InputTensor  & /*a_prev*/) {
            // W grad: sum of outer products over free axes
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                W_.grad += Contract<AxisList<Is...>{}, AxisList<Is...>{}, Mul, Add>(delta_O, X_cache_);
            }(std::make_index_sequence<NFree>{});

            // dX: contract contract-out axes of delta_O against contract-out axes of W
            return [&]<size_t... Is>(std::index_sequence<Is...>) {
                return Contract<AxisList<(NFree + Is)...>{}, AxisList<Is...>{}, Mul, Add>(delta_O, W_.value);
            }(std::make_index_sequence<N_contract_out>{});
        }

        template<size_t Batch>
        Tensor<Batch, OutDims...> BatchedForward(const Tensor<Batch, InDims...> &X) const {
            bX_buf_.assign(X.data(), X.data() + Tensor<Batch, InDims...>::Size);
            return [&]<size_t... Is>(std::index_sequence<Is...>) {
                return Contract<AxisList<(NFree + 1 + Is)...>{}, AxisList<(N_contract_out + Is)...>{}, Mul, Add>(X, W_.value);
            }(std::make_index_sequence<N_contract_in>{});
        }

        template<size_t Batch>
        Tensor<Batch, InDims...> BatchedBackward(const Tensor<Batch, OutDims...> &delta_O,
                                                  const Tensor<Batch, OutDims...> & /*a*/,
                                                  const Tensor<Batch, InDims...>  & /*a_prev*/) {
            const float inv_batch = 1.f / static_cast<float>(Batch);
            Tensor<Batch, InDims...> bX;
            std::copy(bX_buf_.begin(), bX_buf_.begin() + Tensor<Batch, InDims...>::Size, bX.data());

            // W grad: contract (Batch + Free) axes of delta_O and bX
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                auto localW = Contract<AxisList<Is...>{}, AxisList<Is...>{}, Mul, Add>(delta_O, bX);
                localW *= inv_batch;
                W_.grad += localW;
            }(std::make_index_sequence<NFree + 1>{});

            // dX: contract contract-out axes of delta_O against contract-out axes of W
            return [&]<size_t... Is>(std::index_sequence<Is...>) {
                return Contract<AxisList<(NFree + 1 + Is)...>{}, AxisList<Is...>{}, Mul, Add>(delta_O, W_.value);
            }(std::make_index_sequence<N_contract_out>{});
        }
    };

} // namespace TTTN
