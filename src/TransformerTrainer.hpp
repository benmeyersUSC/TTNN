#pragma once
#include "EncoderDecoder.hpp"
#include <concepts>
#include <type_traits>

template <typename T>
concept IsEnum = std::is_enum_v<T>;


/*

// CONSTANTS, CONFIG
//
// DATASET CONSTANTS
constexpr int N_SUBSETS_TRAIN = 5;          // how many folder subsets
constexpr int SUBSET_SIZE_TRAIN = 4849;     // how big is each subset 
constexpr int CHUNK_SIZE_TRAIN = 1024;      // chunk size to take from a single subset
constexpr int LAPS_THIS_SESSION = 99;        // how many full laps (not necessarily full 24k epoch) this run
constexpr int TOTAL_TRAIN = N_SUBSETS_TRAIN * SUBSET_SIZE_TRAIN * LAPS_THIS_SESSION;
constexpr int EXAMPLES_PER_EPOCH = N_SUBSETS_TRAIN * SUBSET_SIZE_TRAIN; // ~24245
// BATCH AND LOGGING CONSTANTS
constexpr int CHUNKS_PER_GROUP = 8;         // data chunks processed before each RL step
constexpr size_t BATCH = 32;                // CE minibatch size
constexpr int LOG_EVERY = 4;                // batches between progress-bar prints
constexpr int SENTINEL_LOG_EVERY = 10000;   // trained examples between 10k-gate evals
// RL CONSTANTS
constexpr int RL_K = 8;                     // autoregressive runs per RL update 
constexpr float RL_BASELINE_DECAY = 0.99f;  // EMA decay for running reward baseline
//

*/


namespace TTTN{

    template <
    // NETWORK SPEC
        size_t SrcLen, 
        size_t TgtLen, 
        size_t VocabSize, 
        IsEnum Vocab, 
        size_t EmbeddingDimension, 
        size_t NumHeads, 
        size_t FFNSize, 
        size_t NEnc, 
        size_t NDec, 
    
    // DATA SPEC
        size_t PadId,
        size_t NSubsetsTrain, 
        size_t SubsetSizeTrain, 
        size_t ChunkSizeTrain, 
        size_t Laps, 

    // BATCH AND LOGGING CONSTANTS
        size_t ChunksPerGroup, 
        size_t Batch, 
        size_t LogEvery, 
        size_t SentinelLogEvery,

    // RL CONSTANTS
        size_t RL_K, 
        float RL_BaselineDecay
    >
    class TransformerTrainer{
        using BlockT = EncoderDecoderBlock<SrcLen, TgtLen, VocabSize, EmbeddingDimension, NumHeads, FFNSize, NEnc, NDec, PadId>;
        using NetworkT = TrainableTensorNetwork<BlockT>;
        using InputT = NetworkT::InputTensor;
        using OutputT = NetworkT::OutputTensor;

        static constexpr size_t TotalTrain = NSubsetsTrain * SubsetSizeTrain * Laps; 
        static constexpr ExamplesPerEpoch = NSubsetsTrain * SubsetSizeTrain;
        static constexpr size_t ExamplesPerGroup = CHUNKS_PER_GROUP * CHUNK_SIZE_TRAIN;
        static constexpr size_t GroupsPerLap = (TOTAL_TRAIN + ExamplesPerGroup - 1) / examples_per_group;
    };


}