#include "src/TTTN.hpp"

using namespace TTTN;


int main() {
    using BPEType = BytePairTokenizer<8192>;
    BPEType bpe;

    // ---- FIT ----
    std::puts("=== Fitting BPE on corpus ===");
    bpe.BPE("tttn_dump.txt", "corpus_tokenized.bin");
    bpe.Save("bpe_model.bin");
    std::puts("Model saved to bpe_model.bin\n");

    // ---- PRINT MAP ----
    std::puts("=== Token Map ===");
    bpe.PrintMap();
    std::putchar('\n');

    // ---- TOKENIZE A TEXT FILE ----
    std::puts("=== Tokenizing text.txt ===");
    BPEType::TextToWords("oldmain.txt", "text_words.bin");
    auto tokens = bpe.Tokenize("text_words.bin");

    std::puts("--- Token sequence ---");
    for (const uint16_t tok: tokens) {
        const auto &entry = bpe.GetMap()[tok];
        std::string display;
        for (uint8_t i = 0; i < entry.len; ++i) {
            uint8_t b = entry.bytes[i];
            if (b >= 32 && b < 127) display += static_cast<char>(b);
            else display += "[" + std::to_string(b) + "]";
        }
        std::printf("%5u : %s\n", tok, display.c_str());
    }

    // ---- DETOKENIZE ----
    std::putchar('\n');
    std::puts("=== Detokenized output ===");
    std::string reconstructed = bpe.Detokenize(tokens);
    std::puts(reconstructed.c_str());

    // ---- RELOAD AND VERIFY ----
    std::putchar('\n');
    std::puts("=== Reload from disk and re-detokenize ===");
    BPEType bpe2;
    bpe2.Load("bpe_model.bin");
    std::printf("Loaded MapLen=%u MergeOrder.size=%zu\n", bpe2.GetMapLen(), bpe2.GetMergeOrder().size());

    auto tokens2 = bpe2.Tokenize("text_words.bin");
    std::string reconstructed2 = bpe2.Detokenize(tokens2);
    std::puts(reconstructed2 == reconstructed ? "MATCH" : "MISMATCH");

    return 0;
}
