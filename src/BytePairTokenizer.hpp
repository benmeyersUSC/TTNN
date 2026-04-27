#pragma once
#include <fstream>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <cstdint>


template<int VOCAB_SIZE>
class BytePairTokenizer {
    static constexpr int MAX_TOKEN_LEN = 64;
    static constexpr int MAX_VOCAB_SIZE = 65536;
    static_assert(VOCAB_SIZE < MAX_VOCAB_SIZE, "Vocabulary size must be <= 2^16 (65536)");

    // collision free hash of two 16-bits by just mapping them to a 32-bit int
    struct PairHash {
        size_t operator()(std::pair<uint16_t, uint16_t> p) const {
            return std::hash<uint32_t>{}(p.first << 16 | p.second);
        }
    };

    using FrequencyMapT = std::unordered_map<std::pair<uint16_t, uint16_t>, size_t, PairHash>;
    using PQEntry = std::pair<size_t, std::pair<uint16_t, uint16_t> >;

    struct TokenEntry {
        uint8_t bytes[MAX_TOKEN_LEN];
        uint8_t len;
    };

    TokenEntry *Map = nullptr;
    unsigned MapLen = 0;
    std::vector<std::pair<uint16_t, uint16_t> > MergeOrder;


    // replace [A,B] with NewToken and update frequency map
    void Merge(std::vector<uint16_t> &corp, const std::pair<uint16_t, uint16_t> &ab, FrequencyMapT &freq,
               std::priority_queue<PQEntry> &pq) {
        // where did we execute merges?
        std::vector<size_t> mergeIndices;
        // how many we've read
        size_t read = 0;
        // how many we've written
        size_t write = 0;
        // while there's more to read
        while (read < corp.size()) {
            // if looking at first half of new-merged token
            //      and corp has at least one more token
            //          and that next token is the second half of new-merged token
            if (corp[read] == ab.first && read + 1 < corp.size() && corp[read + 1] == ab.second) {
                // if we've written already
                if (write > 0) {
                    // 1. INCR prev to NewToken
                    freq[{corp[write - 1], MapLen}]++;
                    pq.push({freq[{corp[write - 1], MapLen}], {corp[write - 1], MapLen}});
                    // 2. DECR prev to a
                    freq[{corp[write - 1], ab.first}]--;
                    if (freq[{corp[write - 1], ab.first}] == 0) {
                        freq.erase({corp[write - 1], ab.first});
                    } else {
                        pq.push({freq[{corp[write - 1], ab.first}], {corp[write - 1], ab.first}});
                    }
                }
                // save index of new token for later freq cleanup
                mergeIndices.push_back(write);
                // assign new token
                corp[write++] = MapLen;
                // read moves past NewToken (was a) + b
                read += 2;
            } else {
                // otherwise, just move ahead
                corp[write++] = corp[read++];
            }
        }
        freq.erase({ab.first, ab.second}); // impossible for this pair to exist anymore
        // now shed excess cap
        corp.resize(write);

        // final clean up of freq
        for (const auto i: mergeIndices) {
            if (i + 1 < corp.size()) {
                freq[{MapLen, corp[i + 1]}]++;
                pq.push({freq[{MapLen, corp[i + 1]}], {MapLen, corp[i + 1]}});
                freq[{ab.second, corp[i + 1]}]--;
                if (freq[{ab.second, corp[i + 1]}] == 0) {
                    freq.erase({ab.second, corp[i + 1]});
                } else {
                    pq.push({freq[{ab.second, corp[i + 1]}], {ab.second, corp[i + 1]}});
                }
            }
        }
    }

    // create new TokenEntry, update corpus, update freq map
    void NewToken(std::vector<uint16_t> &corp, const std::pair<uint16_t, uint16_t> &ab, FrequencyMapT &freq,
                  std::priority_queue<PQEntry> &pq) {
        if (!Map)return;
        const uint8_t aLen = Map[ab.first].len;
        const uint8_t bLen = Map[ab.second].len;
        for (uint8_t i = 0; i < aLen; ++i) Map[MapLen].bytes[i] = Map[ab.first].bytes[i];
        for (uint8_t i = 0; i < bLen; ++i) Map[MapLen].bytes[i + aLen] = Map[ab.second].bytes[i];
        Map[MapLen].len = aLen + bLen;

        Merge(corp, ab, freq, pq);

        MapLen++;
    }

public:
    [[nodiscard]] const std::vector<std::pair<uint16_t, uint16_t> > &GetMergeOrder() const { return MergeOrder; }
    [[nodiscard]] const TokenEntry *GetMap() const { return Map; }
    [[nodiscard]] unsigned GetMapLen() const { return MapLen; }

    void BPE(std::string_view corpusInPath, std::string_view corpusOutPath) {
        if (Map) throw std::runtime_error("Map already exists.");

        std::ifstream corpus(corpusInPath.data(), std::ios::binary);
        if (!corpus.is_open()) throw std::runtime_error("Failed to open corpus.");

        Map = new TokenEntry[VOCAB_SIZE];
        for (int i = 0; i < 256; ++i) {
            Map[i].bytes[0] = i;
            Map[i].len = 1;
        };
        MapLen = 256;

        // READ CORPUS INTO VECTOR
        corpus.seekg(0, std::ios::end);
        const size_t corpusSize = corpus.tellg();
        corpus.seekg(0, std::ios::beg);

        std::vector<uint8_t> raw(corpusSize);
        corpus.read(reinterpret_cast<char *>(raw.data()), corpusSize);
        std::vector<uint16_t> bytes(raw.begin(), raw.end());

        // FILL FREQ MAP
        FrequencyMapT freq;
        for (size_t i = 0; i < corpusSize - 1; ++i) {
            freq[{bytes[i], bytes[i + 1]}]++;
        }

        // priority queue for highest bigram (next merge candidate)
        std::vector<PQEntry> pqVec;
        pqVec.reserve(freq.size());
        for (auto &[p, count]: freq) pqVec.emplace_back(count, p);
        std::priority_queue pq(std::less<PQEntry>{}, std::move(pqVec)); // top will be highest

        // fill vocabulary and tokenize corpus
        while (MapLen < VOCAB_SIZE) {
            // clear stale entries from top
            while (!pq.empty() && pq.top().first != freq[pq.top().second]) {
                pq.pop();
            }
            // if queue is empty or the best bigram count is 1, we're done
            if (pq.empty() || pq.top().first < 2) break;

            // capture the best valid pair
            auto [count, bestPair] = pq.top();
            pq.pop();

            // turn best pair into token, merge, update freq
            NewToken(bytes, bestPair, freq, pq);
            MergeOrder.emplace_back(bestPair);
        }

        // write tokenized corpus
        std::ofstream tokenizedCorpus(corpusOutPath.data(), std::ios::binary);
        tokenizedCorpus.write(reinterpret_cast<char *>(bytes.data()), bytes.size() * 2);
    }

    // serialize the maps
    void Save(std::string_view outPath) const {
        std::ofstream out(outPath.data(), std::ios::binary);
        if (!out.is_open()) throw std::runtime_error("Failed to open output file.");
        const uint32_t ml = MapLen;
        out.write(reinterpret_cast<const char *>(&ml), sizeof(uint32_t));
        out.write(reinterpret_cast<const char *>(Map), MapLen * sizeof(TokenEntry));
        const uint32_t sz = MergeOrder.size();
        out.write(reinterpret_cast<const char *>(&sz), sizeof(uint32_t));
        out.write(reinterpret_cast<const char *>(MergeOrder.data()),
                  sz * sizeof(std::pair<uint16_t, uint16_t>));
    }

    void Load(std::string_view inPath) {
        std::ifstream inFile(inPath.data(), std::ios::binary);
        if (!inFile.is_open()) throw std::runtime_error("Failed to open input file.");
        if (!Map) Map = new TokenEntry[VOCAB_SIZE];
        uint32_t ml = 0;
        inFile.read(reinterpret_cast<char *>(&ml), sizeof(uint32_t));
        MapLen = ml;
        inFile.read(reinterpret_cast<char *>(Map), MapLen * sizeof(TokenEntry));
        uint32_t sz = 0;
        inFile.read(reinterpret_cast<char *>(&sz), sizeof(uint32_t));
        MergeOrder.resize(sz);
        inFile.read(reinterpret_cast<char *>(MergeOrder.data()),
                    sz * sizeof(std::pair<uint16_t, uint16_t>));
    }

    static void TextToWords(std::string_view inPath, std::string_view outPath) {
        std::ifstream in(inPath.data(), std::ios::binary);
        if (!in.is_open()) throw std::runtime_error("Failed to open input file.");
        std::ofstream out(outPath.data(), std::ios::binary);
        if (!out.is_open()) throw std::runtime_error("Failed to open output file.");

        std::vector<uint8_t> raw(
            (std::istreambuf_iterator<char>(in)),
            (std::istreambuf_iterator<char>())
        );
        std::vector<uint16_t> words(raw.begin(), raw.end());
        out.write(reinterpret_cast<const char *>(words.data()), words.size() * sizeof(uint16_t));
    }

    std::vector<uint16_t> Tokenize(std::string_view inPath) {
        if (MergeOrder.empty()) throw std::runtime_error("No merge order loaded. Run BPE().");

        // in file should be result of TextToWords()
        std::ifstream in(inPath.data(), std::ios::binary);
        if (!in.is_open()) throw std::runtime_error("Failed to open input file.");

        // word count
        in.seekg(0, std::ios::end);
        const size_t wordCount = in.tellg() / sizeof(uint16_t);
        in.seekg(0, std::ios::beg);

        // read the file in to a vector
        std::vector<uint16_t> tokens(wordCount);
        in.read(reinterpret_cast<char *>(tokens.data()), wordCount * sizeof(uint16_t));

        // execute merges in order on the word-repr data
        for (size_t m = 0; m < MergeOrder.size(); ++m) {
            auto [a, b] = MergeOrder[m];
            const uint16_t newTok = 256 + m;
            size_t read = 0, write = 0;
            while (read < tokens.size()) {
                if (tokens[read] == a && read + 1 < tokens.size() && tokens[read + 1] == b) {
                    tokens[write++] = newTok;
                    read += 2;
                } else {
                    tokens[write++] = tokens[read++];
                }
            }
            tokens.resize(write);
        }
        // and we're left with the tokenized sequence
        return tokens;
    }

    [[nodiscard]] std::string Detokenize(const std::vector<uint16_t> &tokens) const {
        if (!Map) throw std::runtime_error("Map is not initialized. Run BPE().");

        std::string result;
        for (const uint16_t tok: tokens) {
            if (tok < MapLen) {
                const TokenEntry &entry = Map[tok];
                result.append(reinterpret_cast<const char *>(entry.bytes), entry.len);
            } else result.append("");
        }
        return result;
    }

    void PrintMap() const {
        if (!Map) throw std::runtime_error("Map is not initialized. Run BPE().");
        for (unsigned tok = 0; tok < MapLen; ++tok) {
            const TokenEntry &entry = Map[tok];
            std::string s(reinterpret_cast<const char *>(entry.bytes), entry.len);
            std::string display;
            for (uint8_t b: s) {
                if (b >= 32 && b < 127) {
                    display += static_cast<char>(b);
                } else {
                    display += "[" + std::to_string(b) + "]";
                }
            }
            std::printf("%5u | len=%2u | %s\n", tok, entry.len, display.c_str());
        }
    }
};
