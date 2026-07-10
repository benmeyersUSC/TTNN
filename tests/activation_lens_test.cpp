#include "TTTN.hpp"
#include <cstdio>
#include <random>

using namespace TTTN;

// Three-block MLP: two Tanh layers then a pure linear readout, so boundary 2 -> output
// is exactly affine and the fitted lens there has a closed form (the readout W).
using B1  = DenseMDBlock<Tensor<6>,  Tensor<10>, Tanh>;
using B2  = DenseMDBlock<Tensor<10>, Tensor<8>,  Tanh>;
using B3  = DenseMDBlock<Tensor<8>,  Tensor<5>,  Linear>;
using Net = TrainableTensorNetwork<B1, B2, B3>;

static int g_failures = 0;

static void check(const bool ok, const char *name, const float worst) {
    std::printf("%-52s %s   (max err %.3e)\n", name, ok ? "PASS" : "FAIL", static_cast<double>(worst));
    if (!ok) ++g_failures;
}

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.f, 1.f);

    Net net;
    // Non-trivial biases so the affine-caveat test actually exercises the constant part.
    std::apply([&](auto &... ps) { ((void) [&] {
        for (size_t k = 0; k < ps.Size; ++k) ps.value.flat(k) += 0.1f * dist(rng);
    }(), ...); }, net.all_params());

    constexpr size_t Batch = 6;
    Tensor<Batch, 6> X;
    for (size_t i = 0; i < X.Size; ++i) X.flat(i) = dist(rng);

    // ── 1. Lens at the final boundary is the identity ────────────────────────
    {
        const auto L = FitActivationLens<Net::NumBlocks, Batch>(net, X);
        static_assert(std::is_same_v<std::remove_cvref_t<decltype(L)>, Tensor<5, 5>>);
        float worst = 0.f;
        for (size_t j = 0; j < 5; ++j)
            for (size_t k = 0; k < 5; ++k) {
                const float want = (j == k) ? 1.f : 0.f;
                worst = std::max(worst, std::abs(L.flat(j * 5 + k) - want));
            }
        check(worst < 1e-7f, "final-boundary lens == identity", worst);
    }

    // ── 2. Lens at the last interior boundary == readout weight matrix ──────
    const auto L2 = FitActivationLens<2, Batch>(net, X);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(L2)>, Tensor<5, 8>>);
    {
        const auto &W3 = std::get<4>(net.all_params()).value; // B3's W: Tensor<5, 8>
        float worst = 0.f;
        for (size_t i = 0; i < W3.Size; ++i)
            worst = std::max(worst, std::abs(L2.flat(i) - W3.flat(i)));
        check(worst < 1e-5f, "readout-boundary lens == W (closed form)", worst);
    }

    // ── 3. Zero cross-context variance where downstream is linear ───────────
    {
        ActivationLensAccumulator<Net, 2> acc;
        for (size_t c = 0; c < 16; ++c) {
            Tensor<1, 6> x;
            for (size_t k = 0; k < 6; ++k) x.flat(k) = dist(rng);
            acc.Add(net, x);
        }
        const float coh = acc.Coherence();
        const auto rows = acc.RowCoherence();
        float worst = std::abs(coh - 1.f);
        for (size_t j = 0; j < 5; ++j) worst = std::max(worst, std::abs(rows.flat(j) - 1.f));
        check(worst < 1e-5f, "linear-downstream coherence == 1 (dispersion 0)", worst);
    }

    // ── 4. Per-context lens at the input == finite-difference Jacobian ──────
    {
        Tensor<1, 6> x;
        for (size_t k = 0; k < 6; ++k) x.flat(k) = dist(rng);
        const auto L0 = FitActivationLens<0, 1>(net, x);
        static_assert(std::is_same_v<std::remove_cvref_t<decltype(L0)>, Tensor<5, 6>>);

        constexpr float eps = 1e-2f;
        float worst = 0.f;
        for (size_t k = 0; k < 6; ++k) {
            auto xp = x, xm = x;
            xp.flat(k) += eps;
            xm.flat(k) -= eps;
            const auto yp = net.template Forward<1>(xp);
            const auto ym = net.template Forward<1>(xm);
            for (size_t j = 0; j < 5; ++j) {
                const float fd = (yp.flat(j) - ym.flat(j)) / (2.f * eps);
                worst = std::max(worst, std::abs(L0.flat(j * 6 + k) - fd));
            }
        }
        check(worst < 5e-3f, "input-boundary lens == finite differences", worst);
    }

    // ── 5. Interior TargetIdx: boundary 1 -> boundary 2 == block 1 Jacobian ─
    {
        Tensor<1, 6> x;
        for (size_t k = 0; k < 6; ++k) x.flat(k) = dist(rng);
        const auto L12 = FitActivationLens<1, 1, 2>(net, x);
        static_assert(std::is_same_v<std::remove_cvref_t<decltype(L12)>, Tensor<8, 10>>);

        const auto acts = net.template ForwardAll<1>(x);
        const auto &h1  = acts.template get<1>(); // Tensor<1, 10>

        constexpr float eps = 1e-2f;
        float worst = 0.f;
        for (size_t k = 0; k < 10; ++k) {
            auto hp = h1, hm = h1;
            hp.flat(k) += eps;
            hm.flat(k) -= eps;
            const auto yp = net.template block<1>().template Forward<1>(hp);
            const auto ym = net.template block<1>().template Forward<1>(hm);
            for (size_t j = 0; j < 8; ++j) {
                const float fd = (yp.flat(j) - ym.flat(j)) / (2.f * eps);
                worst = std::max(worst, std::abs(L12.flat(j * 10 + k) - fd));
            }
        }
        check(worst < 5e-3f, "interior-target lens == block Jacobian (FD)", worst);
    }

    // ── 6. ApplyLens at the linear boundary reproduces logits minus bias ────
    {
        Tensor<1, 6> x;
        for (size_t k = 0; k < 6; ++k) x.flat(k) = dist(rng);
        const auto acts = net.template ForwardAll<1>(x);
        Tensor<8> h2;
        for (size_t k = 0; k < 8; ++k) h2.flat(k) = acts.template get<2>().flat(k);

        const auto lin  = ApplyLens(L2, h2); // Tensor<5>: the linearized logits
        const auto &b3  = std::get<5>(net.all_params()).value;
        const auto y    = net.template Forward<1>(x);
        float worst = 0.f;
        for (size_t j = 0; j < 5; ++j)
            worst = std::max(worst, std::abs(lin.flat(j) + b3.flat(j) - y.flat(j)));
        check(worst < 1e-5f, "ApplyLens + bias == logits (affine caveat)", worst);
    }

    // ── 7b. ForwardFrom: interior replay reproduces the full forward ────────
    {
        Tensor<1, 6> x;
        for (size_t k = 0; k < 6; ++k) x.flat(k) = dist(rng);
        const auto y0   = net.template Forward<1>(x);
        const auto yff0 = net.template ForwardFrom<1, 0>(x);
        const auto acts = net.template ForwardAll<1>(x);
        const auto yff2 = net.template ForwardFrom<1, 2>(acts.template get<2>());
        float worst = 0.f;
        for (size_t j = 0; j < 5; ++j) {
            worst = std::max(worst, std::abs(yff0.flat(j) - y0.flat(j)));
            worst = std::max(worst, std::abs(yff2.flat(j) - y0.flat(j)));
        }
        check(worst < 1e-6f, "ForwardFrom<0>/<2> == full forward", worst);
    }

    // ── 7. LensVector row extraction ─────────────────────────────────────────
    {
        const auto v = LensVector(L2, 3); // Tensor<8>: d logit_3 / d h2
        float worst = 0.f;
        for (size_t k = 0; k < 8; ++k)
            worst = std::max(worst, std::abs(v.flat(k) - L2.flat(3 * 8 + k)));
        check(worst == 0.f, "LensVector == lens row", worst);
    }

    std::printf("\n%s\n", g_failures == 0 ? "ALL GOLDEN TESTS PASS" : "FAILURES PRESENT");
    return g_failures;
}
