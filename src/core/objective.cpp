#include "core/objective.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "util/threading.hpp"

namespace miter {

double sigmoid(double z) noexcept {
    if (z >= 0.0) {
        const double ez = std::exp(-z);
        return 1.0 / (1.0 + ez);
    }
    const double ez = std::exp(z);
    return ez / (1.0 + ez);
}

void validate_targets_and_weights(Objective obj, std::span<const double> y,
                                  std::span<const double> sample_weight) {
    if (y.empty()) {
        throw std::invalid_argument("Target array y must be non-empty");
    }
    if (!sample_weight.empty() && sample_weight.size() != y.size()) {
        throw std::invalid_argument("sample_weight length must match y length");
    }

    for (std::size_t i = 0; i < y.size(); ++i) {
        const double yi = y[i];
        if (!std::isfinite(yi)) {
            throw std::invalid_argument("Target array y contains NaN or Inf");
        }
        if (obj == Objective::Binary && (yi < 0.0 || yi > 1.0)) {
            throw std::invalid_argument("Binary objective requires targets in [0, 1]");
        }
    }

    if (!sample_weight.empty()) {
        double weight_sum = 0.0;
        for (const double w : sample_weight) {
            if (!std::isfinite(w) || w < 0.0) {
                throw std::invalid_argument("sample_weight must be finite and non-negative");
            }
            weight_sum += w;
        }
        if (weight_sum <= 0.0) {
            throw std::invalid_argument("Sum of sample_weight must be positive");
        }
    }
}

std::uint32_t infer_and_validate_multiclass_targets(std::span<const double> y,
                                                    std::span<const double> sample_weight,
                                                    std::uint32_t expected_classes) {
    validate_targets_and_weights(Objective::Multiclass, y, sample_weight);
    double max_label = 0.0;
    for (const double yi : y) {
        if (yi < 0.0 || std::floor(yi) != yi) {
            throw std::invalid_argument(
                "Multiclass objective requires integer class labels in [0, n_classes - 1]");
        }
        if (expected_classes > 0U && yi >= static_cast<double>(expected_classes)) {
            throw std::invalid_argument(
                "Validation label out of range [0, n_classes - 1] for multiclass");
        }
        if (yi > max_label) {
            max_label = yi;
        }
    }
    if (expected_classes > 0U) {
        return expected_classes;
    }
    if (max_label < 1.0 || max_label >= 4096.0) {
        throw std::invalid_argument("Multiclass objective requires 2..4096 integer classes");
    }
    return static_cast<std::uint32_t>(max_label) + 1U;
}

std::vector<double> compute_multiclass_init_scores(std::span<const double> y,
                                                   std::span<const double> sample_weight,
                                                   std::uint32_t n_classes, int n_jobs) {
    const std::size_t n = y.size();
    const auto K = static_cast<std::size_t>(n_classes);
    const bool has_w = !sample_weight.empty();
    const std::size_t n_chunks =
        std::max<std::size_t>(1U, (n + kDetChunkRows - 1U) / kDetChunkRows);
    std::vector<double> chunk_w(n_chunks * K, 0.0);
    const int n_threads = resolve_num_threads(n_jobs);
    const bool use_omp = (n_threads > 1) && (n_chunks > 1U) && (n >= 32768U);

    auto count_chunk = [&](std::size_t cu) noexcept {
        const std::size_t i0 = cu * kDetChunkRows;
        const std::size_t i1 = std::min(i0 + kDetChunkRows, n);
        double* dst = chunk_w.data() + cu * K;
        for (std::size_t i = i0; i < i1; ++i) {
            const auto cls = static_cast<std::size_t>(y[i]);
            dst[cls] += has_w ? sample_weight[i] : 1.0;
        }
    };

    if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
        for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
            count_chunk(static_cast<std::size_t>(c));
        }
    } else {
        for (std::size_t c = 0; c < n_chunks; ++c) {
            count_chunk(c);
        }
    }

    std::vector<double> class_w(K, 0.0);
    double total_w = 0.0;
    for (std::size_t c = 0; c < n_chunks; ++c) {
        const double* src = chunk_w.data() + c * K;
        for (std::size_t k = 0; k < K; ++k) {
            class_w[k] += src[k];
            total_w += src[k];
        }
    }

    std::vector<double> init_scores(K, 0.0);
    double mean_log = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
        const double p_k = std::clamp(class_w[k] / total_w, 1e-15, 1.0);
        init_scores[k] = std::log(p_k);
        mean_log += init_scores[k];
    }
    mean_log /= static_cast<double>(K);
    for (std::size_t k = 0; k < K; ++k) {
        init_scores[k] -= mean_log;
    }
    return init_scores;
}

double compute_init_score(Objective obj, std::span<const double> y,
                          std::span<const double> sample_weight, int n_jobs) {
    const std::size_t n = y.size();
    const bool has_w = !sample_weight.empty();
    const std::size_t n_chunks = (n + kDetChunkRows - 1U) / kDetChunkRows;

    double sum_wy = 0.0;
    double sum_w = 0.0;

    if (n_chunks <= 1U) {
        if (!has_w) {
            for (const double yi : y) {
                sum_wy += yi;
            }
            sum_w = static_cast<double>(n);
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                sum_wy += sample_weight[i] * y[i];
                sum_w += sample_weight[i];
            }
        }
    } else {
        std::vector<double> c_wy(n_chunks, 0.0);
        std::vector<double> c_w(n_chunks, 0.0);
        const int n_threads = resolve_num_threads(n_jobs);
        const bool use_omp = (n_threads > 1) && (n >= 32768U);
        if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
            for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
                const auto cu = static_cast<std::size_t>(c);
                const std::size_t i0 = cu * kDetChunkRows;
                const std::size_t i1 = std::min(i0 + kDetChunkRows, n);
                double l_wy = 0.0;
                double l_w = 0.0;
                if (!has_w) {
                    for (std::size_t i = i0; i < i1; ++i) {
                        l_wy += y[i];
                    }
                    l_w = static_cast<double>(i1 - i0);
                } else {
                    for (std::size_t i = i0; i < i1; ++i) {
                        l_wy += sample_weight[i] * y[i];
                        l_w += sample_weight[i];
                    }
                }
                c_wy[cu] = l_wy;
                c_w[cu] = l_w;
            }
        } else {
            for (std::size_t c = 0; c < n_chunks; ++c) {
                const std::size_t i0 = c * kDetChunkRows;
                const std::size_t i1 = std::min(i0 + kDetChunkRows, n);
                double l_wy = 0.0;
                double l_w = 0.0;
                if (!has_w) {
                    for (std::size_t i = i0; i < i1; ++i) {
                        l_wy += y[i];
                    }
                    l_w = static_cast<double>(i1 - i0);
                } else {
                    for (std::size_t i = i0; i < i1; ++i) {
                        l_wy += sample_weight[i] * y[i];
                        l_w += sample_weight[i];
                    }
                }
                c_wy[c] = l_wy;
                c_w[c] = l_w;
            }
        }
        for (std::size_t c = 0; c < n_chunks; ++c) {
            sum_wy += c_wy[c];
            sum_w += c_w[c];
        }
    }

    const double mean = sum_wy / sum_w;
    if (obj == Objective::Regression) {
        return mean;
    }
    const double p0 = std::clamp(mean, 1e-15, 1.0 - 1e-15);
    return std::log(p0 / (1.0 - p0));
}

void compute_gradients(Objective obj, std::span<const double> scores, std::span<const double> y,
                       std::span<const double> sample_weight, float* g, float* h,
                       int n_jobs) noexcept {
    const std::size_t n = y.size();
    const bool has_w = !sample_weight.empty();
    const int n_threads = resolve_num_threads(n_jobs);
    const bool use_omp = (n_threads > 1) && (n >= 16384U);

    if (obj == Objective::Regression) {
        if (!has_w) {
            if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                    g[i] = static_cast<float>(scores[static_cast<std::size_t>(i)] -
                                              y[static_cast<std::size_t>(i)]);
                    h[i] = 1.0F;
                }
            } else {
                for (std::size_t i = 0; i < n; ++i) {
                    g[i] = static_cast<float>(scores[i] - y[i]);
                    h[i] = 1.0F;
                }
            }
        } else {
            if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                    const auto iu = static_cast<std::size_t>(i);
                    const double w = sample_weight[iu];
                    g[iu] = static_cast<float>(w * (scores[iu] - y[iu]));
                    h[iu] = static_cast<float>(w);
                }
            } else {
                for (std::size_t i = 0; i < n; ++i) {
                    const double w = sample_weight[i];
                    g[i] = static_cast<float>(w * (scores[i] - y[i]));
                    h[i] = static_cast<float>(w);
                }
            }
        }
        return;
    }

    if (!has_w) {
        if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                const auto iu = static_cast<std::size_t>(i);
                const double p = sigmoid(scores[iu]);
                g[iu] = static_cast<float>(p - y[iu]);
                h[iu] = static_cast<float>(std::max(p * (1.0 - p), 1e-16));
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                const double p = sigmoid(scores[i]);
                g[i] = static_cast<float>(p - y[i]);
                h[i] = static_cast<float>(std::max(p * (1.0 - p), 1e-16));
            }
        }
    } else {
        if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                const auto iu = static_cast<std::size_t>(i);
                const double p = sigmoid(scores[iu]);
                const double w = sample_weight[iu];
                g[iu] = static_cast<float>(w * (p - y[iu]));
                h[iu] = static_cast<float>(w * std::max(p * (1.0 - p), 1e-16));
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                const double p = sigmoid(scores[i]);
                const double w = sample_weight[i];
                g[i] = static_cast<float>(w * (p - y[i]));
                h[i] = static_cast<float>(w * std::max(p * (1.0 - p), 1e-16));
            }
        }
    }
}

namespace {

void compute_loss_chunk(Objective obj, const double* scores, const double* y,
                        const double* sample_weight, bool has_w, std::size_t i0, std::size_t i1,
                        double& loss_out, double& w_out) noexcept {
    double tl = 0.0;
    double tw = 0.0;
    if (obj == Objective::Regression) {
        if (!has_w) {
            for (std::size_t i = i0; i < i1; ++i) {
                const double diff = scores[i] - y[i];
                tl += diff * diff;
            }
            loss_out = tl;
            w_out = static_cast<double>(i1 - i0);
            return;
        }
        for (std::size_t i = i0; i < i1; ++i) {
            const double diff = scores[i] - y[i];
            const double w = sample_weight[i];
            tl += w * diff * diff;
            tw += w;
        }
        loss_out = tl;
        w_out = tw;
        return;
    }

    if (!has_w) {
        for (std::size_t i = i0; i < i1; ++i) {
            const double z = scores[i];
            const double max_z = (z > 0.0) ? z : 0.0;
            const double abs_z = (z >= 0.0) ? z : -z;
            tl += max_z - y[i] * z + std::log1p(std::exp(-abs_z));
        }
        loss_out = tl;
        w_out = static_cast<double>(i1 - i0);
        return;
    }
    for (std::size_t i = i0; i < i1; ++i) {
        const double z = scores[i];
        const double max_z = (z > 0.0) ? z : 0.0;
        const double abs_z = (z >= 0.0) ? z : -z;
        const double w = sample_weight[i];
        tl += w * (max_z - y[i] * z + std::log1p(std::exp(-abs_z)));
        tw += w;
    }
    loss_out = tl;
    w_out = tw;
}

} // namespace

double compute_loss(Objective obj, std::span<const double> scores, std::span<const double> y,
                    std::span<const double> sample_weight, int n_jobs) noexcept {
    const std::size_t n = y.size();
    const bool has_w = !sample_weight.empty();
    const double* sw_ptr = has_w ? sample_weight.data() : nullptr;
    const std::size_t n_chunks = (n + kDetChunkRows - 1U) / kDetChunkRows;

    if (n_chunks <= 1U) {
        double total_loss = 0.0;
        double total_w = 0.0;
        compute_loss_chunk(obj, scores.data(), y.data(), sw_ptr, has_w, 0U, n, total_loss, total_w);
        return total_loss / total_w;
    }

    constexpr std::size_t kStackChunks = 512U;
    double stack_loss[kStackChunks];
    double stack_w[kStackChunks];

    const int n_threads = resolve_num_threads(n_jobs);
    const bool use_omp = (n_threads > 1) && (n >= 32768U);
    double total_loss = 0.0;
    double total_w = 0.0;

    for (std::size_t c_base = 0; c_base < n_chunks; c_base += kStackChunks) {
        const std::size_t w_chunks = std::min(kStackChunks, n_chunks - c_base);
        if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
            for (std::ptrdiff_t wc = 0; wc < static_cast<std::ptrdiff_t>(w_chunks); ++wc) {
                const auto cu = c_base + static_cast<std::size_t>(wc);
                const std::size_t i0 = cu * kDetChunkRows;
                const std::size_t i1 = std::min(i0 + kDetChunkRows, n);
                compute_loss_chunk(obj, scores.data(), y.data(), sw_ptr, has_w, i0, i1,
                                   stack_loss[wc], stack_w[wc]);
            }
        } else {
            for (std::size_t wc = 0; wc < w_chunks; ++wc) {
                const std::size_t cu = c_base + wc;
                const std::size_t i0 = cu * kDetChunkRows;
                const std::size_t i1 = std::min(i0 + kDetChunkRows, n);
                compute_loss_chunk(obj, scores.data(), y.data(), sw_ptr, has_w, i0, i1,
                                   stack_loss[wc], stack_w[wc]);
            }
        }
        for (std::size_t wc = 0; wc < w_chunks; ++wc) {
            total_loss += stack_loss[wc];
            total_w += stack_w[wc];
        }
    }
    return total_loss / total_w;
}

void compute_multiclass_gradients(const double* scores_class_major, std::size_t n,
                                  std::uint32_t n_classes, std::span<const double> y,
                                  std::span<const double> sample_weight, float* g_class_major,
                                  float* h_class_major, int n_jobs) {
    const auto K = static_cast<std::size_t>(n_classes);
    const bool has_w = !sample_weight.empty();
    const int n_threads = resolve_num_threads(n_jobs);
    const bool use_omp = (n_threads > 1) && (n >= 16384U);

    auto process_row = [&](std::size_t i, double* probs_buf) noexcept {
        double max_z = scores_class_major[i];
        for (std::size_t k = 1U; k < K; ++k) {
            max_z = std::max(max_z, scores_class_major[k * n + i]);
        }
        double sum_exp = 0.0;
        for (std::size_t k = 0U; k < K; ++k) {
            const double e = std::exp(scores_class_major[k * n + i] - max_z);
            probs_buf[k] = e;
            sum_exp += e;
        }
        const double inv_sum = 1.0 / sum_exp;
        const auto yi = static_cast<std::size_t>(y[i]);
        const double w = has_w ? sample_weight[i] : 1.0;
        for (std::size_t k = 0U; k < K; ++k) {
            const double p = probs_buf[k] * inv_sum;
            const double target = (k == yi) ? 1.0 : 0.0;
            g_class_major[k * n + i] = static_cast<float>(w * (p - target));
            h_class_major[k * n + i] = static_cast<float>(w * std::max(2.0 * p * (1.0 - p), 1e-16));
        }
    };

    if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
        {
            double stack_probs[64];
            std::vector<double> heap_probs((K > 64U) ? K : 0U);
            double* probs_buf = (K <= 64U) ? stack_probs : heap_probs.data();
#pragma omp for schedule(static)
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                process_row(static_cast<std::size_t>(i), probs_buf);
            }
        }
#endif
    } else {
        double stack_probs[64];
        std::vector<double> heap_probs((K > 64U) ? K : 0U);
        double* probs_buf = (K <= 64U) ? stack_probs : heap_probs.data();
        for (std::size_t i = 0; i < n; ++i) {
            process_row(i, probs_buf);
        }
    }
}

double compute_multiclass_loss(const double* scores_class_major, std::size_t n,
                               std::uint32_t n_classes, std::span<const double> y,
                               std::span<const double> sample_weight, int n_jobs) {
    const auto K = static_cast<std::size_t>(n_classes);
    const bool has_w = !sample_weight.empty();
    const std::size_t n_chunks =
        std::max<std::size_t>(1U, (n + kDetChunkRows - 1U) / kDetChunkRows);
    std::vector<double> chunk_loss(n_chunks, 0.0);
    std::vector<double> chunk_w(n_chunks, 0.0);
    const int n_threads = resolve_num_threads(n_jobs);
    const bool use_omp = (n_threads > 1) && (n_chunks > 1U) && (n >= 32768U);

    auto loss_chunk = [&](std::size_t cu) noexcept {
        const std::size_t i0 = cu * kDetChunkRows;
        const std::size_t i1 = std::min(i0 + kDetChunkRows, n);
        double tl = 0.0;
        double tw = 0.0;
        for (std::size_t i = i0; i < i1; ++i) {
            double max_z = scores_class_major[i];
            for (std::size_t k = 1U; k < K; ++k) {
                max_z = std::max(max_z, scores_class_major[k * n + i]);
            }
            double sum_exp = 0.0;
            for (std::size_t k = 0U; k < K; ++k) {
                sum_exp += std::exp(scores_class_major[k * n + i] - max_z);
            }
            const auto yi = static_cast<std::size_t>(y[i]);
            const double l_i = max_z + std::log(sum_exp) - scores_class_major[yi * n + i];
            const double w = has_w ? sample_weight[i] : 1.0;
            tl += w * l_i;
            tw += w;
        }
        chunk_loss[cu] = tl;
        chunk_w[cu] = tw;
    };

    if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
        for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
            loss_chunk(static_cast<std::size_t>(c));
        }
    } else {
        for (std::size_t c = 0; c < n_chunks; ++c) {
            loss_chunk(c);
        }
    }

    double total_loss = 0.0;
    double total_w = 0.0;
    for (std::size_t c = 0; c < n_chunks; ++c) {
        total_loss += chunk_loss[c];
        total_w += chunk_w[c];
    }
    return total_loss / total_w;
}

} // namespace miter
