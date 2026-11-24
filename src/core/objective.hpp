#pragma once

#include <miter/model.hpp>

#include <span>

namespace miter {

double sigmoid(double z) noexcept;

void validate_targets_and_weights(Objective obj, std::span<const double> y,
                                  std::span<const double> sample_weight);

double compute_init_score(Objective obj, std::span<const double> y,
                          std::span<const double> sample_weight, int n_jobs = 1);

void compute_gradients(Objective obj, std::span<const double> scores, std::span<const double> y,
                       std::span<const double> sample_weight, float* g, float* h,
                       int n_jobs = 1) noexcept;

double compute_loss(Objective obj, std::span<const double> scores, std::span<const double> y,
                    std::span<const double> sample_weight, int n_jobs = 1) noexcept;

} // namespace miter
