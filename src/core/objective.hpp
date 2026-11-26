#pragma once

#include <miter/model.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace miter {

double sigmoid(double z) noexcept;

void validate_targets_and_weights(Objective obj, std::span<const double> y,
                                  std::span<const double> sample_weight);

std::uint32_t infer_and_validate_multiclass_targets(std::span<const double> y,
                                                    std::span<const double> sample_weight,
                                                    std::uint32_t expected_classes = 0U);

double compute_init_score(Objective obj, std::span<const double> y,
                          std::span<const double> sample_weight, int n_jobs = 1);

std::vector<double> compute_multiclass_init_scores(std::span<const double> y,
                                                   std::span<const double> sample_weight,
                                                   std::uint32_t n_classes, int n_jobs = 1);

void compute_gradients(Objective obj, std::span<const double> scores, std::span<const double> y,
                       std::span<const double> sample_weight, float* g, float* h,
                       int n_jobs = 1) noexcept;

void compute_multiclass_gradients(const double* scores_class_major, std::size_t n,
                                  std::uint32_t n_classes, std::span<const double> y,
                                  std::span<const double> sample_weight, float* g_class_major,
                                  float* h_class_major, int n_jobs = 1) noexcept;

double compute_loss(Objective obj, std::span<const double> scores, std::span<const double> y,
                    std::span<const double> sample_weight, int n_jobs = 1) noexcept;

double compute_multiclass_loss(const double* scores_class_major, std::size_t n,
                               std::uint32_t n_classes, std::span<const double> y,
                               std::span<const double> sample_weight, int n_jobs = 1) noexcept;

} // namespace miter
