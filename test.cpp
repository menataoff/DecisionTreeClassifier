#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <algorithm>
#include <fstream>
#include <cmath>
#include "decision_tree.h"

class ExtendedPruningTester {
private:
    std::mt19937 gen;

    struct Dataset {
        std::string name;
        std::vector<std::vector<double>> X;
        std::vector<int> y;
        int n_classes;
        int n_features;
    };

    struct TestMetrics {
        double train_accuracy;
        double test_accuracy;
        int n_leaves;
        int depth;
        double inference_time_ms; // Можно добавить замер времени
    };

public:
    ExtendedPruningTester(int seed = 42) : gen(seed) {}

    // ==================== РАЗНЫЕ ТИПЫ ДАННЫХ ====================

    Dataset generate_linearly_separable(int n_samples = 300) {
        Dataset ds;
        ds.name = "Linearly Separable";
        ds.n_classes = 2;
        ds.n_features = 5;

        std::normal_distribution<> class0(-1.0, 0.5);
        std::normal_distribution<> class1(1.0, 0.5);
        std::uniform_real_distribution<> noise_feat(-0.5, 0.5);

        for (int i = 0; i < n_samples; ++i) {
            std::vector<double> features(5);

            if (i < n_samples / 2) {
                // Class 0
                features[0] = class0(gen);
                features[1] = class0(gen) * 0.7;
                y.push_back(0);
            } else {
                // Class 1
                features[0] = class1(gen);
                features[1] = class1(gen) * 0.7;
                y.push_back(1);
            }

            // Noise features
            for (int j = 2; j < 5; ++j) {
                features[j] = noise_feat(gen);
            }

            ds.X.push_back(features);
        }

        return ds;
    }

    Dataset generate_xor_pattern(int n_samples = 400) {
        Dataset ds;
        ds.name = "XOR Pattern";
        ds.n_classes = 2;
        ds.n_features = 3;

        std::normal_distribution<> cluster(0.0, 0.3);

        // 4 кластера как в XOR
        for (int i = 0; i < n_samples; ++i) {
            std::vector<double> features(3);
            int cluster_id = i % 4;

            if (cluster_id == 0) {  // (0,0) → class 0
                features[0] = cluster(gen) - 1.0;
                features[1] = cluster(gen) - 1.0;
                y.push_back(0);
            }
            else if (cluster_id == 1) {  // (0,1) → class 1
                features[0] = cluster(gen) - 1.0;
                features[1] = cluster(gen) + 1.0;
                y.push_back(1);
            }
            else if (cluster_id == 2) {  // (1,0) → class 1
                features[0] = cluster(gen) + 1.0;
                features[1] = cluster(gen) - 1.0;
                y.push_back(1);
            }
            else {  // (1,1) → class 0
                features[0] = cluster(gen) + 1.0;
                features[1] = cluster(gen) + 1.0;
                y.push_back(0);
            }

            features[2] = std::uniform_real_distribution<>(-0.5, 0.5)(gen);
            ds.X.push_back(features);
        }

        return ds;
    }

    Dataset generate_multi_class_circles(int n_samples = 500, int n_classes = 4) {
        Dataset ds;
        ds.name = "Multi-class Circles";
        ds.n_classes = n_classes;
        ds.n_features = 3;

        std::uniform_real_distribution<> angle_dist(0, 2 * M_PI);
        std::normal_distribution<> noise(0.0, 0.15);

        for (int i = 0; i < n_samples; ++i) {
            std::vector<double> features(3);
            int label = i % n_classes;
            double radius = 1.0 + label * 0.8;  // Разные радиусы

            double angle = angle_dist(gen);
            features[0] = radius * cos(angle) + noise(gen);
            features[1] = radius * sin(angle) + noise(gen);
            features[2] = std::uniform_real_distribution<>(-1.0, 1.0)(gen);

            ds.X.push_back(features);
            y.push_back(label);
        }

        return ds;
    }

    Dataset generate_high_dimensional(int n_samples = 600, int n_features = 20) {
        Dataset ds;
        ds.name = "High Dimensional";
        ds.n_classes = 3;
        ds.n_features = n_features;

        // Только первые 3 признака информативны
        std::normal_distribution<> informative(0.0, 1.0);
        std::uniform_real_distribution<> noise(-1.0, 1.0);

        for (int i = 0; i < n_samples; ++i) {
            std::vector<double> features(n_features);
            int label = i % 3;

            // Информативные признаки
            if (label == 0) {
                features[0] = informative(gen) - 1.5;
                features[1] = informative(gen) - 1.0;
                features[2] = informative(gen) * 0.5;
            } else if (label == 1) {
                features[0] = informative(gen) + 0.5;
                features[1] = informative(gen) + 1.0;
                features[2] = informative(gen) * 1.5;
            } else {
                features[0] = informative(gen) * 2.0;
                features[1] = informative(gen) * 0.3;
                features[2] = informative(gen) - 0.5;
            }

            // Шумовые признаки
            for (int j = 3; j < n_features; ++j) {
                features[j] = noise(gen);
            }

            ds.X.push_back(features);
            y.push_back(label);
        }

        return ds;
    }

    Dataset generate_imbalanced_data(int n_samples = 800) {
        Dataset ds;
        ds.name = "Imbalanced (90/10)";
        ds.n_classes = 2;
        ds.n_features = 4;

        // 90% class 0, 10% class 1
        std::normal_distribution<> class0_dist(0.0, 1.0);
        std::normal_distribution<> class1_dist(2.5, 0.8);

        for (int i = 0; i < n_samples; ++i) {
            std::vector<double> features(4);

            if (i < n_samples * 0.9) {  // Class 0 (90%)
                for (int j = 0; j < 4; ++j) {
                    features[j] = class0_dist(gen);
                }
                y.push_back(0);
            } else {  // Class 1 (10%)
                for (int j = 0; j < 4; ++j) {
                    features[j] = class1_dist(gen);
                }
                y.push_back(1);
            }

            ds.X.push_back(features);
        }

        return ds;
    }

    // ==================== ТЕСТОВЫЕ ФУНКЦИИ ====================

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    split_data(const std::vector<std::vector<double>>& X, const std::vector<int>& y,
               double test_size = 0.25) {
        std::vector<int> indices(X.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        int split_idx = static_cast<int>(X.size() * (1.0 - test_size));

        std::vector<std::vector<double>> X_train, X_test;
        std::vector<int> y_train, y_test;

        for (int i = 0; i < split_idx; ++i) {
            X_train.push_back(X[indices[i]]);
            y_train.push_back(y[indices[i]]);
        }
        for (int i = split_idx; i < X.size(); ++i) {
            X_test.push_back(X[indices[i]]);
            y_test.push_back(y[indices[i]]);
        }

        return {{X_train, X_test}, {y_train, y_test}};
    }

    TestMetrics evaluate_tree(const Dataset& ds, double ccp_alpha,
                             const std::string& criterion = "gini",
                             int max_depth = 20,
                             int min_samples_split = 2,
                             int min_samples_leaf = 1) {

        auto [X_split, y_split] = split_data(ds.X, ds.y, 0.25);
        auto& X_train = X_split[0];
        auto& X_test = X_split[1];
        auto& y_train = y_split[0];
        auto& y_test = y_split[1];

        DecisionTree tree(max_depth, min_samples_split, min_samples_leaf,
                         criterion, ccp_alpha);
        tree.fit(X_train, y_train);

        TestMetrics metrics;
        metrics.n_leaves = tree.get_n_leaves();

        // Calculate accuracies
        auto calculate_accuracy = [&](const auto& X, const auto& y) {
            int correct = 0;
            for (size_t i = 0; i < X.size(); ++i) {
                if (tree.predict(X[i]) == y[i]) correct++;
            }
            return 100.0 * correct / X.size();
        };

        metrics.train_accuracy = calculate_accuracy(X_train, y_train);
        metrics.test_accuracy = calculate_accuracy(X_test, y_test);

        return metrics;
    }

    void run_dataset_test(const Dataset& ds, const std::string& test_name) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << " DATASET: " << ds.name << std::endl;
        std::cout << " Features: " << ds.n_features << ", Classes: " << ds.n_classes;
        std::cout << ", Samples: " << ds.X.size() << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        std::vector<double> alphas = {0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5};

        std::cout << "\nAlpha     | Leaves | Train Acc | Test Acc  | Gap     | Effect\n";
        std::cout << "----------|--------|-----------|-----------|---------|--------\n";

        std::vector<std::pair<double, TestMetrics>> results;

        for (double alpha : alphas) {
            TestMetrics metrics = evaluate_tree(ds, alpha);
            results.push_back({alpha, metrics});

            double overfit_gap = metrics.train_accuracy - metrics.test_accuracy;

            std::cout << std::fixed << std::setprecision(3);
            std::cout << std::setw(9) << alpha << " | "
                      << std::setw(6) << metrics.n_leaves << " | "
                      << std::setw(9) << metrics.train_accuracy << "% | "
                      << std::setw(9) << metrics.test_accuracy << "% | "
                      << std::setw(7) << overfit_gap << "% | ";

            // Analyze pruning effect
            if (alpha == 0.0) {
                std::cout << "Baseline";
            } else {
                auto& prev = results[results.size()-2].second;
                double acc_change = metrics.test_accuracy - prev.test_accuracy;
                int leaves_change = prev.n_leaves - metrics.n_leaves;

                if (leaves_change > 0 && acc_change >= -1.0) {
                    std::cout << "GOOD (leaves-" << leaves_change << ", acc"
                              << (acc_change >= 0 ? "+" : "") << acc_change << "%)";
                } else if (leaves_change > 0 && acc_change < -5.0) {
                    std::cout << "OVER-PRUNED";
                } else if (leaves_change == 0) {
                    std::cout << "No effect";
                } else {
                    std::cout << "Mixed";
                }
            }
            std::cout << std::endl;
        }

        // Find best alpha (max test accuracy with reasonable size)
        auto best_it = std::max_element(results.begin(), results.end(),
            [](const auto& a, const auto& b) {
                double score_a = a.second.test_accuracy - 0.1 * a.second.n_leaves;
                double score_b = b.second.test_accuracy - 0.1 * b.second.n_leaves;
                return score_a < score_b;
            });

        std::cout << "\n[ANALYSIS] Best alpha: " << best_it->first
                  << " (Test Acc: " << best_it->second.test_accuracy
                  << "%, Leaves: " << best_it->second.n_leaves << ")\n";

        auto& baseline = results[0].second;
        auto& best = best_it->second;

        std::cout << "[IMPROVEMENT] Test accuracy: " << baseline.test_accuracy
                  << "% → " << best.test_accuracy << "% ";
        if (best.test_accuracy > baseline.test_accuracy) {
            std::cout << "(+" << (best.test_accuracy - baseline.test_accuracy) << "%) ";
        }
        std::cout << "| Leaves: " << baseline.n_leaves << " → " << best.n_leaves
                  << " (-" << (100.0 * (baseline.n_leaves - best.n_leaves) / baseline.n_leaves) << "%)";

        if (best.test_accuracy > baseline.test_accuracy && best.n_leaves < baseline.n_leaves) {
            std::cout << " ✓ EXCELLENT";
        } else if (best.test_accuracy >= baseline.test_accuracy - 2.0 && best.n_leaves < baseline.n_leaves * 0.5) {
            std::cout << " ✓ GOOD";
        }
        std::cout << std::endl;
    }

    void run_gini_vs_entropy_test(const Dataset& ds) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << " CRITERION COMPARISON: Gini vs Entropy" << std::endl;
        std::cout << " Dataset: " << ds.name << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        std::cout << "\nAlpha     | Criterion | Leaves | Train Acc | Test Acc  | Gap\n";
        std::cout << "----------|-----------|--------|-----------|-----------|-----\n";

        std::vector<double> alphas = {0.0, 0.01, 0.05, 0.1};

        for (double alpha : alphas) {
            for (const auto& criterion : {"gini", "entropy"}) {
                TestMetrics metrics = evaluate_tree(ds, alpha, criterion);
                double gap = metrics.train_accuracy - metrics.test_accuracy;

                std::cout << std::fixed << std::setprecision(3);
                std::cout << std::setw(9) << alpha << " | "
                          << std::setw(9) << criterion << " | "
                          << std::setw(6) << metrics.n_leaves << " | "
                          << std::setw(9) << metrics.train_accuracy << "% | "
                          << std::setw(9) << metrics.test_accuracy << "% | "
                          << std::setw(5) << gap << "%\n";
            }
        }
    }

    void run_depth_impact_test(const Dataset& ds) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << " MAX DEPTH IMPACT ON PRUNING" << std::endl;
        std::cout << " Dataset: " << ds.name << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        std::cout << "\nDepth | Alpha | Leaves | Train Acc | Test Acc  | Overfit\n";
        std::cout << "------|-------|--------|-----------|-----------|--------\n";

        std::vector<int> depths = {3, 5, 10, 20, 50};
        std::vector<double> alphas = {0.0, 0.01, 0.05};

        for (int depth : depths) {
            for (double alpha : alphas) {
                TestMetrics metrics = evaluate_tree(ds, alpha, "gini", depth);
                double overfit = metrics.train_accuracy - metrics.test_accuracy;

                std::cout << std::fixed << std::setprecision(1);
                std::cout << std::setw(5) << depth << " | "
                          << std::setw(5) << alpha << " | "
                          << std::setw(6) << metrics.n_leaves << " | "
                          << std::setw(9) << metrics.train_accuracy << "% | "
                          << std::setw(9) << metrics.test_accuracy << "% | "
                          << std::setw(6) << overfit << "%";

                if (alpha > 0.0 && overfit < 10.0) {
                    std::cout << " ✓";
                }
                std::cout << std::endl;
            }
            std::cout << "------|-------|--------|-----------|-----------|--------\n";
        }
    }

    void run_min_samples_test(const Dataset& ds) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << " MIN SAMPLES IMPACT" << std::endl;
        std::cout << " Dataset: " << ds.name << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        std::vector<std::pair<int, int>> samples_config = {
            {2, 1}, {5, 3}, {10, 5}, {20, 10}
        };

        std::cout << "\nMinSplit/MinLeaf | Alpha | Leaves | Train Acc | Test Acc\n";
        std::cout << "-----------------|-------|--------|-----------|----------\n";

        for (auto [min_split, min_leaf] : samples_config) {
            for (double alpha : {0.0, 0.05}) {
                DecisionTree tree(20, min_split, min_leaf, "gini", alpha);

                auto [X_split, y_split] = split_data(ds.X, ds.y, 0.25);
                tree.fit(X_split[0], y_split[0]);

                auto calculate_accuracy = [&](const auto& X, const auto& y) {
                    int correct = 0;
                    for (size_t i = 0; i < X.size(); ++i) {
                        if (tree.predict(X[i]) == y[i]) correct++;
                    }
                    return 100.0 * correct / X.size();
                };

                double train_acc = calculate_accuracy(X_split[0], y_split[0]);
                double test_acc = calculate_accuracy(X_split[1], y_split[1]);

                std::cout << std::setw(8) << min_split << "/"
                          << std::setw(7) << min_leaf << " | "
                          << std::setw(5) << alpha << " | "
                          << std::setw(6) << tree.get_n_leaves() << " | "
                          << std::fixed << std::setprecision(1)
                          << std::setw(9) << train_acc << "% | "
                          << std::setw(8) << test_acc << "%\n";
            }
        }
    }

    void run_all_tests() {
        std::cout << "EXTENDED DECISION TREE PRUNING TEST SUITE\n";
        std::cout << "==========================================\n";

        // 1. Основные тесты на разных датасетах
        std::vector<Dataset> datasets = {
            generate_linearly_separable(),
            generate_xor_pattern(),
            generate_multi_class_circles(),
            generate_high_dimensional(),
            generate_imbalanced_data()
        };

        for (size_t i = 0; i < datasets.size(); ++i) {
            std::cout << "\n\nTEST SET " << (i+1) << "/" << datasets.size();
            run_dataset_test(datasets[i], "Dataset_" + std::to_string(i+1));

            // Дополнительные тесты для первых 2 датасетов
            if (i < 2) {
                run_gini_vs_entropy_test(datasets[i]);
                run_depth_impact_test(datasets[i]);
                run_min_samples_test(datasets[i]);
            }
        }

        std::cout << "\n\n" << std::string(80, '=') << std::endl;
        std::cout << " ALL TESTS COMPLETED SUCCESSFULLY" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }
};

int main() {
    ExtendedPruningTester tester(42);
    tester.run_all_tests();
    return 0;
}