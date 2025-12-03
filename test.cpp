#include <chrono>
#include <random>
#include <iostream>
#include <vector>
#include <cmath>
#include "decision_tree.h"
#include <iomanip>

// ===================== DATA GENERATORS =====================

// 1. Classification: Concentric circles (non-linearly separable)
std::vector<DataPoint> generate_circles_dataset(int n_samples = 1000, double noise = 0.1) {
    std::vector<DataPoint> data;
    data.reserve(n_samples);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist_angle(0, 2 * M_PI);
    std::normal_distribution<double> dist_noise(0, noise);

    for (int i = 0; i < n_samples; ++i) {
        double angle = dist_angle(rng);
        double radius;
        int label;

        if (i % 2 == 0) {
            // Inner circle
            radius = 2.0 + dist_noise(rng);
            label = 0;
        } else {
            // Outer circle
            radius = 5.0 + dist_noise(rng);
            label = 1;
        }

        double x = radius * cos(angle);
        double y = radius * sin(angle);
        double z = dist_noise(rng);  // noisy feature
        data.emplace_back(std::vector<double>{x, y, z}, label);
    }

    return data;
}

// 2. Classification: XOR problem with noise
std::vector<DataPoint> generate_xor_dataset(int n_samples = 1000, double noise = 0.3) {
    std::vector<DataPoint> data;
    data.reserve(n_samples);

    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0, noise);

    for (int i = 0; i < n_samples; ++i) {
        double x, y, z;
        int label;

        int quadrant = i % 4;
        if (quadrant == 0) {
            x = 0.0 + dist(rng);
            y = 0.0 + dist(rng);
            label = 0;
        } else if (quadrant == 1) {
            x = 1.0 + dist(rng);
            y = 1.0 + dist(rng);
            label = 0;
        } else if (quadrant == 2) {
            x = 0.0 + dist(rng);
            y = 1.0 + dist(rng);
            label = 1;
        } else {
            x = 1.0 + dist(rng);
            y = 0.0 + dist(rng);
            label = 1;
        }
        z = dist(rng);  // noisy feature

        data.emplace_back(std::vector<double>{x, y, z}, label);
    }

    return data;
}

// 3. Classification: Spiral (3 classes)
std::vector<DataPoint> generate_spiral_dataset(int n_samples = 1000, int n_classes = 3, double noise = 0.1) {
    std::vector<DataPoint> data;
    data.reserve(n_samples);

    std::mt19937 rng(42);
    std::normal_distribution<double> dist_noise(0, noise);

    for (int i = 0; i < n_samples; ++i) {
        double r = double(i) / n_samples * 5.0;
        double angle = 2.0 * M_PI / n_classes * (i % n_classes) + r;

        double x = r * cos(angle) + dist_noise(rng);
        double y = r * sin(angle) + dist_noise(rng);
        double z = dist_noise(rng);  // noisy feature
        int label = i % n_classes;

        data.emplace_back(std::vector<double>{x, y, z}, label);
    }

    return data;
}

// 4. Classification: Moons dataset
std::vector<DataPoint> generate_moons_dataset(int n_samples = 1000, double noise = 0.1) {
    std::vector<DataPoint> data;
    data.reserve(n_samples);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist_uniform(0, 1);
    std::normal_distribution<double> dist_noise(0, noise);

    for (int i = 0; i < n_samples; ++i) {
        double t;
        int label;

        if (i % 2 == 0) {
            t = dist_uniform(rng) * M_PI;
            label = 0;
        } else {
            t = dist_uniform(rng) * M_PI + M_PI;
            label = 1;
        }

        double x = cos(t) + dist_noise(rng);
        double y = sin(t) + dist_noise(rng);
        double z = dist_noise(rng);

        if (i % 2 == 1) {
            x += 1.0;
        }

        data.emplace_back(std::vector<double>{x, y, z}, label);
    }

    return data;
}

// ===================== METRICS =====================

double calculate_accuracy(const DecisionTree& tree, const std::vector<DataPoint>& test_data) {
    int correct = 0;
    for (const auto& point : test_data) {
        if (tree.predict(point.features) == point.label) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / test_data.size();
}

void print_dataset_info(const std::string& name, const std::vector<DataPoint>& data) {
    std::cout << "  Dataset: " << name << "\n";
    std::cout << "  Size: " << data.size() << "\n";
    std::cout << "  Features: " << data[0].features.size() << "\n";

    std::unordered_map<int, int> class_counts;
    for (const auto& point : data) {
        class_counts[point.label]++;
    }
    std::cout << "  Classes: " << class_counts.size() << " (";
    for (const auto& [label, count] : class_counts) {
        std::cout << label << ":" << count << " ";
    }
    std::cout << ")\n";
}

// ===================== TEST 1: NO PRUNING =====================

void test_without_pruning() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "TEST 1: DECISION TREE WITHOUT PRUNING (ccp_alpha = 0.0)\n";
    std::cout << std::string(60, '=') << "\n";

    // Test datasets
    std::vector<std::pair<std::string, std::vector<DataPoint>>> datasets = {
        {"Concentric circles", generate_circles_dataset(1000)},
        {"XOR problem", generate_xor_dataset(1000)},
        {"Spiral (3 classes)", generate_spiral_dataset(1000, 3)},
        {"Moons", generate_moons_dataset(1000)}
    };

    for (const auto& [name, data] : datasets) {
        std::cout << "\n" << std::string(40, '-') << "\n";
        std::cout << "Dataset: " << name << "\n";
        std::cout << std::string(40, '-') << "\n";

        print_dataset_info(name, data);

        // Split into train/test (80/20)
        std::vector<DataPoint> train_data, test_data;
        for (size_t i = 0; i < data.size(); ++i) {
            if (i % 5 == 0) {  // every 5th to test
                test_data.push_back(data[i]);
            } else {
                train_data.push_back(data[i]);
            }
        }

        std::cout << "  Train: " << train_data.size() << ", Test: " << test_data.size() << "\n";

        // Tree settings (similar to sklearn defaults)
        DecisionTree tree(
            10,           // max_depth
            2,            // min_samples_split (sklearn default = 2)
            1,            // min_samples_leaf (sklearn default = 1)
            "gini",       // criterion
            0.0           // ccp_alpha (no pruning)
        );

        // Training
        auto train_start = std::chrono::high_resolution_clock::now();
        tree.fit(train_data);
        auto train_end = std::chrono::high_resolution_clock::now();
        auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);

        // Testing
        auto test_start = std::chrono::high_resolution_clock::now();
        double train_acc = calculate_accuracy(tree, train_data);
        double test_acc = calculate_accuracy(tree, test_data);
        auto test_end = std::chrono::high_resolution_clock::now();
        auto test_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_start);

        // Results
        std::cout << "\n  Results:\n";
        std::cout << "  Training time: " << train_duration.count() << " ms\n";
        std::cout << "  Testing time: " << test_duration.count() << " ms\n";
        std::cout << "  Number of leaves: " << tree.get_n_leaves() << "\n";
        std::cout << "  Train accuracy: " << train_acc * 100 << "%\n";
        std::cout << "  Test accuracy: " << test_acc * 100 << "%\n";

        // Feature importances
        const auto& importances = tree.get_feature_importances();
        std::cout << "  Feature importances: ";
        for (size_t i = 0; i < importances.size(); ++i) {
            std::cout << "F" << i << ": " << importances[i] << " ";
        }
        std::cout << "\n";
    }
}

// ===================== TEST 2: WITH PRUNING =====================

void test_with_pruning() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "TEST 2: DECISION TREE WITH PRUNING (different ccp_alpha values)\n";
    std::cout << std::string(60, '=') << "\n";

    // Use spiral dataset for pruning tests
    auto data = generate_spiral_dataset(1000, 3);

    // Split into train/validation/test (60/20/20)
    std::vector<DataPoint> train_data, val_data, test_data;
    for (size_t i = 0; i < data.size(); ++i) {
        if (i % 5 == 0) {
            test_data.push_back(data[i]);
        } else if (i % 5 == 1) {
            val_data.push_back(data[i]);
        } else {
            train_data.push_back(data[i]);
        }
    }

    std::cout << "\nDataset: Spiral (3 classes)\n";
    std::cout << "Sizes: train=" << train_data.size()
              << ", validation=" << val_data.size()
              << ", test=" << test_data.size() << "\n\n";

    // Different pruning strengths
    std::vector<double> alphas = {0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0};

    std::cout << "alpha | Leaves | Train Acc | Val Acc  | Test Acc | Train Time\n";
    std::cout << "------|--------|-----------|----------|----------|-----------\n";

    for (double alpha : alphas) {
        DecisionTree tree(
            20,           // max_depth (deeper tree for pruning)
            2,            // min_samples_split
            1,            // min_samples_leaf
            "gini",       // criterion
            alpha         // ccp_alpha
        );

        auto train_start = std::chrono::high_resolution_clock::now();
        tree.fit(train_data);
        auto train_end = std::chrono::high_resolution_clock::now();
        auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);

        double train_acc = calculate_accuracy(tree, train_data);
        double val_acc = calculate_accuracy(tree, val_data);
        double test_acc = calculate_accuracy(tree, test_data);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::setw(5) << alpha << " | "
                  << std::setw(6) << tree.get_n_leaves() << " | "
                  << std::setw(9) << train_acc * 100 << "% | "
                  << std::setw(8) << val_acc * 100 << "% | "
                  << std::setw(8) << test_acc * 100 << "% | "
                  << std::setw(10) << train_duration.count() << "ms\n";
    }
}

// ===================== TEST 3: EDGE CASES =====================

void test_edge_cases() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "TEST 3: EDGE CASES\n";
    std::cout << std::string(60, '=') << "\n";

    // 1. Empty data
    std::cout << "\n1. Empty data:\n";
    try {
        DecisionTree tree;
        tree.fit(std::vector<DataPoint>{});
        std::cout << "  OK: No crash on empty data\n";
    } catch (...) {
        std::cout << "  ERROR: Crashed on empty data\n";
    }

    // 2. All points same class
    std::cout << "\n2. All points same class:\n";
    std::vector<DataPoint> same_class_data;
    for (int i = 0; i < 100; ++i) {
        same_class_data.emplace_back(std::vector<double>{double(i), double(i)}, 0);
    }

    DecisionTree tree1;
    tree1.fit(same_class_data);
    std::cout << "  Leaves: " << tree1.get_n_leaves() << " (should be 1)\n";

    // 3. Prediction without training
    std::cout << "\n3. Prediction without training:\n";
    DecisionTree tree2;
    int prediction = tree2.predict({1.0, 2.0});
    std::cout << "  predict({1,2}) = " << prediction << " (should be -1)\n";

    // 4. Single feature
    std::cout << "\n4. Data with single feature:\n";
    std::vector<DataPoint> single_feature_data;
    for (int i = 0; i < 100; ++i) {
        single_feature_data.emplace_back(std::vector<double>{double(i)}, i % 2);
    }

    DecisionTree tree3;
    tree3.fit(single_feature_data);
    double acc = calculate_accuracy(tree3, single_feature_data);
    std::cout << "  Accuracy: " << acc * 100 << "%\n";

    // 5. Large number of features
    std::cout << "\n5. Data with 50 features (mostly noise):\n";
    std::vector<DataPoint> many_features_data;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0, 1);

    for (int i = 0; i < 200; ++i) {
        std::vector<double> features(50);
        for (int j = 0; j < 50; ++j) {
            features[j] = dist(rng);
        }
        // Only first feature matters
        int label = (features[0] > 0.5) ? 1 : 0;
        many_features_data.emplace_back(features, label);
    }

    DecisionTree tree4(5, 10, 5, "entropy");
    tree4.fit(many_features_data);

    const auto& importances = tree4.get_feature_importances();
    std::cout << "  Top 5 feature importances:\n";
    std::vector<size_t> indices(importances.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) { return importances[a] > importances[b]; });

    for (int i = 0; i < 5 && i < indices.size(); ++i) {
        std::cout << "    Feature " << indices[i] << ": " << importances[indices[i]]
                  << (indices[i] == 0 ? " (expected highest)" : "") << "\n";
    }
}

// ===================== TEST 4: SCALABILITY =====================

void test_scalability() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "TEST 4: SCALABILITY TEST\n";
    std::cout << std::string(60, '=') << "\n";

    std::vector<int> sample_sizes = {100, 500, 1000, 2000, 5000};

    std::cout << "\nSamples | Train Time | Leaves | Train Acc | Test Acc\n";
    std::cout << "--------|------------|--------|-----------|----------\n";

    for (int n_samples : sample_sizes) {
        auto data = generate_spiral_dataset(n_samples, 3);

        // Split
        std::vector<DataPoint> train_data, test_data;
        for (size_t i = 0; i < data.size(); ++i) {
            if (i % 5 == 0) {
                test_data.push_back(data[i]);
            } else {
                train_data.push_back(data[i]);
            }
        }

        DecisionTree tree(10, 2, 1, "gini", 0.0);

        auto train_start = std::chrono::high_resolution_clock::now();
        tree.fit(train_data);
        auto train_end = std::chrono::high_resolution_clock::now();
        auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);

        double train_acc = calculate_accuracy(tree, train_data);
        double test_acc = calculate_accuracy(tree, test_data);

        std::cout << std::setw(7) << n_samples << " | "
                  << std::setw(10) << train_duration.count() << "ms | "
                  << std::setw(6) << tree.get_n_leaves() << " | "
                  << std::setw(9) << train_acc * 100 << "% | "
                  << std::setw(8) << test_acc * 100 << "%\n";
    }
}

// ===================== MAIN =====================

int main() {
    std::cout << "DECISION TREE TEST SUITE\n";
    std::cout << "========================\n";

    // Run all tests
    test_without_pruning();
    test_with_pruning();
    test_edge_cases();
    test_scalability();

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "ALL TESTS COMPLETED!\n";
    std::cout << std::string(60, '=') << "\n";

    return 0;
}