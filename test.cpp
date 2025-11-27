#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <functional>
#include "decision_tree.h"

std::uniform_real_distribution<double> dist(0.0, 1.0);

double calculate_accuracy(const DecisionTree& tree, const std::vector<DataPoint>& data) {
    int correct = 0;
    for (const auto& point : data) {
        if (tree.predict(point.features) == point.label) correct++;
    }
    return static_cast<double>(correct) / data.size();
}

int count_leaves(Node* node) {
    if (!node) return 0;
    if (dynamic_cast<LeafNode*>(node)) return 1;
    auto internal = dynamic_cast<InternalNode*>(node);
    return count_leaves(internal->left_child) + count_leaves(internal->right_child);
}

std::random_device rd;
std::mt19937 gen(rd());

// Функция для автоматического подбора alpha
double find_optimal_alpha(const std::vector<DataPoint>& train_data,
                         const std::vector<DataPoint>& val_data,
                         int max_depth = 20) {
    std::vector<double> candidate_alphas = {0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02};
    double best_alpha = 0.0;
    double best_score = -1.0;

    std::cout << "Alpha Search: ";
    for (double alpha : candidate_alphas) {
        DecisionTree tree(max_depth, 2, 1, "gini", alpha);
        tree.fit(train_data);
        double score = calculate_accuracy(tree, val_data);

        std::cout << alpha << "(" << score << ") ";

        if (score > best_score) {
            best_score = score;
            best_alpha = alpha;
        }
    }
    std::cout << "-> Best: " << best_alpha << " (score: " << best_score << ")" << std::endl;

    return best_alpha;
}

// 1. Умеренно зашумленные данные (реалистичный случай)
std::vector<DataPoint> generate_realistic_data(int samples, int features = 3, double noise = 0.2) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<DataPoint> data;
    for (int i = 0; i < samples; ++i) {
        std::vector<double> features_vec;
        double decision_value = 0.0;

        for (int j = 0; j < features; ++j) {
            double f = dist(gen);
            features_vec.push_back(f);
            decision_value += f * (j + 1) * 0.3;
        }

        // Нелинейная граница с умеренным шумом
        int true_label = (decision_value + 0.3 * std::sin(decision_value * 3) > 0.7) ? 1 : 0;
        if (dist(gen) < noise) {
            true_label = 1 - true_label;
        }

        data.emplace_back(features_vec, true_label);
    }
    return data;
}

// 2. Четко разделимые данные (прунинг не должен сильно менять)
std::vector<DataPoint> generate_clean_data(int samples) {
    std::vector<DataPoint> data;
    for (int i = 0; i < samples; ++i) {
        double x1 = dist(gen);
        double x2 = dist(gen);
        double x3 = dist(gen);
        // Четкая линейная граница
        int label = (x1 + 2*x2 - x3 > 1.2) ? 1 : 0;
        data.emplace_back(std::vector<double>{x1, x2, x3}, label);
    }
    return data;
}

// 3. Сильно зашумленные данные (прунинг должен помочь)
std::vector<DataPoint> generate_very_noisy_data(int samples, double noise = 0.35) {
    std::vector<DataPoint> data;
    for (int i = 0; i < samples; ++i) {
        double x1 = dist(gen);
        double x2 = dist(gen);
        double x3 = dist(gen);

        int true_label = (x1 * x1 + x2 * x2 - x3 > 0.3) ? 1 : 0;
        // Много шума
        if (dist(gen) < noise) true_label = 1 - true_label;

        data.emplace_back(std::vector<double>{x1, x2, x3}, true_label);
    }
    return data;
}




void run_alpha_sweep_test(const std::string& test_name,
                         const std::vector<DataPoint>& train_data,
                         const std::vector<DataPoint>& test_data,
                         const std::vector<DataPoint>& val_data) {

    std::cout << "\n🎯 " << test_name << std::endl;
    std::cout << "Data: train=" << train_data.size() << ", val=" << val_data.size()
              << ", test=" << test_data.size() << std::endl;

    // Автоподбор alpha
    double optimal_alpha = find_optimal_alpha(train_data, val_data, 15);

    // Сравнение с разными alpha
    std::vector<double> alphas = {0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02};

    std::cout << "\nAlpha\tLeaves\tTrain\tTest\tImprove\tStability" << std::endl;
    std::cout << "-----\t------\t----\t----\t-------\t---------" << std::endl;

    double baseline_test = 0.0;

    for (double alpha : alphas) {
        DecisionTree tree(15, 2, 1, "gini", alpha);
        tree.fit(train_data);

        int leaves = count_leaves(tree.get_root());
        double train_acc = calculate_accuracy(tree, train_data);
        double test_acc = calculate_accuracy(tree, test_data);

        if (alpha == 0.0) baseline_test = test_acc;

        double improvement = test_acc - baseline_test;
        double stability = (train_acc - test_acc); // Меньше = стабильнее

        std::cout << std::fixed << std::setprecision(3);
        std::cout << alpha << "\t" << leaves << "\t" << train_acc
                  << "\t" << test_acc << "\t"
                  << (improvement > 0 ? "+" : "") << improvement << "\t"
                  << stability;

        if (alpha == optimal_alpha) std::cout << " *OPTIMAL*";
        if (alpha == 0.0) std::cout << " *BASELINE*";
        std::cout << std::endl;
    }
}

void test_realistic_scenario() {
    auto train_data = generate_realistic_data(400, 3, 0.15);
    auto val_data = generate_realistic_data(100, 3, 0.15);
    auto test_data = generate_realistic_data(200, 3, 0.15);

    run_alpha_sweep_test("REALISTIC DATA (15% noise, 3 features)",
                        train_data, test_data, val_data);
}

void test_clean_data_scenario() {
    auto train_data = generate_clean_data(300);
    auto val_data = generate_clean_data(100);
    auto test_data = generate_clean_data(150);

    run_alpha_sweep_test("CLEAN DATA (Well-separable)",
                        train_data, test_data, val_data);
}

void test_noisy_data_scenario() {
    auto train_data = generate_very_noisy_data(350, 0.3);
    auto val_data = generate_very_noisy_data(100, 0.3);
    auto test_data = generate_very_noisy_data(175, 0.3);

    run_alpha_sweep_test("NOISY DATA (30% noise, prone to overfit)",
                        train_data, test_data, val_data);
}

void test_different_depths() {
    std::cout << "\n🌳 DEPTH SENSITIVITY ANALYSIS" << std::endl;

    auto train_data = generate_realistic_data(300, 3, 0.2);
    auto val_data = generate_realistic_data(100, 3, 0.2);
    auto test_data = generate_realistic_data(150, 3, 0.2);

    std::vector<int> depths = {5, 10, 15, 20, 30};

    std::cout << "Depth\tBestAlpha\tLeavesNoPrune\tLeavesPruned\tTestNoPrune\tTestPruned\tGain" << std::endl;
    std::cout << "-----\t---------\t------------\t-----------\t-----------\t----------\t----" << std::endl;

    for (int depth : depths) {
        // Без прунинга
        DecisionTree no_prune(depth, 2, 1, "gini", 0.0);
        no_prune.fit(train_data);
        double no_prune_test = calculate_accuracy(no_prune, test_data);
        int no_prune_leaves = count_leaves(no_prune.get_root());

        // С автоподбором alpha
        double best_alpha = find_optimal_alpha(train_data, val_data, depth);
        DecisionTree pruned(depth, 2, 1, "gini", best_alpha);
        pruned.fit(train_data);
        double pruned_test = calculate_accuracy(pruned, test_data);
        int pruned_leaves = count_leaves(pruned.get_root());

        double gain = pruned_test - no_prune_test;

        std::cout << depth << "\t" << best_alpha << "\t\t"
                  << no_prune_leaves << "\t\t" << pruned_leaves << "\t\t"
                  << std::fixed << std::setprecision(3) << no_prune_test << "\t"
                  << pruned_test << "\t" << (gain > 0 ? "+" : "") << gain << std::endl;
    }
}

void test_alpha_distribution() {
    std::cout << "\n📊 ALPHA DISTRIBUTION ACROSS MULTIPLE RUNS" << std::endl;

    std::vector<double> all_selected_alphas;
    int runs = 10;

    for (int i = 0; i < runs; ++i) {
        auto train_data = generate_realistic_data(300, 3, 0.15 + (i * 0.02));
        auto val_data = generate_realistic_data(100, 3, 0.15 + (i * 0.02));

        double best_alpha = find_optimal_alpha(train_data, val_data, 15);
        all_selected_alphas.push_back(best_alpha);
    }

    // Статистика по выбранным alpha
    std::sort(all_selected_alphas.begin(), all_selected_alphas.end());
    double median = all_selected_alphas[runs / 2];
    double mean = std::accumulate(all_selected_alphas.begin(), all_selected_alphas.end(), 0.0) / runs;

    std::cout << "Alpha Statistics over " << runs << " runs:" << std::endl;
    std::cout << "Min: " << *std::min_element(all_selected_alphas.begin(), all_selected_alphas.end()) << std::endl;
    std::cout << "Max: " << *std::max_element(all_selected_alphas.begin(), all_selected_alphas.end()) << std::endl;
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Median: " << median << std::endl;
    std::cout << "Typical sklearn range: 0.0001 - 0.02" << std::endl;

    // Проверка что alpha в разумном диапазоне
    bool in_sklearn_range = (median >= 0.0001 && median <= 0.02);
    std::cout << "In sklearn range: " << (in_sklearn_range ? "✅ YES" : "❌ NO") << std::endl;
}

void summary_analysis() {
    std::cout << "\n📈 FINAL ANALYSIS" << std::endl;
    std::cout << "=================" << std::endl;
    std::cout << "✅ Alpha Range: Should be 0.0001 - 0.02 (sklearn-compatible)" << std::endl;
    std::cout << "✅ Stability: Pruning should reduce overfitting (train-test gap)" << std::endl;
    std::cout << "✅ Improvement: Should improve or maintain test accuracy" << std::endl;
    std::cout << "✅ Reasonable Reduction: 20-60% leaves removed, not 90%" << std::endl;
    std::cout << "🎯 Goal: Alpha selection should be consistent and meaningful" << std::endl;
}

int main() {
    std::cout << "🧪 COMPREHENSIVE ALPHA VALIDATION TEST SUITE" << std::endl;
    std::cout << "Testing alpha range consistency and pruning effectiveness" << std::endl;
    std::cout << "==========================================================" << std::endl;

    test_realistic_scenario();
    test_clean_data_scenario();
    test_noisy_data_scenario();
    test_different_depths();
    test_alpha_distribution();
    summary_analysis();

    std::cout << "\n🎉 ALPHA VALIDATION COMPLETED!" << std::endl;
    std::cout << "Check if alpha values are sklearn-compatible and pruning is meaningful." << std::endl;

    return 0;
}