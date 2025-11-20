#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <functional>
#include "decision_tree.h"

std::vector<DataPoint> load_iris_data(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    std::string line;

    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;
        int label;

        for (int i = 0; i < 4; ++i) {
            std::getline(ss, value, ',');
            features.push_back(std::stod(value));
        }

        std::getline(ss, value, ',');
        label = std::stoi(value);
        data.emplace_back(features, label);
    }
    return data;
}

// Функция для подсчета листьев в дереве
int count_tree_leaves(Node* node) {
    if (node == nullptr) return 0;
    if (dynamic_cast<LeafNode*>(node) != nullptr) return 1;

    if (auto internal = dynamic_cast<InternalNode*>(node)) {
        return count_tree_leaves(internal->get_left_child()) +
               count_tree_leaves(internal->get_right_child());
    }
    return 0;
}

// Функция для подсчета всех узлов
int count_tree_nodes(Node* node) {
    if (node == nullptr) return 0;
    if (dynamic_cast<LeafNode*>(node) != nullptr) return 1;

    if (auto internal = dynamic_cast<InternalNode*>(node)) {
        return 1 + count_tree_nodes(internal->get_left_child()) +
                   count_tree_nodes(internal->get_right_child());
    }
    return 0;
}

void run_pruning_test(const std::vector<DataPoint>& data, double ccp_alpha) {
    std::cout << "\n=== CCP_ALPHA = " << ccp_alpha << " ===" << std::endl;

    // Перемешиваем данные
    auto shuffled_data = data;
    std::mt19937 gen(42);
    std::shuffle(shuffled_data.begin(), shuffled_data.end(), gen);

    // Разделяем на train/test
    int train_size = shuffled_data.size() * 0.7;  // 70% train
    int val_size = shuffled_data.size() * 0.15;   // 15% validation
    std::vector<DataPoint> train_data(shuffled_data.begin(), shuffled_data.begin() + train_size);
    std::vector<DataPoint> val_data(shuffled_data.begin() + train_size, shuffled_data.begin() + train_size + val_size);
    std::vector<DataPoint> test_data(shuffled_data.begin() + train_size + val_size, shuffled_data.end());

    // Дерево БЕЗ прунинга (базовое)
    DecisionTree base_tree(10, 5, 2);
    base_tree.fit(train_data);

    // Дерево С прунингом
    DecisionTree pruned_tree(10, 5, 2);
    pruned_tree.set_ccp_alpha(ccp_alpha);
    pruned_tree.fit(train_data);

    // Сравниваем accuracy
    auto calculate_accuracy = [](DecisionTree& tree, const std::vector<DataPoint>& test_data) {
        int correct = 0;
        for (const auto& point : test_data) {
            if (tree.predict(point.features) == point.label) {
                correct++;
            }
        }
        return static_cast<double>(correct) / test_data.size();
    };

    double base_accuracy = calculate_accuracy(base_tree, test_data);
    double pruned_accuracy = calculate_accuracy(pruned_tree, test_data);

    // Сравниваем размеры деревьев
    int base_leaves = count_tree_leaves(base_tree.get_root());
    int pruned_leaves = count_tree_leaves(pruned_tree.get_root());
    int base_nodes = count_tree_nodes(base_tree.get_root());
    int pruned_nodes = count_tree_nodes(pruned_tree.get_root());

    std::cout << "Base Tree - Accuracy: " << base_accuracy << ", Leaves: " << base_leaves << ", Nodes: " << base_nodes << std::endl;
    std::cout << "Pruned Tree - Accuracy: " << pruned_accuracy << ", Leaves: " << pruned_leaves << ", Nodes: " << pruned_nodes << std::endl;
    std::cout << "Reduction - Leaves: " << (1.0 - static_cast<double>(pruned_leaves)/base_leaves) * 100 << "%, "
              << "Nodes: " << (1.0 - static_cast<double>(pruned_nodes)/base_nodes) * 100 << "%" << std::endl;

    // Feature importances сравнение
    auto base_imps = base_tree.get_feature_importances();
    auto pruned_imps = pruned_tree.get_feature_importances();
    std::cout << "Base Feature Importances: ";
    for (double imp : base_imps) std::cout << imp << " ";
    std::cout << "\nPruned Feature Importances: ";
    for (double imp : pruned_imps) std::cout << imp << " ";
    std::cout << std::endl;
}

int main() {
    std::cout << "MANUAL PRUNING TEST (CCP_ALPHA) - IRIS DATASET" << std::endl;
    std::cout << "==============================================" << std::endl;

    auto data = load_iris_data("iris_data.csv");
    std::cout << "Loaded " << data.size() << " samples" << std::endl;

    // Тестируем разные значения ccp_alpha
    std::vector<double> alphas = {0.0, 0.001, 0.005, 0.01, 0.02, 0.05};

    for (double alpha : alphas) {
        run_pruning_test(data, alpha);
    }

    return 0;
}