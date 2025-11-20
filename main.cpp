#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include "decision_tree.h"
// Your decision tree code here...

class DecisionTreeTester {
private:
    static double calculate_accuracy(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels) {
        if (true_labels.size() != predicted_labels.size() || true_labels.empty()) {
            return 0.0;
        }

        int correct = 0;
        for (size_t i = 0; i < true_labels.size(); ++i) {
            if (true_labels[i] == predicted_labels[i]) {
                correct++;
            }
        }
        return static_cast<double>(correct) / true_labels.size();
    }

    static std::unordered_map<int, double> calculate_class_metrics(
        const std::vector<int>& true_labels,
        const std::vector<int>& predicted_labels,
        int class_label) {

        std::unordered_map<int, double> metrics;
        int true_positive = 0;
        int false_positive = 0;
        int false_negative = 0;
        int true_negative = 0;

        for (size_t i = 0; i < true_labels.size(); ++i) {
            if (predicted_labels[i] == class_label && true_labels[i] == class_label) {
                true_positive++;
            } else if (predicted_labels[i] == class_label && true_labels[i] != class_label) {
                false_positive++;
            } else if (predicted_labels[i] != class_label && true_labels[i] == class_label) {
                false_negative++;
            } else {
                true_negative++;
            }
        }

        double precision = (true_positive + false_positive > 0) ?
            static_cast<double>(true_positive) / (true_positive + false_positive) : 0.0;
        double recall = (true_positive + false_negative > 0) ?
            static_cast<double>(true_positive) / (true_positive + false_negative) : 0.0;
        double f1 = (precision + recall > 0) ?
            2 * precision * recall / (precision + recall) : 0.0;

        metrics['precision'] = precision;
        metrics['recall'] = recall;
        metrics['f1'] = f1;
        metrics['support'] = true_positive + false_negative;

        return metrics;
    }

public:
    // Test 1: Binary classification (your existing test)
    static void test_binary_classification() {
        std::cout << "=== Test 1: Binary Classification ===" << std::endl;

        auto train_data = generate_test_data(1000);
        auto test_data = generate_test_data(200);

        DecisionTree tree(5, 2, 1, "entropy");
        tree.fit(train_data);

        std::vector<int> true_labels;
        std::vector<int> predicted_labels;

        for (const auto& data_point : test_data) {
            true_labels.push_back(data_point.label);
            predicted_labels.push_back(tree.predict(data_point.features));
        }

        double accuracy = calculate_accuracy(true_labels, predicted_labels);
        std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

        auto feature_importances = tree.get_feature_importances();
        std::cout << "Feature importances: [";
        for (size_t i = 0; i < feature_importances.size(); ++i) {
            std::cout << feature_importances[i];
            if (i < feature_importances.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Test predict_proba
        std::cout << "\nTesting predict_proba for first 3 examples:" << std::endl;
        for (int i = 0; i < 3 && i < test_data.size(); ++i) {
            auto proba = tree.predict_proba(test_data[i].features);
            std::cout << "Example " << i << ": ";
            for (const auto& [label, prob] : proba) {
                std::cout << "Class " << label << "=" << prob * 100 << "%, ";
            }
            std::cout << "Predicted: " << tree.predict(test_data[i].features);
            std::cout << ", True: " << test_data[i].label << std::endl;
        }
        std::cout << std::endl;
    }

    // Test 2: Multiclass classification (3 classes)
    static void test_multiclass_classification() {
        std::cout << "=== Test 2: Multiclass Classification (3 classes) ===" << std::endl;

        std::vector<DataPoint> data;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        // Generate data for 3 classes
        for (int i = 0; i < 300; ++i) {
            double x = dist(gen);
            double y = dist(gen);

            if (x < 0.4 && y < 0.4) {
                data.push_back(DataPoint({x, y}, 0)); // Class 0 - bottom left
            } else if (x > 0.6 && y > 0.6) {
                data.push_back(DataPoint({x, y}, 1)); // Class 1 - top right
            } else {
                data.push_back(DataPoint({x, y}, 2)); // Class 2 - rest
            }
        }

        // Shuffle data
        std::shuffle(data.begin(), data.end(), gen);

        // Split into train and test
        size_t split_idx = data.size() * 0.8;
        std::vector<DataPoint> train_data(data.begin(), data.begin() + split_idx);
        std::vector<DataPoint> test_data(data.begin() + split_idx, data.end());

        DecisionTree tree(5, 2, 1, "gini");
        tree.fit(train_data);

        std::vector<int> true_labels;
        std::vector<int> predicted_labels;

        for (const auto& data_point : test_data) {
            true_labels.push_back(data_point.label);
            predicted_labels.push_back(tree.predict(data_point.features));
        }

        double accuracy = calculate_accuracy(true_labels, predicted_labels);
        std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

        // Per-class metrics
        std::cout << "\nPer-class metrics:" << std::endl;
        for (int class_label = 0; class_label < 3; ++class_label) {
            auto metrics = calculate_class_metrics(true_labels, predicted_labels, class_label);
            std::cout << "Class " << class_label << ": "
                      << "Precision=" << metrics['precision'] * 100 << "%, "
                      << "Recall=" << metrics['recall'] * 100 << "%, "
                      << "F1=" << metrics['f1'] * 100 << "%, "
                      << "Support=" << metrics['support'] << std::endl;
        }

        auto feature_importances = tree.get_feature_importances();
        std::cout << "\nFeature importances: [";
        for (size_t i = 0; i < feature_importances.size(); ++i) {
            std::cout << feature_importances[i];
            if (i < feature_importances.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << std::endl;
    }

    // Test 3: Simple deterministic data
    static void test_deterministic_data() {
        std::cout << "=== Test 3: Deterministic Data ===" << std::endl;

        std::vector<DataPoint> train_data = {
            DataPoint({1.0, 1.0}, 0),
            DataPoint({1.0, 2.0}, 0),
            DataPoint({2.0, 1.0}, 1),
            DataPoint({2.0, 2.0}, 1),
            DataPoint({3.0, 1.0}, 2),
            DataPoint({3.0, 2.0}, 2)
        };

        DecisionTree tree(10, 1, 1, "entropy");
        tree.fit(train_data);

        // Test on training data (should be 100% accuracy)
        int correct = 0;
        for (const auto& data_point : train_data) {
            if (tree.predict(data_point.features) == data_point.label) {
                correct++;
            }
        }

        std::cout << "Accuracy on training data: "
                  << (static_cast<double>(correct) / train_data.size()) * 100 << "%" << std::endl;

        // Test predict_proba
        std::cout << "\nTesting probabilities:" << std::endl;
        for (const auto& data_point : train_data) {
            auto proba = tree.predict_proba(data_point.features);
            std::cout << "Point [" << data_point.features[0] << ", " << data_point.features[1]
                      << "] -> Class " << data_point.label << ", Probabilities: ";
            for (const auto& [label, prob] : proba) {
                std::cout << "Class " << label << "=" << prob * 100 << "%, ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // Test 4: Edge cases
    static void test_edge_cases() {
        std::cout << "=== Test 4: Edge Cases ===" << std::endl;

        // Test with empty data
        std::cout << "1. Test with empty data:" << std::endl;
        DecisionTree tree1;
        std::vector<DataPoint> empty_data;
        tree1.fit(empty_data);
        std::vector<double> test_features = {1.0, 2.0};
        int prediction = tree1.predict(test_features);
        std::cout << "Prediction for empty model: " << prediction << " (expected -1)" << std::endl;

        // Test with single class
        std::cout << "\n2. Test with single class:" << std::endl;
        std::vector<DataPoint> single_class_data = {
            DataPoint({1.0, 1.0}, 0),
            DataPoint({1.5, 1.5}, 0),
            DataPoint({2.0, 2.0}, 0)
        };
        DecisionTree tree2;
        tree2.fit(single_class_data);

        for (const auto& data_point : single_class_data) {
            auto proba = tree2.predict_proba(data_point.features);
            std::cout << "Point [" << data_point.features[0] << ", " << data_point.features[1]
                      << "] -> Probabilities: ";
            for (const auto& [label, prob] : proba) {
                std::cout << "Class " << label << "=" << prob * 100 << "%, ";
            }
            std::cout << std::endl;
        }

        // Test with minimal data
        std::cout << "\n3. Test with minimal data:" << std::endl;
        std::vector<DataPoint> min_data = {
            DataPoint({1.0, 1.0}, 0),
            DataPoint({2.0, 2.0}, 1)
        };
        DecisionTree tree3(1, 2, 1);
        tree3.fit(min_data);

        for (const auto& data_point : min_data) {
            int pred = tree3.predict(data_point.features);
            std::cout << "Point [" << data_point.features[0] << ", " << data_point.features[1]
                      << "] -> Predicted: " << pred << ", True: " << data_point.label << std::endl;
        }
        std::cout << std::endl;
    }

    // Test 5: Compare split criteria
    static void test_split_criteria() {
        std::cout << "=== Test 5: Split Criteria Comparison ===" << std::endl;

        auto data = generate_test_data(500);

        // Split into train and test
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(data.begin(), data.end(), gen);
        size_t split_idx = data.size() * 0.7;
        std::vector<DataPoint> train_data(data.begin(), data.begin() + split_idx);
        std::vector<DataPoint> test_data(data.begin() + split_idx, data.end());

        // Test with entropy
        DecisionTree tree_entropy(5, 2, 1, "entropy");
        tree_entropy.fit(train_data);

        // Test with Gini
        DecisionTree tree_gini(5, 2, 1, "gini");
        tree_gini.fit(train_data);

        std::vector<int> true_labels;
        std::vector<int> pred_entropy;
        std::vector<int> pred_gini;

        for (const auto& data_point : test_data) {
            true_labels.push_back(data_point.label);
            pred_entropy.push_back(tree_entropy.predict(data_point.features));
            pred_gini.push_back(tree_gini.predict(data_point.features));
        }

        double accuracy_entropy = calculate_accuracy(true_labels, pred_entropy);
        double accuracy_gini = calculate_accuracy(true_labels, pred_gini);

        std::cout << "Accuracy with entropy criterion: " << accuracy_entropy * 100 << "%" << std::endl;
        std::cout << "Accuracy with Gini criterion: " << accuracy_gini * 100 << "%" << std::endl;

        auto importance_entropy = tree_entropy.get_feature_importances();
        auto importance_gini = tree_gini.get_feature_importances();

        std::cout << "\nFeature importances (entropy): [";
        for (size_t i = 0; i < importance_entropy.size(); ++i) {
            std::cout << importance_entropy[i];
            if (i < importance_entropy.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "Feature importances (Gini): [";
        for (size_t i = 0; i < importance_gini.size(); ++i) {
            std::cout << importance_gini[i];
            if (i < importance_gini.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << std::endl;
    }

    // Run all tests
    static void run_all_tests() {
        test_binary_classification();
        test_multiclass_classification();
        test_deterministic_data();
        test_edge_cases();
        test_split_criteria();

        std::cout << "=== All tests completed ===" << std::endl;
    }
};

int main() {
    std::cout << "Running Decision Tree tests..." << std::endl;
    DecisionTreeTester::run_all_tests();
    return 0;
}
