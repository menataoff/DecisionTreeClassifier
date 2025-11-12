//
// Created by myk1n on 12.11.2025.
//

#include<vector>
#include <memory>
#include <ranges>
#include <unordered_map>
#include <cmath>

struct DataPoint {
    std::vector<double> features;
    int label;

    DataPoint(const std::vector<double>& features, int label) : features(features), label(label) {}
};

class Node {
public:
    virtual ~Node() = default;
    virtual int predict(const std::vector<double>& features) const = 0;

};

class LeafNode : public Node {
private:
    int class_label;
public:
    LeafNode(int class_label) : class_label(class_label) {}
    int predict(const std::vector<double>& features) const override {
        return class_label;
    }
};

class InternalNode : public Node {
private:
    int feature_index;
    double threshold;
    Node* left_child;
    Node* right_child;
public:
    InternalNode(int feature_index, double threshold, Node* left_child, Node* right_child) :
    feature_index(feature_index), threshold(threshold), left_child(left_child), right_child(right_child) {}
    int predict(const std::vector<double>& features) const override {
        if (features[feature_index] <= threshold) {
            return left_child->predict(features);
        } else {
            return right_child->predict(features);
        }
    }
    ~InternalNode() {
        delete left_child;
        delete right_child;
    }
};

class DecisionTree {
private:
    Node* root;
    int max_depth;


    static std::vector<std::pair<int, double>> calculate_probabilities(const std::vector<DataPoint>& data, const std::vector<int>& indices) {
        int total = indices.size();
        std::unordered_map<int, int> class_counts;

        for (int idx : indices) {
            int label = data[idx].label;
            class_counts[label]++;
        }
        std::vector<std::pair<int, double>> labels_probabilities;
        for (const auto& [class_label, count] : class_counts) {
            labels_probabilities.push_back(std::pair<int, double>(class_label, (static_cast<double>(count)/total)));
        }
        return labels_probabilities;
    }

    static double calculate_entropy(const std::vector<DataPoint>& data, const std::vector<int>& indices) {
        double entropy = 0.0;
        std::vector<std::pair<int, double>> labels_probabilities = calculate_probabilities(data, indices);
        for (auto [label, probability] : labels_probabilities) {
            if (probability != 0.0) {
                entropy -= probability * std::log2(probability);
            } else {
                entropy -= 0.0;
            }
        }
        return entropy;
    }

    static double calculate_information_gain(const std::vector<DataPoint>& data, const std::vector<int>& parent_indices,
                const std::vector<int>& left_indices, const std::vector<int>& right_indices) {
        double info_gain = 0.0;
        double parent_entropy = calculate_entropy(data, parent_indices);
        double left_entropy = calculate_entropy(data, left_indices);
        double right_entropy = calculate_entropy(data, right_indices);
        return parent_entropy - ((static_cast<double>(left_indices.size()) / parent_indices.size())*left_entropy + (static_cast<double>(right_indices.size()) / parent_indices.size())*right_entropy);
    }



    bool all_same_class(const std::vector<DataPoint>& data, const std::vector<int>& indices) {
        if (indices.empty()) return true;
        int first_label = data[indices[0]].label;
        for (int idx : indices) {
            if (data[idx].label != first_label) {
                return false;
            }
        }
        return true;
    }

    int get_majority_class(const std::vector<DataPoint>& data, const std::vector<int>& indices) {
        if (indices.empty()) return 0;

        std::unordered_map<int, int> class_counts;

        for (int idx : indices) {
            int label = data[idx].label;
            class_counts[label]++;
        }

        int majority_class = 0;
        int max_count = 0;
        for (const auto& [class_label, count] : class_counts) {
            if (count > max_count) {
                max_count = count;
                majority_class = class_label;
            }
        }
        return majority_class;
    }

    Node* build_tree(const std::vector<DataPoint>& data,
        const std::vector<int>& indices,
        int depth) {

        int feature_index = 0;
        double threshold = 0.5;

        if (indices.empty()) {
            return new LeafNode(get_majority_class(data, indices));
        }

        if (all_same_class(data, indices)) return new LeafNode(data[indices[0]].label);

        if (depth >= max_depth) {
            return new LeafNode(get_majority_class(data, indices));
        }

        auto [left_indices, right_indices] = split_data(data, indices, feature_index, threshold);

        auto left_child = build_tree(data, left_indices, depth + 1);
        auto right_child = build_tree(data, right_indices, depth + 1);
        //TODO: Поиск лучшего разделения и рекурсия

        return new InternalNode(feature_index, threshold, left_child, right_child); //заглушка
    }




    std::pair<std::vector<int>, std::vector<int>> split_data(const std::vector<DataPoint>& data,
        const std::vector<int>& indices, int feature_index, double threshold) {

        std::vector<int> left_idx;
        std::vector<int> right_idx;

        for (int idx : indices) {
            if (data[idx].features[feature_index] <= threshold) {
                left_idx.push_back(idx);
            } else {
                right_idx.push_back(idx);
            }
        }

        return std::pair{left_idx, right_idx};
    }

public:
    DecisionTree() : root(nullptr), max_depth(32) {}
    DecisionTree(int max_depth) : root(nullptr), max_depth(max_depth) {}

    void fit(const std::vector<DataPoint>& data) {
        std::vector<int> indices(data.size());
        for (int i = 0; i < data.size(); ++i) {
            indices[i] = i;
        }

        root = build_tree(data, indices, 0);
    }

    int predict(const std::vector<double>& features) const {
        if (root == nullptr) return -1;
        return root->predict(features);
    }

    ~DecisionTree() {
        delete root;
    }
};


#include <random>
#include <chrono>

std::vector<DataPoint> generate_test_data(int data_size) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<DataPoint> data;
    data.reserve(data_size);

    std::uniform_real_distribution<double> feature_x1(0.0, 0.4);
    std::uniform_real_distribution<double> feature_x2(0.6, 1.0);
    std::uniform_real_distribution<double> feature_y(0.0, 1.0);
    std::uniform_int_distribution<int> choice(1, 2);

    for (int i = 0; i < data_size; ++i) {
        int random_choice = choice(gen);
        if (random_choice == 1) {
            data.push_back(DataPoint(std::vector<double> {feature_x1(gen), feature_y(gen)}, 0));
        } else {
            data.push_back(DataPoint(std::vector<double> {feature_x2(gen), feature_y(gen)}, 1));
        }
    }
    return data;
}

#ifndef DECISIONTREE_DECISION_TREE_H
#define DECISIONTREE_DECISION_TREE_H

#endif //DECISIONTREE_DECISION_TREE_H