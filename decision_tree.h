#include <vector>
#include <memory>
#include <ranges>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
struct DataPoint {
    std::vector<double> features;
    int label;

    DataPoint(const std::vector<double>& features, int label) : features(features), label(label) {}
};

class Node {
protected:
    double node_error;
    int sample_count;
    int majority_class;
    std::unordered_map<int, double> class_probabilities;
public:
    Node(double node_error, int sample_count, int majority_class, const std::unordered_map<int, double>& class_probabilities) :
    node_error(node_error), sample_count(sample_count), majority_class(majority_class), class_probabilities(class_probabilities) {}
    double get_node_error() const { return node_error; }        // ← с реализацией
    int get_sample_count() const { return sample_count; }       // ← с реализацией
    int get_majority_class() const { return majority_class; }
    const std::unordered_map<int, double>& get_class_probabilities() const { return class_probabilities; }
    virtual ~Node() = default;
    virtual int predict(const std::vector<double>& features) const = 0;
    virtual std::unordered_map<int, double> predict_proba(const std::vector<double>& features) const = 0;
};

class LeafNode : public Node {
public:
    LeafNode(const std::unordered_map<int, double>& class_probabilities, int sample_count, double node_error) :
    Node(node_error, sample_count, 0, class_probabilities) {
        double max_prob = 0.0;
        for (const auto& [class_label, prob] : class_probabilities) {
            if (prob > max_prob) {
                majority_class = class_label;
                max_prob = prob;
            }
        }
    }

    std::unordered_map<int, double> predict_proba(const std::vector<double>& features) const override{
        return class_probabilities;
    }

    int predict(const std::vector<double>& features) const override {
        return majority_class;
    }
};

class InternalNode : public Node {
private:
    int feature_index;
    double threshold;
public:
    Node* left_child;
    Node* right_child;

    InternalNode(int feature_index, double threshold, Node* left_child, Node* right_child,
        int majority_class, const std::unordered_map<int, double>& class_probabilities, int sample_count, double node_error) :
    Node(node_error, sample_count, majority_class, class_probabilities), feature_index(feature_index),
    threshold(threshold), left_child(left_child), right_child(right_child) {}

    Node* get_left_child() const { return left_child; }
    Node* get_right_child() const { return right_child; }

    int get_feature_index() const {return feature_index;}
    double get_threshold() const {return threshold;}

    int predict(const std::vector<double>& features) const override {
        if (features[feature_index] <= threshold) {
            return left_child->predict(features);
        } else {
            return right_child->predict(features);
        }
    }

    std::unordered_map<int, double> predict_proba(const std::vector<double>& features) const override {
        if (features[feature_index] <= threshold) {
            return left_child->predict_proba(features);
        } else {
            return right_child->predict_proba(features);
        }
    }

    ~InternalNode() {
        delete left_child;
        delete right_child;
    }
};

enum class SplitCriterion { ENTROPY, GINI };

class DecisionTree {
private:
    Node* root = nullptr;
    int max_depth = 32;
    int min_samples_split = 2;
    int min_samples_leaf = 1;
    std::vector<double> feature_importances;
    SplitCriterion criterion = SplitCriterion::ENTROPY;
    double ccp_alpha = 0.0;


    struct SplitInfo {
        int feature_index;
        double threshold;
        double information_gain;
        std::vector<int> left_indices;
        std::vector<int> right_indices;

        SplitInfo() : feature_index(-1), threshold(0.0), information_gain(-1.0) {}
    };

    SplitInfo find_best_split(const std::vector<DataPoint>& data, const std::vector<int>& indices) {
        SplitInfo best_split;

        int num_features = data[0].features.size();
        for (int feature_index = 0; feature_index < num_features; ++feature_index) {
            double threshold = 0.0;

            std::vector<double> feature_values;
            for (int idx : indices) {
                feature_values.push_back(data[idx].features[feature_index]);
            }

            std::sort(feature_values.begin(), feature_values.end());

            for (int i = 0; i < feature_values.size()-1; ++i) {
                threshold = ((static_cast<double>(feature_values[i] + feature_values[i+1])) / 2);
                auto [left_indices, right_indices] = split_data(data, indices, feature_index, threshold);
                double gain = calculate_impurity_gain(data, indices, left_indices, right_indices);
                if (gain > best_split.information_gain) {
                    best_split.feature_index = feature_index;
                    best_split.threshold = threshold;
                    best_split.information_gain = gain;
                    best_split.left_indices = left_indices;
                    best_split.right_indices = right_indices;
                }
            }
        }
        return best_split;
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

    static std::unordered_map<int, double> calculate_probabilities(const std::vector<DataPoint>& data, const std::vector<int>& indices) {
        if (indices.empty()) return {{-1, 1.0}};

        int total = indices.size();
        std::unordered_map<int, int> class_counts;

        std::unordered_map<int, double> class_probabilities;

        for (int idx : indices) {
            class_counts[data[idx].label]++;
        }

        for (const auto& [class_label, count] : class_counts) {
            class_probabilities[class_label] = (static_cast<double>(count)/total);
        }

        return class_probabilities;
    }

    static double calculate_gini(const std::vector<DataPoint>& data, const std::vector<int>& indices) {
        double gini = 1.0;
        auto probabilities = calculate_probabilities(data, indices);
        for (const auto& [label, prob] : probabilities) {
            gini -= prob * prob;
        }
        return gini;
    }

    static double calculate_entropy(const std::vector<DataPoint>& data, const std::vector<int>& indices) {
        double entropy = 0.0;
        std::unordered_map<int, double> labels_probabilities = calculate_probabilities(data, indices);
        for (const auto& [label, probability] : labels_probabilities) {
            if (probability != 0.0) {
                entropy -= probability * std::log2(probability);
            } else {
                entropy -= 0.0;
            }
        }
        return entropy;
    }

    static double calculate_gini_gain(const std::vector<DataPoint>& data, const std::vector<int>& parent_indices,
                const std::vector<int>& left_indices, const std::vector<int>& right_indices) {
        if (parent_indices.size() == 0) return 0.0;
        double parent_gini = calculate_gini(data, parent_indices);
        double left_gini= calculate_gini(data, left_indices);
        double right_gini = calculate_gini(data, right_indices);
        return parent_gini - ((static_cast<double>(left_indices.size()) / parent_indices.size())*left_gini + (static_cast<double>(right_indices.size()) / parent_indices.size())*right_gini);
    }

    static double calculate_information_gain(const std::vector<DataPoint>& data, const std::vector<int>& parent_indices,
                const std::vector<int>& left_indices, const std::vector<int>& right_indices) {
        if (parent_indices.size() == 0) return 0.0;
        double parent_entropy = calculate_entropy(data, parent_indices);
        double left_entropy = calculate_entropy(data, left_indices);
        double right_entropy = calculate_entropy(data, right_indices);
        return parent_entropy - ((static_cast<double>(left_indices.size()) / parent_indices.size())*left_entropy + (static_cast<double>(right_indices.size()) / parent_indices.size())*right_entropy);
    }

    double calculate_impurity_gain(const std::vector<DataPoint>& data, const std::vector<int>& parent_indices,
                const std::vector<int>& left_indices, const std::vector<int>& right_indices) {
        if (criterion == SplitCriterion::ENTROPY) {
            return calculate_information_gain(data, parent_indices, left_indices, right_indices);
        } else {
            return calculate_gini_gain(data, parent_indices, left_indices, right_indices);
        }
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

    int get_majority_class_in_node(const std::vector<DataPoint>& data, const std::vector<int>& indices) {
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

    // double calculate_node_accuracy(const Node* node, const std::vector<DataPoint>& data, const std::vector<int>& indices) {
    //     if (node == nullptr or indices.empty()) {
    //         return 0.0;
    //     }
    //     int total_samples = indices.size();
    //     int correct_predictions = 0;
    //     for (int idx : indices) {
    //         if (node->predict(data[idx].features) == data[idx].label) {
    //             correct_predictions++;
    //         }
    //     }
    //     return static_cast<double>(correct_predictions)/total_samples;
    // }

    double calculate_node_error(const Node* node, const std::vector<DataPoint>& data,
                          const std::vector<int>& indices, int total_training_samples) {
        if (node == nullptr || indices.empty() || total_training_samples == 0) {
            return 0.0;
        }

        double impurity = 0.0;
        if (criterion == SplitCriterion::ENTROPY) {
            impurity = calculate_entropy(data, indices);
        } else {
            impurity = calculate_gini(data, indices);
        }

        double weight = static_cast<double>(indices.size()) / total_training_samples;

        return weight * impurity;
    }

    std::vector<int> get_indices_for_node(Node* node, const std::vector<DataPoint>& data,
                                    const std::vector<int>& parent_indices) {
        if (node == nullptr) {
            return {};
        }
        return parent_indices;
    }

    std::pair<double, int> calculate_tree_error(Node* node) {
        if (dynamic_cast<LeafNode*>(node) != nullptr) {
            return {node->get_node_error(), 1};
        }

        if (dynamic_cast<InternalNode*>(node) != nullptr) {
            InternalNode* internal_node = dynamic_cast<InternalNode*>(node);

            auto left_error = calculate_tree_error(internal_node->left_child);
            auto right_error = calculate_tree_error(internal_node->right_child);

            return {left_error.first + right_error.first, left_error.second + right_error.second};
        }
        return {0.0, 0};
    }

    std::pair<Node*, double> find_global_weakest_link(
    Node* node,
    Node* current_best_node = nullptr,
    double current_min_alpha = std::numeric_limits<double>::infinity()) {
        if (node == nullptr || dynamic_cast<LeafNode*>(node) != nullptr) {
            return {current_best_node, current_min_alpha};
        }

        if (dynamic_cast<InternalNode*>(node) != nullptr) {
            InternalNode* internal_node = dynamic_cast<InternalNode*>(node);

            double R_t = node->get_node_error();
            auto [R_Tt, T_t] = calculate_tree_error(node);

            if (T_t <= 1) {
            } else {
                double alpha = (R_t - R_Tt) / (T_t - 1);

                if (alpha >= 0 && alpha < current_min_alpha) {
                    current_min_alpha = alpha;
                    current_best_node = node;
                }
            }

            auto [left_candidate, left_alpha] = find_global_weakest_link(
                internal_node->get_left_child(), current_best_node, current_min_alpha);

            auto [right_candidate, right_alpha] = find_global_weakest_link(
                internal_node->get_right_child(), current_best_node, current_min_alpha);

            if (left_alpha < current_min_alpha && left_alpha <= right_alpha) {
                return {left_candidate, left_alpha};
            } else if (right_alpha < current_min_alpha && right_alpha <= left_alpha) {
                return {right_candidate, right_alpha};
            } else {
                return {current_best_node, current_min_alpha};
            }
        }

        return {current_best_node, current_min_alpha};
    }

    bool is_leaf_node(Node* node) {
        return (dynamic_cast<LeafNode*>(node) != nullptr);
    }

    void cost_complexity_prune(double ccp_alpha) {
        int iteration = 0;
        const int MAX_ITERATIONS = 1024;

        while (!is_leaf_node(root) && iteration < MAX_ITERATIONS) {
            iteration++;
            auto [weakest_node, alpha] = find_global_weakest_link(root);

            if (weakest_node == nullptr || alpha == std::numeric_limits<double>::infinity() || alpha > ccp_alpha) {
                break;
            }

            auto [parent, is_left] = find_parent(root, weakest_node);

            prune_node_to_leaf(static_cast<InternalNode*>(weakest_node), parent, is_left);
        }
    }

    void prune_node_to_leaf(InternalNode* node_to_prune,
        Node* parent,
        bool is_left_child) {

        int sample_count = node_to_prune->get_sample_count();
        double node_error = node_to_prune->get_node_error();
        auto class_probabilities = node_to_prune->get_class_probabilities();

        LeafNode* new_leaf = new LeafNode(class_probabilities, sample_count, node_error);

        if (parent != nullptr) {
            InternalNode* parent_internal = static_cast<InternalNode*>(parent);

            if (is_left_child) {
                parent_internal->left_child = new_leaf;
            } else {
                parent_internal->right_child = new_leaf;
            }
        } else {
            root = new_leaf;
        }
        node_to_prune->left_child = nullptr;
        node_to_prune->right_child = nullptr;

        delete node_to_prune;
    }

    int count_subtree_leaves(Node* node) {
        if (node == nullptr) {
            return 0;
        }
        if (dynamic_cast<LeafNode*>(node) != nullptr) {
            return 1;
        }

        auto internal_node = dynamic_cast<InternalNode*>(node);

        if (internal_node != nullptr) {
            return count_subtree_leaves(internal_node->get_left_child()) + count_subtree_leaves(internal_node->get_right_child());
        }
        return 0;
    }

    std::pair<Node*, bool> find_parent(Node* root, Node* target) {
        if (root == nullptr || target == nullptr) return {nullptr, false};

        if (dynamic_cast<InternalNode*>(root) != nullptr) {
            InternalNode* internal = static_cast<InternalNode*>(root);

            if (internal->left_child == target) return {internal, true};
            if (internal->right_child == target) return {internal, false};

            auto left_result = find_parent(internal->left_child, target);
            if (left_result.first != nullptr) return left_result;

            auto right_result = find_parent(internal->right_child, target);
            if (right_result.first != nullptr) return right_result;
        }

        return {nullptr, false};
    }

    Node* build_tree(const std::vector<DataPoint>& data,
        const std::vector<int>& indices,
        int depth, int total_samples) {
        auto probabilities = calculate_probabilities(data, indices);

        double impurity = 0.0;
        if (criterion == SplitCriterion::ENTROPY) {
            impurity = calculate_entropy(data, indices);
        } else {
            impurity = calculate_gini(data, indices);
        }

        double node_error = (static_cast<double>(indices.size()) / total_samples) * impurity;
        int majority_class = get_majority_class_in_node(data, indices);

        if (indices.size() < min_samples_split ||
        indices.empty() ||
        all_same_class(data, indices) ||
        depth >= max_depth) {

            return new LeafNode(probabilities, indices.size(), node_error);  // ← передаем node_error
        }

        auto best_split = find_best_split(data, indices);

        if (best_split.left_indices.size() < min_samples_leaf ||
        best_split.right_indices.size() < min_samples_leaf ||
        best_split.information_gain < 0.0) {

            return new LeafNode(probabilities, indices.size(), node_error);  // ← передаем node_error
        }

        int feature_index = best_split.feature_index;
        double threshold = best_split.threshold;

        if  (!feature_importances.empty()) {
            double weight = static_cast<double>(indices.size()) / total_samples;
            feature_importances[feature_index] += weight * best_split.information_gain;
        }

        auto left_child = build_tree(data, best_split.left_indices, depth + 1, total_samples);
        auto right_child = build_tree(data, best_split.right_indices, depth + 1, total_samples);

        return new InternalNode(feature_index, threshold, left_child, right_child, majority_class, probabilities, indices.size(), node_error);
    }

//=======================================================PUBLIC========================================================================//

public:
    DecisionTree(int max_depth = 32,
                 int min_samples_split = 5,
                 int min_samples_leaf = 2,
                 const std::string& string_criterion = "entropy",
                 double ccp_alpha = 0.0)
        : max_depth(max_depth),
          min_samples_split(min_samples_split),
          min_samples_leaf(min_samples_leaf),
          ccp_alpha(ccp_alpha > 0 ? ccp_alpha : 0.0)
        {

         if (ccp_alpha > 0) {
             this->ccp_alpha = ccp_alpha;
         }

        if (string_criterion == "entropy") {
            criterion = SplitCriterion::ENTROPY;
        } else {
            criterion = SplitCriterion::GINI;
        }
    }

    std::unordered_map<int, double> predict_proba(const std::vector<double>& features) const {
        if (!root) return {};
        return root->predict_proba(features);
    }


    Node* get_root() {
        return root;
    }

    const std::vector<double>& get_feature_importances() const {
        return feature_importances;
    }

    std::string get_criterion() const {
        if (criterion == SplitCriterion::ENTROPY) {
            return "entropy";
        } else {
            return "gini";
        }
    }

    void fit(const std::vector<DataPoint>& data) {
        if (data.empty()) {
            root = nullptr;
            feature_importances.clear();
            return;
        }

        std::vector<int> indices(data.size());
        for (int i = 0; i < data.size(); ++i) {
            indices[i] = i;
        }

        feature_importances = std::vector<double>(data[0].features.size(), 0.0);
        root = build_tree(data, indices, 0, data.size());

        double summary_importance = 0.0;
        for (const auto& importance : feature_importances) {
            summary_importance += importance;
        }

        if (summary_importance > 0.0) {
            for (size_t i = 0; i < feature_importances.size(); ++i) {
                feature_importances[i] /= summary_importance;
            }
        }

        if (ccp_alpha > 0.0) {
            cost_complexity_prune(ccp_alpha);
        }
    }

    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
        if (X.size() != y.size() || X.empty()) {
            std::cout << "Wrong size of data\n";
        }

        std::vector<DataPoint> data;
        data.reserve(X.size());

        for (size_t i = 0; i < X.size(); ++i) {
            data.emplace_back(X[i], y[i]);
        }

        fit(data);
    }

    int get_n_leaves() {
        return count_subtree_leaves(root);
    }

    void set_ccp_alpha(double new_alpha) {
        ccp_alpha = new_alpha;
    }

    int predict(const std::vector<double>& features) const {
        if (root == nullptr) return -1;
        return root->predict(features);
    }

    ~DecisionTree() {
        delete root;
    }
};

#ifndef DECISIONTREE_DECISION_TREE_H
#define DECISIONTREE_DECISION_TREE_H

#endif //DECISIONTREE_DECISION_TREE_H
