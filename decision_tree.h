//
// Created by myk1n on 12.11.2025.
//

#include <vector>
#include <memory>
#include <ranges>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <queue>
#include <random>

struct DataPoint {
    std::vector<double> features;
    int label;

    DataPoint(const std::vector<double>& features, int label) : features(features), label(label) {}
};

class Node {
public:
    virtual int get_sample_count() const = 0;
    virtual ~Node() = default;
    virtual int predict(const std::vector<double>& features) const = 0;
    virtual std::unordered_map<int, double> predict_proba(const std::vector<double>& features) const = 0;
};

class LeafNode : public Node {
private:
    std::unordered_map<int, double> class_probabilities;
    int majority_class;
    int sample_count;

public:
    LeafNode(std::unordered_map<int, double> class_probabilities, int sample_count) : class_probabilities(class_probabilities), sample_count(sample_count) {
        majority_class = 0;
        double max_prob = 0.0;
        for (const auto& [class_label, prob] : class_probabilities) {
            if (prob > max_prob) {
                majority_class = class_label;
                max_prob = prob;
            }
        }
    }

    int get_sample_count() const override {return sample_count;}

    std::unordered_map<int, double> get_class_probabilities() const {
        return class_probabilities;
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
    int sample_count;
public:
    Node* left_child;
    Node* right_child;

    InternalNode(int feature_index, double threshold, Node* left_child, Node* right_child, int sample_count) :
    feature_index(feature_index), threshold(threshold), left_child(left_child), right_child(right_child), sample_count(sample_count) {}

    Node* get_left_child() const { return left_child; }
    Node* get_right_child() const { return right_child; }

    int get_feature_index() const {return feature_index;}
    double get_threshold() const {return threshold;}
    int get_sample_count() const override {return sample_count;}

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
    bool auto_prune = true;
    double validation_size = 0.15;


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


    double calculate_node_accuracy(Node* node, const std::vector<DataPoint>& data, const std::vector<int>& indices) {
        if (node == nullptr or indices.empty()) {
            return 0.0;
        }

        int total_samples = indices.size();
        int correct_predictions = 0;

        for (int idx : indices) {
            if (node->predict(data[idx].features) == data[idx].label) {
                correct_predictions++;
            }
        }

        return static_cast<double>(correct_predictions)/total_samples;
    }

    double calculate_node_error(Node* node, const std::vector<DataPoint>& data, const std::vector<int>& indices) {
        return (1 - calculate_node_accuracy(node, data, indices));
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

    double calculate_subtree_error(Node* node, const std::vector<DataPoint>& error_data, const std::vector<int>& error_indices, int total_training_samples) {
        if (node == nullptr or total_training_samples == 0) {
            return 0.0;
        }

        if (dynamic_cast<LeafNode*>(node) != nullptr) {
            auto leaf_node = dynamic_cast<LeafNode*>(node);
            double error_rate = calculate_node_error(leaf_node, error_data, error_indices);
            double weight = static_cast<double>(leaf_node->get_sample_count()) / total_training_samples;
            return weight*error_rate;
        }

        if (dynamic_cast<InternalNode*>(node) != nullptr) {
            auto internal_node = dynamic_cast<InternalNode*>(node);
            double weighted_left_error_rate = calculate_subtree_error(internal_node->get_left_child(), error_data, error_indices, total_training_samples);
            double weighted_right_error_rate = calculate_subtree_error(internal_node->get_right_child(), error_data, error_indices, total_training_samples);
            return weighted_left_error_rate + weighted_right_error_rate;
        }

        return 0.0;
    }

    double calculate_prune_metric(Node* node, const std::vector<DataPoint>& error_data, const std::vector<int>& error_indices, int total_training_samples) {
        if (dynamic_cast<LeafNode*>(node) != nullptr) {
            return std::numeric_limits<double>::infinity();
        }

        int T_t = count_subtree_leaves(node);
        double weighted_subtree_error = calculate_subtree_error(node, error_data, error_indices, total_training_samples);

        int sample_count_if_leaf = node->get_sample_count();
        double error_if_leaf = calculate_node_error(node, error_data, error_indices);
        double weighted_leaf_error = (static_cast<double>(sample_count_if_leaf) / total_training_samples) * error_if_leaf;

        return (weighted_leaf_error - weighted_subtree_error) / (T_t - 1);
    }

    void find_best_weakest_links(Node* node, Node*& best_candidate, double& min_g_value,
        const std::vector<DataPoint>& error_data, const std::vector<int>& error_indices, int total_training_samples) {

        if (node == nullptr) return;

        if (dynamic_cast<InternalNode*>(node) != nullptr) {
            InternalNode* internal_node = dynamic_cast<InternalNode*>(node) ;

            find_best_weakest_links(internal_node->get_left_child(), best_candidate, min_g_value, error_data, error_indices, total_training_samples);
            find_best_weakest_links(internal_node->get_right_child(), best_candidate, min_g_value, error_data, error_indices, total_training_samples);

            double g_value = calculate_prune_metric(internal_node, error_data, error_indices, total_training_samples);

            if (g_value <= min_g_value) {
                best_candidate = internal_node;
                min_g_value = g_value;
            }
        }
    }

    Node* find_weakest_links(Node* node, const std::vector<DataPoint>& error_data,
        const std::vector<int>& error_indices, int total_training_samples) {
        Node* best_candidate = nullptr;
        double min_g_value = std::numeric_limits<double>::infinity();
        find_best_weakest_links(node, best_candidate, min_g_value, error_data, error_indices, total_training_samples);
        return best_candidate;
    }

    struct NodeWithParent {
        Node* node;
        Node* parent;
        bool is_left_child;

        NodeWithParent(Node* n, Node* p, bool is_left) : node(n), parent(p), is_left_child(is_left) {}
    };

    void prune_node_to_leaf(InternalNode* node_to_prune, Node* parent, bool is_left_child, const std::vector<DataPoint>& error_data, const std::vector<int>& error_indices) {
        int sample_count = node_to_prune->get_sample_count();
        auto class_probabilities = calculate_probabilities(error_data, error_indices);

        LeafNode* new_leaf = new LeafNode(class_probabilities, sample_count);

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

        delete node_to_prune;
    }

    void cost_complexity_prune(double alpha, const std::vector<DataPoint>& data, const std::vector<int>& indices, int total_training_samples) {
        if (root == nullptr || alpha < 0 || data.empty() || indices.empty()) return;

        std::queue<NodeWithParent> nodes_queue;
        nodes_queue.push(NodeWithParent(root, nullptr, false));

        while (!nodes_queue.empty()) {
            NodeWithParent current = nodes_queue.front();
            nodes_queue.pop();
            if (dynamic_cast<InternalNode*>(current.node) != nullptr) {
                InternalNode* internal_node = dynamic_cast<InternalNode*>(current.node);
                double g_value = calculate_prune_metric(internal_node, data, indices, total_training_samples);

                if (g_value <= alpha) {
                    prune_node_to_leaf(internal_node, current.parent, current.is_left_child, data, indices);
                } else {
                    nodes_queue.push(NodeWithParent(internal_node->get_left_child(), internal_node, true));
                    nodes_queue.push(NodeWithParent(internal_node->get_right_child(), internal_node, false));
                }
            }
        }
    }

    void manual_pruning(const std::vector<DataPoint>& data, double alpha) {
        std::vector<int> all_indices(data.size());
        for (int i = 0; i < data.size(); ++i) {
            all_indices[i] = i;
        }

        cost_complexity_prune(alpha, data, all_indices, all_indices.size());
    }

    std::pair<std::vector<int>, std::vector<int>> split_data_indices(int total_samples, double validation_size) {

        std::vector<int> all_indices(total_samples);
        for (int i = 0; i < total_samples; ++i) {
            all_indices[i] = i;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(all_indices.begin(), all_indices.end(), gen);

        int idx_to_split = static_cast<int>(validation_size * total_samples);

        if (idx_to_split <= 0) idx_to_split = 1;
        if (idx_to_split >= total_samples) idx_to_split = total_samples - 1;

        auto validation_iter_begin = all_indices.begin();
        auto validation_iter_end = all_indices.begin() + idx_to_split;
        auto train_iter_begin = all_indices.begin() + idx_to_split;
        auto train_iter_end = all_indices.end();

        std::vector<int> train_indices(train_iter_begin, train_iter_end);
        std::vector<int> validation_indices(validation_iter_begin, validation_iter_end);
        return std::pair{train_indices, validation_indices};
    }

    Node* recursive_copy_node(Node* original) const {
        if (original == nullptr) return nullptr;

        if (dynamic_cast<LeafNode*>(original) != nullptr) {
            LeafNode* original_leaf_node = dynamic_cast<LeafNode*>(original);
            LeafNode* copy_leaf_node = new LeafNode(original_leaf_node->get_class_probabilities(), original_leaf_node->get_sample_count());
            return copy_leaf_node;
        }

        if (dynamic_cast<InternalNode*>(original) != nullptr) {
            InternalNode* original_internal_node = dynamic_cast<InternalNode*>(original);
            Node* copy_left_child = recursive_copy_node(original_internal_node->get_left_child());
            Node* copy_right_child = recursive_copy_node(original_internal_node->get_right_child());
            InternalNode* copy_internal_node = new InternalNode(original_internal_node->get_feature_index(), original_internal_node->get_threshold(),
                copy_left_child, copy_right_child, original_internal_node->get_sample_count());
            return copy_internal_node;
        }

        return nullptr;
    }

    DecisionTree* create_copy() const {
        DecisionTree* copy_tree = new DecisionTree(
            max_depth,
            min_samples_split,
            min_samples_leaf,
            get_criterion(),
            ccp_alpha,
            auto_prune,
            validation_size
        );

        copy_tree->root = recursive_copy_node(root);

        copy_tree->feature_importances = this->feature_importances;

        return copy_tree;
    }

    struct PrunedTree {
        double alpha;
        DecisionTree* tree;

        PrunedTree(double alpha, DecisionTree* tree) : alpha(alpha), tree(tree) {}
    };

    std::vector<PrunedTree> build_pruning_sequence(const std::vector<DataPoint>& data,
                                              const std::vector<int>& train_indices) {
        std::vector<PrunedTree> sequence;

        DecisionTree* current_tree = create_copy();
        sequence.push_back(PrunedTree(0.0, current_tree));

        int max_iterations = 1024;
        int iteration = 0;
        int total_training_samples = train_indices.size();

        while (iteration < max_iterations) {
            Node* weakest_link = find_weakest_links(current_tree->root, data, train_indices, total_training_samples);
            if (weakest_link == nullptr) break;

            double alpha = calculate_prune_metric(weakest_link, data, train_indices, total_training_samples);
            DecisionTree* pruned_tree = current_tree->create_copy();
            pruned_tree->cost_complexity_prune(alpha, data, train_indices, total_training_samples);

            if (count_subtree_leaves(pruned_tree->root) == count_subtree_leaves(current_tree->root)) {
                delete pruned_tree;
                break;
            }

            sequence.push_back(PrunedTree(alpha, pruned_tree));
            current_tree = pruned_tree;
            ++iteration;
        }

        return sequence;
    }

    DecisionTree* select_best_tree(const std::vector<PrunedTree>& sequence,
        const std::vector<DataPoint>& data,
        const std::vector<int>& validation_indices) {
        if (sequence.empty()) return nullptr;
        if (validation_indices.size() == 0) return nullptr;
        DecisionTree* best_tree = sequence[0].tree;
        double min_error = calculate_node_error(best_tree->root, data, validation_indices);

        for (size_t i = 1; i < sequence.size(); ++i) {
            double current_error = calculate_node_error(sequence[i].tree->root, data, validation_indices);
            if (current_error < min_error) {
                min_error = current_error;
                best_tree = sequence[i].tree;
            }
        }

        return best_tree->create_copy();
    }

    void automatic_pruning(const std::vector<DataPoint>& data) {
        auto [train_indices, validation_indices] = split_data_indices(data.size(), validation_size);

        auto sequence = build_pruning_sequence(data, train_indices);

        auto best_tree = select_best_tree(sequence, data, validation_indices);

        if (best_tree != nullptr) {
            delete this->root;
            this->root = best_tree->root;
            best_tree->root = nullptr;
        }
        else {
            for (auto& pruned_tree : sequence) {
                delete pruned_tree.tree;
            }
            return;
        }

        for (auto& pruned_tree : sequence) {
            delete pruned_tree.tree;
        }

        delete best_tree;
    }

    Node* build_tree(const std::vector<DataPoint>& data,
        const std::vector<int>& indices,
        int depth, int total_samples) {

        auto probabilities = calculate_probabilities(data, indices);

        if (indices.size() < min_samples_split) {
            return new LeafNode(probabilities, indices.size());
        }

        if (indices.empty()) {
            return new LeafNode(probabilities, indices.size());
        }

        if (all_same_class(data, indices)) return new LeafNode(probabilities, indices.size());

        if (depth >= max_depth) {
            return new LeafNode(probabilities, indices.size());
        }

        auto best_split = find_best_split(data, indices);

        if ((best_split.left_indices.size() < min_samples_leaf) || (best_split.right_indices.size() < min_samples_leaf)) {
            return new LeafNode(probabilities, indices.size());
        }

        if (best_split.information_gain < 0.0) {
            return new LeafNode(probabilities, indices.size());
        }


        int feature_index = best_split.feature_index;
        double threshold = best_split.threshold;

        double weight = static_cast<double>(indices.size()) / total_samples;
        if (feature_importances.empty() && !data.empty()) {
            feature_importances[feature_index] += weight * best_split.information_gain;
        }

        auto left_child = build_tree(data, best_split.left_indices, depth + 1, total_samples);
        auto right_child = build_tree(data, best_split.right_indices, depth + 1, total_samples);
        return new InternalNode(feature_index, threshold, left_child, right_child, indices.size()); //заглушка
    }

//=======================================================PUBLIC========================================================================//

public:
    DecisionTree(int max_depth = 32,
                 int min_samples_split = 2,
                 int min_samples_leaf = 1,
                 const std::string& string_criterion = "entropy",
                 double ccp_alpha = 0.0,
                 bool auto_prune = true,
                 double validation_size = 0.15)
        : max_depth(max_depth),
          min_samples_split(min_samples_split),
          min_samples_leaf(min_samples_leaf),
          ccp_alpha(ccp_alpha > 0 ? ccp_alpha : 0.0),
          auto_prune(ccp_alpha > 0 ? false : auto_prune),
          validation_size(validation_size)
        {

        if (ccp_alpha > 0) {
            this->ccp_alpha = ccp_alpha;
            this->auto_prune = false;
        } else {
            this->auto_prune = auto_prune;
            this->validation_size = validation_size;
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

        if (auto_prune) {
            automatic_pruning(data);
        } else if (ccp_alpha > 0.0) {
            manual_pruning(data, ccp_alpha);
        }
    }

    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
        if (X.size() != y.size() || X.empty()) {
            throw std::invalid_argument("Invalid train data.\n");
        }

        std::vector<DataPoint> data;
        data.reserve(X.size());

        for (size_t i = 0; i < X.size(); ++i) {
            data.emplace_back(X[i], y[i]);
        }

        fit(data);
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


#include <random>
#include <chrono>

std::vector<DataPoint> generate_test_data(int data_size) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<DataPoint> data;
    data.reserve(data_size);

    std::uniform_real_distribution<double> feature_x1(0.0, 0.6);
    std::uniform_real_distribution<double> feature_x2(0.4, 1.0);
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
