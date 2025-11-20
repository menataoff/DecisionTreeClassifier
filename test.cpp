#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "decision_tree.h"

std::vector<DataPoint> load_csv(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    std::string line;
    
    // Пропускаем заголовок
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;
        int label;

        while (std::getline(ss, value, ',')) {
            if (ss.eof()) {
                label = std::stoi(value);
            } else {
                features.push_back(std::stod(value));
            }
        }

        data.emplace_back(features, label);
    }
    return data;
}

int main() {
    auto data = load_csv("data.csv");

    int train_size = data.size() * 0.8;
    std::vector<DataPoint> train_data(data.begin(), data.begin() + train_size);
    std::vector<DataPoint> test_data(data.begin() + train_size, data.end());

    DecisionTree tree;
    tree.fit(train_data);

    int correct = 0;
    for (const auto& point : test_data) {
        if (tree.predict(point.features) == point.label) {
            correct++;
        }
    }

    double accuracy = static_cast<double>(correct) / test_data.size();
    std::cout << "Accuracy: " << accuracy << " (" << correct << "/" << test_data.size() << ")" << std::endl;

    // Важность признаков
    auto importances = tree.get_feature_importances();
    std::cout << "\nFeature importances:" << std::endl;
    for (size_t i = 0; i < importances.size(); ++i) {
        std::cout << "Feature " << i << ": " << importances[i] << std::endl;
    }

    // Вероятности для случайной точки
    if (!test_data.empty()) {
        auto& random_point = test_data[0];
        auto proba = tree.predict_proba(random_point.features);

        std::cout << "\nProbabilities for random point:" << std::endl;
        for (const auto& [class_label, probability] : proba) {
            std::cout << "Class " << class_label << ": " << probability << std::endl;
        }
        std::cout << "Actual class: " << random_point.label << std::endl;
    }
    
    return 0;
}