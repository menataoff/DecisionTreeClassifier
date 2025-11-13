#include <iostream>
#include "decision_tree.h"

int main() {
    // // 1. Генерация данных
    // std::vector<DataPoint> data = generate_test_data(100);
    //
    // // 2. Статистика данных
    // int count0 = 0, count1 = 0;
    // for (const auto& point : data) {
    //     if (point.label == 0) count0++;
    //     else count1++;
    // }
    // std::cout << "Data stats:\n";
    // std::cout << "Label 0: " << count0 << " objects\n";
    // std::cout << "Label  1: " << count1 << " objects\n";
    // std::cout << "Majority class: " << (count0 > count1 ? 0 : 1) << "\n\n";
    //
    // // 3. Обучение дерева
    // DecisionTree tree(5);  // max_depth = 5
    // tree.fit(data);
    // std::cout << "Tree has been trained!\n\n";
    //
    // // 4. Тестирование предсказаний
    // std::cout << "Test:\n";
    //
    // // Точки из кластера 0 (должны быть класс 0)
    // std::vector<double> point1 = {0.2, 0.3};
    // std::vector<double> point2 = {0.1, 0.8};
    //
    // // Точки из кластера 1 (должны быть класс 1)
    // std::vector<double> point3 = {0.8, 0.7};
    // std::vector<double> point4 = {0.9, 0.2};
    //
    // std::cout << "Object (0.2, 0.3): " << tree.predict(point1) << "\n";
    // std::cout << "Object (0.1, 0.8): " << tree.predict(point2) << "\n";
    // std::cout << "Object (0.8, 0.7): " << tree.predict(point3) << "\n";
    // std::cout << "Object (0.9, 0.2): " << tree.predict(point4) << "\n";
    //
    // // Test predict_proba
    // std::cout << "=== TESTING PREDICT_PROBA ===" << std::endl;
    //
    // // Точка из кластера 0
    // auto proba1 = tree.predict_proba({0.5, 0.3});
    // std::cout << "Point (0.5, 0.3) probabilities: ";
    // for (auto& [cls, prob] : proba1) {
    //     std::cout << "class " << cls << ": " << prob << " ";
    // }
    // std::cout << std::endl;
    //
    // // Точка из кластера 1
    // auto proba2 = tree.predict_proba({0.8, 0.7});
    // std::cout << "Point (0.8, 0.7) probabilities: ";
    // for (auto& [cls, prob] : proba2) {
    //     std::cout << "class " << cls << ": " << prob << " ";
    // }
    // std::cout << std::endl;
    //
    // // Test feature importance
    // std::cout << "=== TESTING FEATURE IMPORTANCE ===" << std::endl;
    // auto importances = tree.get_feature_importances();
    // std::cout << "Feature importances: ";
    // for (size_t i = 0; i < importances.size(); ++i) {
    //     std::cout << "feature_" << i << ": " << importances[i] << " ";
    // }
    // std::cout << std::endl;
    //
    // // Сумма должна быть <= 1.0 (после нормализации)
    // double sum = 0.0;
    // for (auto imp : importances) sum += imp;
    // std::cout << "Sum of importances: " << sum << std::endl;
    //
    // // Test both criteria
    // std::cout << "=== TESTING BOTH CRITERIA ===" << std::endl;
    //
    // DecisionTree tree_entropy(10, 2, "entropy");
    // DecisionTree tree_gini(10, 2, "gini");

    std::cout << "=== TESTING BOTH CRITERIA ===" << std::endl;

    DecisionTree tree_entropy(10, 2, 1, "entropy");
    DecisionTree tree_gini(10, 2, 1, "gini");

    auto data = generate_test_data(100);

    tree_entropy.fit(data);
    tree_gini.fit(data);

    // Feature importance comparison
    std::cout << "ENTROPY feature importance: ";
    auto imp_entropy = tree_entropy.get_feature_importances();
    for (auto imp : imp_entropy) std::cout << imp << " ";

    std::cout << "\nGINI feature importance: ";
    auto imp_gini = tree_gini.get_feature_importances();
    for (auto imp : imp_gini) std::cout << imp << " ";

    // Prediction comparison
    std::cout << "\n\nPredictions for (0.2, 0.5):" << std::endl;
    std::cout << "ENTROPY: " << tree_entropy.predict({0.2, 0.5}) << std::endl;
    std::cout << "GINI: " << tree_gini.predict({0.2, 0.5}) << std::endl;

    return 0;
}