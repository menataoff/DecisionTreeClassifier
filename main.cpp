#include <iostream>
#include "decision_tree.h"

int main() {
    // 1. Генерация данных
    std::vector<DataPoint> data = generate_test_data(100);

    // 2. Статистика данных
    int count0 = 0, count1 = 0;
    for (const auto& point : data) {
        if (point.label == 0) count0++;
        else count1++;
    }
    std::cout << "Data stats:\n";
    std::cout << "Label 0: " << count0 << " objects\n";
    std::cout << "Label  1: " << count1 << " objects\n";
    std::cout << "Majority class: " << (count0 > count1 ? 0 : 1) << "\n\n";

    // 3. Обучение дерева
    DecisionTree tree(5);  // max_depth = 5
    tree.fit(data);
    std::cout << "Tree has been trained!\n\n";

    // 4. Тестирование предсказаний
    std::cout << "Test:\n";

    // Точки из кластера 0 (должны быть класс 0)
    std::vector<double> point1 = {0.2, 0.3};
    std::vector<double> point2 = {0.1, 0.8};

    // Точки из кластера 1 (должны быть класс 1)
    std::vector<double> point3 = {0.8, 0.7};
    std::vector<double> point4 = {0.9, 0.2};

    std::cout << "Object (0.2, 0.3): " << tree.predict(point1) << "\n";
    std::cout << "Object (0.1, 0.8): " << tree.predict(point2) << "\n";
    std::cout << "Object (0.8, 0.7): " << tree.predict(point3) << "\n";
    std::cout << "Object (0.9, 0.2): " << tree.predict(point4) << "\n";

    std::cout << "=== TESTING ENTROPY ===" << std::endl;

    // Test case 1: All same class (entropy should be 0)
    std::vector<DataPoint> test1 = {
        {{0.0, 0.0}, 0},
        {{0.0, 0.0}, 0},
        {{0.0, 0.0}, 0}
    };
    // double entropy1 = tree.calculate_entropy(test1, {0, 1, 2});
    // std::cout << "Test 1 - All class 0: " << entropy1 << " (expected: 0.0)" << std::endl;
    //
    // // Test case 2: Mixed classes (entropy should be ~1.0)
    // std::vector<DataPoint> test2 = {
    //     {{0.0, 0.0}, 0},
    //     {{0.0, 0.0}, 0},
    //     {{0.0, 0.0}, 1},
    //     {{0.0, 0.0}, 1}
    // };
    // double entropy2 = tree.calculate_entropy(test2, {0, 1, 2, 3});
    // std::cout << "Test 2 - 50/50 split: " << entropy2 << " (expected: 1.0)" << std::endl;
    //
    // // Test case 3: One of each class (entropy should be ~1.58)
    // std::vector<DataPoint> test3 = {
    //     {{0.0, 0.0}, 0},
    //     {{0.0, 0.0}, 1},
    //     {{0.0, 0.0}, 2}
    // };
    // double entropy3 = tree.calculate_entropy(test3, {0, 1, 2});
    // std::cout << "Test 3 - Three classes: " << entropy3 << " (expected: ~1.58)" << std::endl;

    return 0;
}