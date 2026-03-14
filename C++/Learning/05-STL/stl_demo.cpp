/**
 * 05-STL: 容器、算法、迭代器
 */
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
using namespace std;

int main() {
    system("chcp 65001 >nul");

    cout << "--- 1. vector 容器 ---" << endl;
    vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};
    v.push_back(7);
    cout << "大小: " << v.size() << ", 元素: ";
    for (int x : v) cout << x << " ";
    cout << endl;

    cout << "\n--- 2. 迭代器 ---" << endl;
    cout << "遍历: ";
    for (auto it = v.begin(); it != v.end(); ++it)
        cout << *it << " ";
    cout << endl;

    cout << "\n--- 3. sort 排序 ---" << endl;
    sort(v.begin(), v.end());
    cout << "排序后: ";
    for (int x : v) cout << x << " ";
    cout << endl;

    cout << "\n--- 4. find 查找 ---" << endl;
    auto pos = find(v.begin(), v.end(), 5);
    if (pos != v.end())
        cout << "找到 5，下标: " << (pos - v.begin()) << endl;
    else
        cout << "未找到 5" << endl;

    cout << "\n--- 5. max_element ---" << endl;
    auto maxIt = max_element(v.begin(), v.end());
    cout << "最大值: " << *maxIt << endl;

    return 0;
}
