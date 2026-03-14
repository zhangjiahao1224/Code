/**
 * 07-Advanced-Topics: 模板、异常、多线程
 */
#include <iostream>
#include <string>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <cstdlib>
using namespace std;

template<typename T>
T add(T a, T b) { return a + b; }

template<typename T>
class Box {
    T value;
public:
    Box(T v) : value(v) {}
    T get() const { return value; }
};

double divide(double a, double b) {
    if (b == 0) throw runtime_error("除数不能为 0");
    return a / b;
}

void modernFeatures() {
    auto x = 42;
    auto arr = {1, 2, 3};
    cout << "auto + 范围 for: ";
    for (auto v : arr) cout << v << " ";
    cout << endl;
}

void lambdaDemo() {
    auto f = [](int a, int b) { return a * b; };
    cout << "lambda 3*4 = " << f(3, 4) << endl;
}

void threadFunc() {
    cout << "子线程运行中..." << endl;
    this_thread::sleep_for(chrono::milliseconds(100));
}

int main() {
    system("chcp 65001 >nul");

    cout << "--- 1. 函数模板 ---" << endl;
    cout << "add(1,2)=" << add(1, 2) << " add(1.5,2.5)=" << add(1.5, 2.5) << endl;

    cout << "\n--- 2. 类模板 ---" << endl;
    Box<int> bi(10);
    Box<string> bs("hello");
    cout << "Box<int>=" << bi.get() << " Box<string>=" << bs.get() << endl;

    cout << "\n--- 3. 异常 ---" << endl;
    try {
        cout << "10/2 = " << divide(10, 2) << endl;
        cout << "10/0 = " << divide(10, 0) << endl;
    } catch (const exception& e) {
        cout << "异常: " << e.what() << endl;
    }

    cout << "\n--- 4. auto、范围 for ---" << endl;
    modernFeatures();

    cout << "\n--- 5. lambda ---" << endl;
    lambdaDemo();

    cout << "\n--- 6. 多线程 ---" << endl;
    thread t(threadFunc);
    cout << "主线程等待..." << endl;
    t.join();
    cout << "子线程结束" << endl;

    return 0;
}
