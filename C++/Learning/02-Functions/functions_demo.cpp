/**
 * 02-Functions: 函数定义、参数、返回值、重载
 */
#include <iostream>
#include <cstdlib>
using namespace std;

void sayHello() { cout << "你好！" << endl; }
int add(int a, int b) { return a + b; }
double area(double w, double h = 1.0) { return w * h; }
int max2(int a, int b) { return a > b ? a : b; }
double max2(double a, double b) { return a > b ? a : b; }
int max3(int a, int b, int c) { return max2(max2(a, b), c); }
void byValue(int x) { x++; }
void byRef(int& x) { x++; }

int main() {
    system("chcp 65001 >nul");

    cout << "--- 1. 无参函数 ---" << endl;
    sayHello();

    cout << "\n--- 2. 参数与返回值 ---" << endl;
    cout << "add(3,5)=" << add(3, 5) << endl;

    cout << "\n--- 3. 默认参数 ---" << endl;
    cout << "area(3,4)=" << area(3, 4) << " area(5)=" << area(5) << endl;

    cout << "\n--- 4. 函数重载 ---" << endl;
    cout << "max2(10,20)=" << max2(10, 20) << " max2(1.5,2.3)=" << max2(1.5, 2.3)
         << " max3(1,5,3)=" << max3(1, 5, 3) << endl;

    cout << "\n--- 5. 传值 vs 传引用 ---" << endl;
    int n = 10;
    byValue(n);
    cout << "传值后 n=" << n << endl;
    byRef(n);
    cout << "传引用后 n=" << n << endl;

    return 0;
}
