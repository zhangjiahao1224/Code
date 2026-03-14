/**
 * 03-Arrays-and-Pointers: 数组、指针、引用
 */
#include <iostream>
#include <cstdlib>
using namespace std;

int main() {
    system("chcp 65001 >nul");

    cout << "--- 1. 数组 ---" << endl;
    int arr[5] = {10, 20, 30, 40, 50};
    for (int i = 0; i < 5; i++)
        cout << "arr[" << i << "] = " << arr[i] << endl;

    cout << "\n--- 2. 指针 ---" << endl;
    int x = 42;
    int* p = &x;
    cout << "x = " << x << ", &x = " << &x << endl;
    cout << "p = " << p << ", *p = " << *p << endl;
    *p = 100;
    cout << "修改后 x = " << x << endl;

    cout << "\n--- 3. 指针与数组 ---" << endl;
    int* pArr = arr;
    cout << "arr[0] = " << *pArr << ", arr[1] = " << *(pArr + 1) << endl;

    cout << "\n--- 4. 引用 ---" << endl;
    int a = 5;
    int& ref = a;
    cout << "a = " << a << ", ref = " << ref << endl;
    ref = 99;
    cout << "修改 ref 后 a = " << a << endl;

    cout << "\n--- 5. const 引用 ---" << endl;
    const int& cref = a;
    cout << "cref = " << cref << endl;

    return 0;
}
