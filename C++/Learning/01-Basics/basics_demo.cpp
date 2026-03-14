/**
 * 01-Basics: 变量、数据类型、运算符、流程控制
 *
 * 本文件演示 C++ 基础语法
 */
#include <iostream>  // 标准输入输出
#include <cstdlib>   // system() 函数
using namespace std; // 使用 std 命名空间，避免写 std::cout

int main() {
    system("chcp 65001 >nul");  // 控制台设为 UTF-8，避免中文乱码

    // ---------- 1. 变量与数据类型 ----------
    cout << "--- 1. 变量与数据类型 ---" << endl;
    int age = 18;      // 整型
    double price = 19.99;  // 双精度浮点
    char grade = 'A';  // 字符，单引号
    bool passed = true;    // 布尔，true/false
    cout << "年龄: " << age << ", 价格: " << price << endl;
    cout << "等级: " << grade << ", 通过: " << (passed ? "是" : "否") << endl;  // 三元运算符

    // ---------- 2. 运算符 ----------
    cout << "\n--- 2. 运算符 ---" << endl;
    int a = 10, b = 3;
    // + - * / 加减乘除，% 取余
    cout << "a+b=" << (a + b) << " a-b=" << (a - b) << " a*b=" << (a * b)
         << " a/b=" << (a / b) << " a%b=" << (a % b) << endl;
    // 注意：整数除法 a/b=3，会截断小数

    // ---------- 3. if-else 分支 ----------
    cout << "\n--- 3. if-else ---" << endl;
    int score = 85;
    if (score >= 90) cout << "优秀" << endl;
    else if (score >= 60) cout << "及格" << endl;
    else cout << "不及格" << endl;

    // ---------- 4. for 循环 ----------
    cout << "\n--- 4. for 循环 ---" << endl;
    // for(初始化; 条件; 步进) { 循环体 }
    for (int i = 0; i < 5; i++) cout << i << " ";
    cout << endl;

    // ---------- 5. while 循环 ----------
    cout << "\n--- 5. while 循环 ---" << endl;
    int j = 0;
    while (j < 3) { cout << j << " "; j++; }  // 条件为真时继续循环
    cout << endl;

    return 0;  // 返回 0 表示程序正常结束
}
