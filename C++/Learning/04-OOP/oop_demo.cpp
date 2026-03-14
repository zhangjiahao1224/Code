/**
 * 04-OOP: 类、封装、继承、多态
 */
#include <iostream>
#include <cstdlib>
using namespace std;

class Rectangle {
    double width, height;
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    double area() const { return width * height; }
};

class Shape {
public:
    virtual double area() const { return 0; }
    virtual ~Shape() {}
};
class Rectangle : public Shape { // Rectangle 改为继承自 Shape
    double width, height;
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    double area() const override { return width * height; }  // 需要 override，因为基类是虚函数
};

class Circle : public Shape {
    double r;
public:
    Circle(double radius) : r(radius) {}
    double area() const override { return 3.14159 * r * r; }
};

void printArea(const Shape& s) {
    cout << "面积: " << s.area() << endl;
}

int main() {
    system("chcp 65001 >nul");

    cout << "--- 1. 封装 ---" << endl;
    Rectangle rect(3, 4);
    cout << "矩形面积: " << rect.area() << endl;

    cout << "\n--- 2. 继承 ---" << endl;
    Circle circle(5);
    cout << "圆面积: " << circle.area() << endl;

    cout << "\n--- 3. 多态 ---" << endl;
    Shape* p1 = new Rectangle(2, 5);
    Shape* p2 = new Circle(3);
    printArea(*p1);
    printArea(*p2);
    delete p1;
    delete p2;

    return 0;
}
