#include <iostream>
using namespace std;

int add(int a, int b) {
    return a + b;
}

int main() {
    cout << "Hello, world!" << endl;

    int x = 1;
    int y = 2;
    x = y;
    y = y + 1;
    int c = add(x, y);
    cout << "c = " << c << endl;

    return 0;
}
