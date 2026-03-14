/**
 * 06-Memory-Management: new/delete、RAII、智能指针
 */
#include <iostream>
#include <memory>
#include <cstdlib>
using namespace std;

void rawPointerDemo() {
    cout << "--- 1. new/delete ---" << endl;
    int* p = new int(42);
    cout << "*p = " << *p << endl;
    delete p;
}

void uniquePtrDemo() {
    cout << "\n--- 2. unique_ptr ---" << endl;
    unique_ptr<int> up(new int(100));
    cout << "*up = " << *up << endl;
}

void sharedPtrDemo() {
    cout << "\n--- 3. shared_ptr ---" << endl;
    shared_ptr<int> sp1 = make_shared<int>(200);
    shared_ptr<int> sp2 = sp1;
    cout << "引用计数: " << sp1.use_count() << endl;
    cout << "*sp1 = " << *sp1 << ", *sp2 = " << *sp2 << endl;
}

void makeSharedDemo() {
    cout << "\n--- 4. make_shared ---" << endl;
    auto sp = make_shared<int>(300);
    cout << "*sp = " << *sp << endl;
}

int main() {
    system("chcp 65001 >nul");

    rawPointerDemo();
    uniquePtrDemo();
    sharedPtrDemo();
    makeSharedDemo();
    cout << "\n--- 5. RAII ---" << endl;
    cout << "RAII：构造时获取资源，析构时自动释放" << endl;

    return 0;
}
