/**
 * 08-Data-Structures: 常用数据结构实现
 *
 * 本文件演示 C++ 中常见数据结构的实现与使用：
 * - 数组与字符串操作
 * - 链表（单链表）
 * - 栈与队列
 * - 哈希表（unordered_map/set）
 */

#include <iostream>
#include <vector>
#include <string>
#include <list>
#include <stack>
#include <queue>
#include <unordered_map>
#include <unordered_set>

using namespace std;

// ==================== 1. 数组与字符串 ====================

void array_and_string_demo() {
    cout << "--- 1. 数组与字符串 ---" << endl;

    // 动态数组（vector）
    vector<int> nums = {1, 2, 3, 4, 5};
    nums.push_back(6);        // 尾部添加
    nums.pop_back();          // 尾部删除
    cout << "数组大小: " << nums.size() << endl;
    cout << "第一个元素: " << nums.front() << endl;
    cout << "最后一个元素: " << nums.back() << endl;

    // 遍历
    cout << "遍历: ";
    for (int num : nums) {
        cout << num << " ";
    }
    cout << endl;

    // 字符串操作
    string text = "Hello, World!";
    cout << "字符串长度: " << text.length() << endl;
    cout << "子串: " << text.substr(0, 5) << endl;
    cout << "查找 'World': " << text.find("World") << endl;
}

// ==================== 2. 链表（单链表） ====================

// 链表节点定义
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

// 创建链表
ListNode* create_list(const vector<int>& vals) {
    if (vals.empty()) return nullptr;
    ListNode* head = new ListNode(vals[0]);
    ListNode* current = head;
    for (size_t i = 1; i < vals.size(); ++i) {
        current->next = new ListNode(vals[i]);
        current = current->next;
    }
    return head;
}

// 打印链表
void print_list(ListNode* head) {
    cout << "链表: ";
    while (head) {
        cout << head->val << " -> ";
        head = head->next;
    }
    cout << "nullptr" << endl;
}

// 反转链表
ListNode* reverse_list(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* current = head;
    while (current) {
        ListNode* next_temp = current->next;
        current->next = prev;
        prev = current;
        current = next_temp;
    }
    return prev;
}

// 释放链表内存
void free_list(ListNode* head) {
    while (head) {
        ListNode* temp = head;
        head = head->next;
        delete temp;
    }
}

void linked_list_demo() {
    cout << "\n--- 2. 链表 ---" << endl;

    // 创建链表 1->2->3->4->5
    vector<int> vals = {1, 2, 3, 4, 5};
    ListNode* head = create_list(vals);
    print_list(head);

    // 反转链表
    ListNode* reversed = reverse_list(head);
    cout << "反转后: ";
    print_list(reversed);

    // 释放内存
    free_list(reversed);
}

// ==================== 3. 栈与队列 ====================

void stack_and_queue_demo() {
    cout << "\n--- 3. 栈与队列 ---" << endl;

    // 栈（后进先出 LIFO）
    stack<int> s;
    s.push(1);
    s.push(2);
    s.push(3);
    cout << "栈顶元素: " << s.top() << endl;
    s.pop();
    cout << "弹出后栈顶: " << s.top() << endl;
    cout << "栈大小: " << s.size() << endl;

    // 队列（先进先出 FIFO）
    queue<int> q;
    q.push(1);
    q.push(2);
    q.push(3);
    cout << "\n队首元素: " << q.front() << endl;
    q.pop();
    cout << "出队后队首: " << q.front() << endl;
    cout << "队列大小: " << q.size() << endl;
}

// ==================== 4. 哈希表 ====================

void hash_table_demo() {
    cout << "\n--- 4. 哈希表 ---" << endl;

    // unordered_map（键值对）
    unordered_map<string, int> scores;
    scores["Alice"] = 95;
    scores["Bob"] = 87;
    scores["Charlie"] = 92;

    cout << "Alice 的分数: " << scores["Alice"] << endl;

    // 遍历
    cout << "所有成绩:" << endl;
    for (const auto& pair : scores) {
        cout << "  " << pair.first << ": " << pair.second << endl;
    }

    // 查找
    if (scores.count("Bob")) {
        cout << "Bob 的成绩存在: " << scores["Bob"] << endl;
    }

    // unordered_set（唯一元素集合）
    unordered_set<int> seen;
    vector<int> nums = {1, 2, 2, 3, 4, 4, 5};
    for (int num : nums) {
        seen.insert(num);
    }
    cout << "\n去重后的元素: ";
    for (int num : seen) {
        cout << num << " ";
    }
    cout << endl;
}

// ==================== 主函数 ====================

int main() {
    system("chcp 65001 >nul");  // 设置 UTF-8 编码

    array_and_string_demo();
    linked_list_demo();
    stack_and_queue_demo();
    hash_table_demo();

    cout << "\n✅ 数据结构演示完成！" << endl;
    return 0;
}
