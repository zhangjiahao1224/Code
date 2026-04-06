# -*- coding: utf-8 -*-
"""
数据结构基础练习 - Python 实现

本文件演示 Python 中常见数据结构的使用：
- 列表（动态数组）
- 链表
- 栈与队列
- 字典与集合（哈希表）
"""

# ==================== 1. 列表（动态数组） ====================

def list_demo():
    """演示 Python 列表的基本操作"""
    print("--- 1. 列表（动态数组） ---")
    
    # 创建列表
    nums = [1, 2, 3, 4, 5]
    print(f"原始列表: {nums}")
    
    # 添加元素
    nums.append(6)           # 尾部添加
    nums.insert(0, 0)        # 指定位置插入
    print(f"添加后: {nums}")
    
    # 删除元素
    nums.pop()               # 尾部删除
    nums.pop(0)              # 指定位置删除
    nums.remove(3)           # 按值删除
    print(f"删除后: {nums}")
    
    # 访问元素
    print(f"第一个元素: {nums[0]}")
    print(f"最后一个元素: {nums[-1]}")
    
    # 切片
    print(f"前 3 个元素: {nums[:3]}")
    print(f"后 2 个元素: {nums[-2:]}")
    
    # 遍历
    print("遍历:", end=" ")
    for num in nums:
        print(num, end=" ")
    print()

# ==================== 2. 链表 ====================

class ListNode:
    """链表节点类"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def create_linked_list(values):
    """从列表创建链表"""
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    return head

def print_linked_list(head):
    """打印链表"""
    values = []
    while head:
        values.append(str(head.val))
        head = head.next
    print(" -> ".join(values) + " -> None")

def reverse_linked_list(head):
    """反转链表"""
    prev = None
    current = head
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    return prev

def linked_list_demo():
    """演示链表操作"""
    print("\n--- 2. 链表 ---")
    
    # 创建链表 1->2->3->4->5
    head = create_linked_list([1, 2, 3, 4, 5])
    print("原始链表:", end=" ")
    print_linked_list(head)
    
    # 反转链表
    reversed_head = reverse_linked_list(head)
    print("反转后:  ", end=" ")
    print_linked_list(reversed_head)

# ==================== 3. 栈与队列 ====================

from collections import deque

def stack_and_queue_demo():
    """演示栈和队列的使用"""
    print("\n--- 3. 栈与队列 ---")
    
    # 栈（使用 list 模拟，后进先出 LIFO）
    stack = []
    stack.append(1)    # push
    stack.append(2)
    stack.append(3)
    print(f"栈: {stack}")
    print(f"栈顶元素: {stack[-1]}")
    print(f"弹出: {stack.pop()}")
    print(f"弹出后栈: {stack}")
    
    # 队列（使用 deque，先进先出 FIFO）
    queue = deque()
    queue.append(1)    # enqueue
    queue.append(2)
    queue.append(3)
    print(f"\n队列: {list(queue)}")
    print(f"队首元素: {queue[0]}")
    print(f"出队: {queue.popleft()}")
    print(f"出队后队列: {list(queue)}")

# ==================== 4. 字典与集合（哈希表） ====================

def hash_table_demo():
    """演示字典和集合的使用"""
    print("\n--- 4. 字典与集合（哈希表） ---")
    
    # 字典（键值对）
    scores = {
        "Alice": 95,
        "Bob": 87,
        "Charlie": 92
    }
    print(f"字典: {scores}")
    print(f"Alice 的分数: {scores['Alice']}")
    
    # 添加/修改
    scores["David"] = 88
    scores["Bob"] = 90
    print(f"更新后: {scores}")
    
    # 遍历
    print("所有成绩:")
    for name, score in scores.items():
        print(f"  {name}: {score}")
    
    # 查找
    if "Alice" in scores:
        print(f"Alice 的成绩存在: {scores['Alice']}")
    
    # 集合（唯一元素）
    nums = [1, 2, 2, 3, 4, 4, 5]
    unique_nums = set(nums)
    print(f"\n原始列表: {nums}")
    print(f"去重后: {unique_nums}")
    
    # 集合操作
    set_a = {1, 2, 3, 4}
    set_b = {3, 4, 5, 6}
    print(f"并集: {set_a | set_b}")
    print(f"交集: {set_a & set_b}")
    print(f"差集: {set_a - set_b}")

# ==================== 主函数 ====================

if __name__ == "__main__":
    list_demo()
    linked_list_demo()
    stack_and_queue_demo()
    hash_table_demo()
    
    print("\n✅ 数据结构演示完成！")
