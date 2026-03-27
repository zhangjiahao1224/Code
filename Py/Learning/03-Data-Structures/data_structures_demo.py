# 03-数据结构演示：列表、元组、字典、集合、字符串操作

# 列表 (Lists) - 可变序列
fruits = ["apple", "banana", "cherry"]  # 创建水果列表
print("Fruits:", fruits)  # 打印水果列表
print("First fruit:", fruits[0])  # 打印第一个水果
print("Last fruit:", fruits[-1])  # 打印最后一个水果

# 修改列表
fruits.append("orange")  # 在末尾添加橙子
fruits.insert(1, "blueberry")  # 在索引1处插入蓝莓
print("After modifications:", fruits)  # 打印修改后的列表

# 删除元素
fruits.remove("banana")  # 移除香蕉
popped = fruits.pop()  # 弹出并返回最后一个元素
print("After removals:", fruits)  # 打印删除后的列表
print("Popped item:", popped)  # 打印弹出的元素

# 列表操作
print("Length:", len(fruits))  # 打印列表长度
print("Sorted:", sorted(fruits))  # 打印排序后的列表
print("Reversed:", list(reversed(fruits)))  # 打印反转后的列表

# 元组 (Tuples) - 不可变序列
coordinates = (10.0, 20.0)  # 创建坐标元组
print("\nCoordinates:", coordinates)  # 打印坐标
# coordinates[0] = 15.0  # 这会导致错误，因为元组不可变

# 字典 (Dictionaries) - 键值对映射
student = {
    "name": "John Doe",  # 姓名
    "age": 20,  # 年龄
    "major": "Computer Science",  # 专业
    "grades": [85, 92, 78]  # 成绩列表
}
print("\nStudent info:")  # 打印学生信息
print("Name:", student["name"])  # 打印姓名
print("Age:", student["age"])  # 打印年龄
print("Major:", student["major"])  # 打印专业

# 修改字典
student["year"] = "Sophomore"  # 添加年级
student["grades"].append(96)  # 在成绩列表中添加新成绩
print("Updated student:", student)  # 打印更新后的学生信息

# 字典方法
print("Keys:", list(student.keys()))  # 打印所有键
print("Values:", list(student.values()))  # 打印所有值
print("Items:", list(student.items()))  # 打印所有键值对

# 集合 (Sets) - 无序唯一元素
numbers = {1, 2, 3, 4, 5, 5, 3}  # 创建数字集合，重复元素会被移除
print("\nSet of numbers:", numbers)  # 打印数字集合
numbers.add(6)  # 添加元素6
numbers.discard(3)  # 移除元素3
print("After modifications:", numbers)  # 打印修改后的集合

# 集合操作
set_a = {1, 2, 3, 4}  # 集合A
set_b = {3, 4, 5, 6}  # 集合B
print("Union:", set_a | set_b)  # 并集
print("Intersection:", set_a & set_b)  # 交集
print("Difference:", set_a - set_b)  # 差集

# 字符串操作 (String operations)
text = "  Hello, World!  "  # 创建字符串，包含前后空格
print("\nOriginal string:", repr(text))  # 打印原始字符串（使用repr显示引号和空格）
print("Stripped:", repr(text.strip()))  # 打印去除前后空格的字符串
print("Uppercase:", text.upper())  # 打印大写字符串
print("Lowercase:", text.lower())  # 打印小写字符串
print("Split:", text.split(","))  # 按逗号分割字符串
print("Replace:", text.replace("World", "Python"))  # 替换字符串中的"World"为"Python"
print("Starts with 'Hello':", text.startswith("Hello"))  # 检查是否以"Hello"开头
print("Ends with '!':", text.endswith("!"))  # 检查是否以"!"结尾

# 字符串格式化 (String formatting)
name = "Alice"  # 姓名
age = 25  # 年龄
print(f"\nFormatted: {name} is {age} years old.")  # 使用f-string格式化
print("Old style: {} is {} years old.".format(name, age))  # 使用format方法格式化