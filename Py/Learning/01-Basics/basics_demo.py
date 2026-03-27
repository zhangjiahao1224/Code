# 变量和数据类型
age = 18                 # 整型 (integer)
height = 1.75            # 浮点型 (float)
name = "Alice"           # 字符串 (string)
is_student = True        # 布尔型 (boolean)0/1

print("Name:", name)
print("Age:", age)
print("Height:", height)
print("Is student:", is_student)

# 输入演示
print("\n=== 输入演示 ===")

# 基本输入
user_name = input("请输入您的名字：")
user_age = int(input("请输入您的年龄："))  # 注意：input() 返回字符串，需要转换类型

print(f"您好，{user_name}！您今年 {user_age} 岁。")

# 使用输入进行计算
num1 = float(input("请输入第一个数字："))
num2 = float(input("请输入第二个数字："))
operation = input("请输入运算符 (+, -, *, /)：")

if operation == "+":
    result = num1 + num2
elif operation == "-": 
    result = num1 - num2
elif operation == "*":
    result = num1 * num2
elif operation == "/":
    if num2 != 0:
        result = num1 / num2
    else:
        result = "除数不能为零"
else:
    result = "无效运算符"

print(f"结果：{result}")

# 算术运算符
a = 10
b = 3
print(a)
print(b)
print("a + b =", a + b)   # 加法
print("a - b =", a - b)   # 减法
print("a * b =", a * b)   # 乘法
print("a / b =", a / b)   # 除法（返回浮点数）
print("a // b =", a // b) # 地板除法（返回整数）
print("a % b =", a % b)   # 取模（余数）
print("a ** b =", a ** b) # 指数运算

# 比较运算符
print("a == b:", a == b)  # 等于
print("a != b:", a != b)  # 不等于
print("a > b:", a > b)    # 大于
print("a < b:", a < b)    # 小于
print("a >= b:", a >= b)  # 大于等于
print("a <= b:", a <= b)  # 小于等于

# 逻辑运算符
print("True and False:", True and False)  # 逻辑与
print("True or False:", True or False)    # 逻辑或
print("not True:", not True)              # 逻辑非

# 条件语句：if-elif-else
score = 85
if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
elif score >= 70:
    print("Grade: C")
else:
    print("Grade: D")

# 循环：for 循环
print("\nFor loop:")
for i in range(5):
    print(i, end=' ')
print()  # 换行

# 循环：while 循环
print("\nWhile loop:")
count = 0
while count < 3:
    print(count, end=' ')
    count += 1
print()  # 换行

# 循环控制：break 和 continue
print("\nBreak and Continue example:")
for i in range(10):
    if i == 3:
        continue  # 跳过当前迭代（不执行后续代码，直接进入下次循环）
    if i == 8:
        break     # 完全退出循环
    print(i, end=' ')
print()  # 换行