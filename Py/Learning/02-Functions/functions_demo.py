# 02-Functions 函数定义、参数、返回值、递归、lambda 表达式演示
# 演示 Python 中的函数概念：定义函数、参数传递、返回值、递归和匿名函数

# 函数定义与参数返回值
def greet(name):
    """返回一个问候消息"""
    return f"你好，{name}！"

print(greet("Alice"))

# 默认参数值
def power(base, exponent=2):
    """返回 base 的 exponent 次方"""
    return base ** exponent

print("2^3 =", power(2, 3))
print("5^2 =", power(5))  # 使用默认 exponent=2

# 关键字参数
print(power(exponent=3, base=2))  # 使用关键字参数时，顺序无关紧要

# 可变长度参数 (*args, **kwargs)
def sum_all(*args):
    """返回所有参数的和"""
    return sum(args)

def print_info(**kwargs):
    """打印键值对"""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print("和:", sum_all(1, 2, 3, 4))
print_info(name="Alice", age=25, city="New York")

# 递归示例：阶乘
def factorial(n):
    """使用递归返回 n 的阶乘"""
    if n <= 1:
        return 1
    else:
        return n * factorial(n-1)

print("5 的阶乘:", factorial(5))

# Lambda 函数（匿名函数）
square = lambda x: x ** 2
print("5 的平方:", square(5))

# 将 lambda 与内置函数如 map、filter 结合使用
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print("平方后的数字:", squared)

even = list(filter(lambda x: x % 2 == 0, numbers))
print("偶数:", even)

# 使用自定义键进行排序
students = [("Alice", 85), ("Bob", 92), ("Charlie", 78)]
sorted_by_score = sorted(students, key=lambda x: x[1])
print("按分数排序:", sorted_by_score)