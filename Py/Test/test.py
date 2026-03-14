# Variable assignment and arithmetic operations
a = 100
b = 50
# 增加异常处理：避免输入非数字导致程序崩溃
try:
    x = int(input("Enter a number: "))
    print(x)
    print(a, b)
    print("(a+b)/x =", (a + b) / x)
except ValueError:
    print("错误：请输入有效的整数！")


# Unicode characters
print(ord('张'))  # 输出'张'的Unicode编码
print(chr(24352), end='')
print(chr(22025), end='')
print(chr(35946))


'''
多行注释
可以使用三个单引号或三个双引号
进行多行注释
'''

# File operations - 优化1：note.txt添加utf-8编码，避免乱码
with open('note.txt', 'w', encoding='utf-8') as fp:  # 替代手动close，指定编码
    print('Hello, World!', file=fp)

with open('note.txt', 'r', encoding='utf-8') as fp:  # 读取也指定编码，统一标准
    content = fp.read()
    print(content)

# 基础类定义（空类，语法合法）
class Student:
    pass

# 基础函数定义（空函数，语法合法）
def func():
    pass

# File operations - 优化2：text.txt读取时也指定编码，彻底避免乱码
with open('text.txt', 'w', encoding='utf-8') as fp:
    fp.write('人生苦短，我用Python')

with open('text.txt', 'r', encoding='utf-8') as fp:  # 关键：读取时也指定utf-8
    content = fp.read()
    print(content)