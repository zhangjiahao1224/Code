# Variable assignment and arithmetic operations
a = 100
b = 50
try:
    x = int(input("Enter a number: "))
    print(x)
    print(a, b)
    if x == 0:
        print("Error: Cannot divide by zero")
    else:
        print("(a+b)/x =", (a + b) / x)
except ValueError:
    print("Error: Please enter a valid integer")


# Unicode characters
print(ord('张'))
print(chr(24352), end='')
print(chr(22025), end='')
print(chr(35946))

'''
多行注释
可以使用三个单引号或三个双引号
进行多行注释
'''

# File operations (使用 with 自动管理文件关闭)
with open('note.txt', 'w', encoding='utf-8') as fp:
    print('Hello, World!', file=fp)
with open('note.txt', 'r', encoding='utf-8') as fp:
    content = fp.read()
    print(content)

class Student:
    pass

def func():
    pass


with open('text.txt', 'w', encoding='utf-8') as fp:
    fp.write('人生苦短，我用Python')
with open('text.txt', 'r', encoding='utf-8') as fp:
    content = fp.read()
    print(content)
