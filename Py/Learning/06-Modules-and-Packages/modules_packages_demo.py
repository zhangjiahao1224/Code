# 06-模块和包演示：导入模块、包、标准库概述、虚拟环境

# 导入整个模块
import math
print("Square root of 16:", math.sqrt(16))
print("Pi:", math.pi)
print("Factorial of 5:", math.factorial(5))

# 从模块导入特定函数
from random import randint, choice
print("Random integer between 1 and 10:", randint(1, 10))
print("Random choice from list:", choice(['apple', 'banana', 'cherry']))

# 使用别名导入模块
import numpy as np  # 需要安装numpy
# print("NumPy array:", np.array([1, 2, 3]))

# 从模块导入所有内容（不推荐）
# from math import *  # 将所有函数导入当前命名空间
# print("sqrt(25):", sqrt(25))  # 可以直接使用，无需math前缀

# 创建和使用简单模块
# 注意：我们为演示目的创建临时模块文件。
with open('utils.py', 'w') as f:
    f.write('''
def add(a, b):
    """Add two numbers."""
    return a + b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

PI = 3.14159

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
''')

# 现在导入并使用模块
# LSP可能无法识别动态创建的文件，但在运行时有效
import Py.Utils.utils as utils
print("Using custom module:")
print("utils.add(5, 3):", utils.add(5, 3))
print("utils.multiply(4, 6):", utils.multiply(4, 6))
print("utils.PI:", utils.PI)

calc = utils.Calculator()
print("Calculator.add(10, 20):", calc.add(10, 20))
print("History:", calc.history)

# 探索标准库（概述）
import os
import sys
import datetime
import json

print("\n--- Standard Library Overview ---")
print("OS name:", os.name)
print("Current directory:", os.getcwd())
print("Python version:", sys.version)
print("Today's date:", datetime.date.today())
print("Current time:", datetime.datetime.now().strftime("%H:%M:%S"))

# JSON示例
data = {"name": "Alice", "age": 25, "city": "New York"}
json_string = json.dumps(data, indent=2)
print("JSON data:")
print(json_string)

# 清理创建的文件
import os
if os.path.exists('utils.py'):
    os.remove('utils.py')

# 虚拟环境说明：
# 创建虚拟环境：
#   python -m venv myenv
# 激活：
#   Windows: myenv\Scripts\activate
#   Unix/MacOS: source myenv/bin/activate
# 安装包：
#   pip install package_name