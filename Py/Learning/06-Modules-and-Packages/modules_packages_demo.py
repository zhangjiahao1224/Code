# 06-Modules-and-Packages demo: importing modules, packages, standard library overview, virtual environments

# Importing entire module
import math
print("Square root of 16:", math.sqrt(16))
print("Pi:", math.pi)
print("Factorial of 5:", math.factorial(5))

# Importing specific functions from module
from random import randint, choice
print("Random integer between 1 and 10:", randint(1, 10))
print("Random choice from list:", choice(['apple', 'banana', 'cherry']))

# Importing module with alias
import numpy as np  # would need numpy installed
# print("NumPy array:", np.array([1, 2, 3]))

# Importing everything from module (not recommended)
# from math import *  # imports all functions into current namespace
# print("sqrt(25):", sqrt(25))  # can use directly without math prefix

# Creating and using a simple module
# Note: We create a temporary module file for demonstration purposes.
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

# Now import and use the module
# LSP may not recognize dynamically created files, but this works when running
import utils
print("Using custom module:")
print("utils.add(5, 3):", utils.add(5, 3))
print("utils.multiply(4, 6):", utils.multiply(4, 6))
print("utils.PI:", utils.PI)

calc = utils.Calculator()
print("Calculator.add(10, 20):", calc.add(10, 20))
print("History:", calc.history)

# Exploring standard library (overview)
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

# JSON example
data = {"name": "Alice", "age": 25, "city": "New York"}
json_string = json.dumps(data, indent=2)
print("JSON data:")
print(json_string)

# Clean up created file
import os
if os.path.exists('utils.py'):
    os.remove('utils.py')

# Virtual environment note:
# To create a virtual environment:
#   python -m venv myenv
# To activate:
#   Windows: myenv\Scripts\activate
#   Unix/MacOS: source myenv/bin/activate
# To install packages:
#   pip install package_name