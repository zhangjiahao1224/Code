# 08-标准库演示：常用标准库模块

import os
import sys
import json
import csv
import datetime
import time
import random
import math
import statistics
from collections import Counter, defaultdict, deque
import itertools
import re
import urllib.request
import urllib.parse

print("=== Python Standard Library Demo ===\n")

# --- OS模块 ---
print("1. OS Module:")
print(f"   Current working directory: {os.getcwd()}")
print(f"   OS name: {os.name}")
print(f"   Path separator: {os.sep}")
print(f"   List directory: {os.listdir('.')[:5]}...")  # 前5个项目
print()

# --- Sys模块 ---
print("2. Sys Module:")
print(f"   Python version: {sys.version}")
print(f"   Platform: {sys.platform}")
print(f"   Version info: {sys.version_info}")
print()

# --- DateTime模块 ---
print("3. DateTime Module:")
now = datetime.datetime.now()
print(f"   Current date and time: {now}")
print(f"   Today's date: {datetime.date.today()}")
print(f"   Time only: {now.time()}")
# Timedelta示例
delta = datetime.timedelta(days=7, hours=3)
future_date = now + delta
print(f"   One week and 3 hours from now: {future_date}")
print()

# --- Time模块 ---
print("4. Time Module:")
timestamp = time.time()
print(f"   Current timestamp: {timestamp}")
print(f"   Formatted time: {time.ctime(timestamp)}")
print(f"   Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}")
print()

# --- Random模块 ---
print("5. Random Module:")
print(f"   Random float between 0 and 1: {random.random():.4f}")
print(f"   Random integer between 1 and 10: {random.randint(1, 10)}")
print(f"   Random choice from list: {random.choice(['apple', 'banana', 'cherry'])}")
numbers = list(range(1, 11))
random.shuffle(numbers)
print(f"   Shuffled list: {numbers}")
print(f"   Random sample of 3: {random.sample(numbers, 3)}")
print()

# --- Math模块 ---
print("6. Math Module:")
print(f"   sqrt(144): {math.sqrt(144)}")
print(f"   factorial(5): {math.factorial(5)}")
print(f"   ceil(4.2): {math.ceil(4.2)}")
print(f"   floor(4.8): {math.floor(4.8)}")
print(f"   pi: {math.pi}")
print(f"   sin(pi/2): {math.sin(math.pi/2):.4f}")
print()

# --- Statistics模块 ---
print("7. Statistics Module:")
data = [2.5, 3.7, 2.8, 3.4, 2.1, 3.6, 2.9]
print(f"   Data: {data}")
print(f"   Mean: {statistics.mean(data):.2f}")
print(f"   Median: {statistics.median(data):.2f}")
print(f"   Stdev: {statistics.stdev(data):.2f}")
print(f"   Variance: {statistics.variance(data):.2f}")
print()

# --- Collections模块 ---
print("8. Collections Module:")
# Counter
words = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
word_counts = Counter(words)
print(f"   Counter: {dict(word_counts)}")
print(f"   Most common: {word_counts.most_common(2)}")

# DefaultDict
dd = defaultdict(list)
dd['fruits'].append('apple')
dd['fruits'].append('banana')
dd['vegetables'].append('carrot')
print(f"   DefaultDict: {dict(dd)}")

# Deque
dq = deque([1, 2, 3])
dq.appendleft(0)
dq.append(4)
print(f"   Deque: {list(dq)}")
print()

# --- Itertools模块 ---
print("9. Itertools Module:")
# chain
list1 = [1, 2, 3]
list2 = [4, 5, 6]
chained = list(itertools.chain(list1, list2))
print(f"   chain: {chained}")
# combinations
items = ['A', 'B', 'C', 'D']
combos = list(itertools.combinations(items, 2))
print(f"   combinations of 2: {combos}")
# permutations
perms = list(itertools.permutations(['A', 'B', 'C'], 2))
print(f"   permutations of 2: {perms}")
# product (Cartesian product)
product_result = list(itertools.product([1, 2], ['a', 'b']))
print(f"   product: {product_result}")
print()

# --- 正则表达式 ---
print("10. Regular Expressions (re):")
text = "Contact us at support@example.com or sales@company.org"
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
emails = re.findall(email_pattern, text)
print(f"   Text: {text}")
print(f"   Found emails: {emails}")
# 替换
censored = re.sub(email_pattern, '[EMAIL REDACTED]', text)
print(f"   After substitution: {censored}")
print()

# --- JSON模块 ---
print("11. JSON Module:")
data = {
    "name": "John Doe",
    "age": 30,
    "is_student": False,
    "courses": ["Math", "Physics"],
    "address": {
        "street": "123 Main St",
        "city": "Anytown"
    }
}
json_str = json.dumps(data, indent=2)
print(f"   JSON string:\n{json_str}")
parsed = json.loads(json_str)
print(f"   Parsed name: {parsed['name']}")
print()

# --- CSV模块 ---
print("12. CSV Module:")
# 写入CSV
with open('sample.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Age', 'City'])
    writer.writerow(['Alice', 25, 'New York'])
    writer.writerow(['Bob', 30, 'London'])
    writer.writerow(['Charlie', 35, 'Tokyo'])

# 读取CSV
print("   CSV contents:")
with open('sample.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"     {row}")
# 清理
import os
if os.path.exists('sample.csv'):
    os.remove('sample.csv')
print()

# --- Urllib (基本HTTP请求示例) ---
print("13. Urllib (HTTP request):")
try:
    # 示例：获取公共API (httpbin.org)
    with urllib.request.urlopen('https://httpbin.org/json', timeout=5) as response:
        data = json.loads(response.read().decode())
        print(f"   Successfully fetched JSON data")
        print(f"   Slideshow title: {data.get('slideshow', {}).get('title', 'N/A')}")
except Exception as e:
    print(f"   Could not fetch remote data (likely network issue): {e}")
    print("   This is normal in offline environments")
print()

print("=== End of Standard Library Demo ===")