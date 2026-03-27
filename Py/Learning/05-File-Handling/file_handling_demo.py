# 05-文件处理演示：读取/写入文件、CSV、JSON、路径操作

import os
import json
import csv
from pathlib import Path

# --- 基本文件操作 ---
# 写入文件
with open('example.txt', 'w', encoding='utf-8') as f:
    f.write('Hello, World!\n')
    f.write('This is a test file.\n')

# 从文件读取
with open('example.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    print("File content:")
    print(content)

# 逐行读取
print("\nReading line by line:")
with open('example.txt', 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        print(f"{line_num}: {line.strip()}")

# --- 路径操作 ---
current_dir = Path('.')
print(f"\nCurrent directory: {current_dir.absolute()}")

# 创建新目录
new_dir = current_dir / 'test_dir'
new_dir.mkdir(exist_ok=True)
print(f"Created directory: {new_dir}")

# 在新目录中创建文件
test_file = new_dir / 'test.txt'
test_file.write_text('This is a test file in a subdirectory.', encoding='utf-8')
print(f"Created file: {test_file}")
print(f"File content: {test_file.read_text(encoding='utf-8')}")

# --- CSV操作 ---
csv_file = 'data.csv'
# 写入CSV
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Age', 'City'])
    writer.writerow(['Alice', 25, 'New York'])
    writer.writerow(['Bob', 30, 'London'])
    writer.writerow(['Charlie', 35, 'Tokyo'])

# 读取CSV
print("\nCSV data:")
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)

# --- JSON操作 ---
json_file = 'data.json'
# 写入JSON
data = {
    'students': [
        {'name': 'Alice', 'age': 25, 'major': 'CS'},
        {'name': 'Bob', 'age': 22, 'major': 'Math'}
    ],
    'course': 'Introduction to Programming',
    'semester': 'Fall 2025'
}

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# 读取JSON
with open(json_file, 'r', encoding='utf-8') as f:
    loaded_data = json.load(f)
    print("\nJSON data:")
    print(json.dumps(loaded_data, indent=2, ensure_ascii=False))

# --- 文件和目录操作 ---
print("\nFile and directory info:")
print(f"Does example.txt exist? {Path('example.txt').exists()}")
print(f"Is test_dir a directory? {new_dir.is_dir()}")
print(f"Size of example.txt: {Path('example.txt').stat().st_size} bytes")

# 列出当前目录中的文件
print("\nFiles in current directory:")
for item in current_dir.iterdir():
    if item.is_file():
        print(f"  File: {item.name}")
    elif item.is_dir():
        print(f"  Directory: {item.name}")

# 清理（可选，如果要删除创建的文件则取消注释）
# os.remove('example.txt')
# os.remove(csv_file)
# os.remove(json_file)
# import shutil
# shutil.rmtree(new_dir)