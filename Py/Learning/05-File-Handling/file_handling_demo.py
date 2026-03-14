# 05-File-Handling demo: reading/writing files, CSV, JSON, path operations

import os
import json
import csv
from pathlib import Path

# --- Basic file operations ---
# Writing to a file
with open('example.txt', 'w', encoding='utf-8') as f:
    f.write('Hello, World!\n')
    f.write('This is a test file.\n')

# Reading from a file
with open('example.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    print("File content:")
    print(content)

# Reading line by line
print("\nReading line by line:")
with open('example.txt', 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        print(f"{line_num}: {line.strip()}")

# --- Working with paths ---
current_dir = Path('.')
print(f"\nCurrent directory: {current_dir.absolute()}")

# Creating a new directory
new_dir = current_dir / 'test_dir'
new_dir.mkdir(exist_ok=True)
print(f"Created directory: {new_dir}")

# Creating a file in the new directory
test_file = new_dir / 'test.txt'
test_file.write_text('This is a test file in a subdirectory.', encoding='utf-8')
print(f"Created file: {test_file}")
print(f"File content: {test_file.read_text(encoding='utf-8')}")

# --- CSV operations ---
csv_file = 'data.csv'
# Writing CSV
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Age', 'City'])
    writer.writerow(['Alice', 25, 'New York'])
    writer.writerow(['Bob', 30, 'London'])
    writer.writerow(['Charlie', 35, 'Tokyo'])

# Reading CSV
print("\nCSV data:")
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)

# --- JSON operations ---
json_file = 'data.json'
# Writing JSON
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

# Reading JSON
with open(json_file, 'r', encoding='utf-8') as f:
    loaded_data = json.load(f)
    print("\nJSON data:")
    print(json.dumps(loaded_data, indent=2, ensure_ascii=False))

# --- File and directory operations ---
print("\nFile and directory info:")
print(f"Does example.txt exist? {Path('example.txt').exists()}")
print(f"Is test_dir a directory? {new_dir.is_dir()}")
print(f"Size of example.txt: {Path('example.txt').stat().st_size} bytes")

# Listing files in current directory
print("\nFiles in current directory:")
for item in current_dir.iterdir():
    if item.is_file():
        print(f"  File: {item.name}")
    elif item.is_dir():
        print(f"  Directory: {item.name}")

# Clean up (optional, uncomment if you want to delete the created files)
# os.remove('example.txt')
# os.remove(csv_file)
# os.remove(json_file)
# import shutil
# shutil.rmtree(new_dir)