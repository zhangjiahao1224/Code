# 03-Data-Structures demo: lists, tuples, dictionaries, sets, string operations

# Lists
fruits = ["apple", "banana", "cherry"]
print("Fruits:", fruits)
print("First fruit:", fruits[0])
print("Last fruit:", fruits[-1])

# Modify list
fruits.append("orange")
fruits.insert(1, "blueberry")
print("After modifications:", fruits)

# Remove items
fruits.remove("banana")
popped = fruits.pop()  # removes and returns last item
print("After removals:", fruits)
print("Popped item:", popped)

# List operations
print("Length:", len(fruits))
print("Sorted:", sorted(fruits))
print("Reversed:", list(reversed(fruits)))

# Tuples (immutable)
coordinates = (10.0, 20.0)
print("\nCoordinates:", coordinates)
# coordinates[0] = 15.0  # This would cause an error

# Dictionaries
student = {
    "name": "John Doe",
    "age": 20,
    "major": "Computer Science",
    "grades": [85, 92, 78]
}
print("\nStudent info:")
print("Name:", student["name"])
print("Age:", student["age"])
print("Major:", student["major"])

# Modify dictionary
student["year"] = "Sophomore"
student["grades"].append(96)
print("Updated student:", student)

# Dictionary methods
print("Keys:", list(student.keys()))
print("Values:", list(student.values()))
print("Items:", list(student.items()))

# Sets (unique elements)
numbers = {1, 2, 3, 4, 5, 5, 3}  # duplicates removed
print("\nSet of numbers:", numbers)
numbers.add(6)
numbers.discard(3)
print("After modifications:", numbers)

# Set operations
set_a = {1, 2, 3, 4}
set_b = {3, 4, 5, 6}
print("Union:", set_a | set_b)
print("Intersection:", set_a & set_b)
print("Difference:", set_a - set_b)

# String operations
text = "  Hello, World!  "
print("\nOriginal string:", repr(text))
print("Stripped:", repr(text.strip()))
print("Uppercase:", text.upper())
print("Lowercase:", text.lower())
print("Split:", text.split(","))
print("Replace:", text.replace("World", "Python"))
print("Starts with 'Hello':", text.startswith("Hello"))
print("Ends with '!':", text.endswith("!"))

# String formatting
name = "Alice"
age = 25
print(f"\nFormatted: {name} is {age} years old.")
print("Old style: {} is {} years old.".format(name, age))