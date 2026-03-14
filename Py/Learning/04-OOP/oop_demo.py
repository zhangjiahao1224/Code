# 04-OOP demo: classes, encapsulation, inheritance, polymorphism, special methods

# Basic class definition
class BasicDog:
    # Class attribute (shared by all instances)
    species = "Canis familiaris"
    
    # Initializer / Instance attributes
    def __init__(self, name, age):
        self.name = name          # instance attribute
        self.age = age            # instance attribute
    
    # Instance method
    def description(self):
        return f"{self.name} is {self.age} years old"
    
    # Another instance method
    def speak(self, sound):
        return f"{self.name} says {sound}"

# Creating instances
buddy = BasicDog("Buddy", 9)
miles = BasicDog("Miles", 4)

print(buddy.description())
print(miles.description())

# Modifying instance attributes
buddy.age = 10
print(buddy.description())

# Modifying class attribute
BasicDog.species = "Felis catus"
print(buddy.species)  # This will show the changed class attribute for all instances

# Inheritance
# Parent class
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError("Subclass must implement abstract method")

# Child class
class Cat(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # Call parent class constructor
        self.breed = breed
    
    def speak(self):
        return f"{self.name} says Meow!"

# Another child class
class Dog(Animal):
    def __init__(self, name, size):
        super().__init__(name)
        self.size = size
    
    def speak(self):
        return f"{self.name} says Woof!"

# Polymorphism: same interface (speak method) for different classes
pet1 = Cat("Whiskers", "Siamese")
pet2 = Dog("Fido", "Large")

print(pet1.speak())
print(pet2.speak())

# Special methods (dunder methods)
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

v1 = Vector(2, 3)
v2 = Vector(1, 4)
print("v1:", v1)
print("v2:", v2)
print("v1 + v2:", v1 + v2)
print("v1 * 3:", v1 * 3)

# Encapsulation example (using properties)
class Circle:
    def __init__(self, radius):
        self._radius = radius  # protected convention
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value >= 0:
            self._radius = value
        else:
            raise ValueError("Radius cannot be negative")
    
    @property
    def area(self):
        return 3.14159 * self._radius ** 2

c = Circle(5)
print("Radius:", c.radius)
print("Area:", c.area)
c.radius = 10
print("New radius:", c.radius)
print("New area:", c.area)
# c.radius = -5  # This would raise ValueError