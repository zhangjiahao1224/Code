# 04-面向对象编程演示：类、封装、继承、多态、特殊方法
# 基本类定义
class BasicDog:
    # 类属性（所有实例共享）
    species = "Canis familiaris"

    # 初始化器 / 实例属性
    def __init__(self, name, age):
        self.name = name          # 实例属性
        self.age = age            # 实例属性

    # 实例方法
    def description(self):
        return f"{self.name} is {self.age} years old"

    # 另一个实例方法
    def speak(self, sound):
        return f"{self.name} says {sound}"

# 创建实例
buddy = BasicDog("Buddy", 9) # 创建一个名为Buddy的BasicDog实例，年龄为9岁
miles = BasicDog("Miles", 4) # 创建另一个名为Miles的BasicDog实例，年龄为4岁

print(buddy.description())
print(miles.description())

# 修改实例属性
buddy.age = 10 
print(buddy.description())

# 修改类属性
BasicDog.species = "Felis catus"
print(buddy.species)  # 这将显示所有实例的更改后的类属性

# 继承
# 父类
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("子类必须实现抽象方法")

# 子类
class Cat(Animal):
    def __init__(self, name, breed): # 子类构造函数，接受额外的breed参数
        super().__init__(name)  # 调用父类构造函数
        self.breed = breed

    def speak(self):
        return f"{self.name} says Meow!"

# 另一个子类
class Dog(Animal):
    def __init__(self, name, size):
        super().__init__(name)
        self.size = size

    def speak(self):
        return f"{self.name} says Woof!"

# 多态：不同类的相同接口（speak方法）
pet1 = Cat("Whiskers", "Siamese")
pet2 = Dog("Fido", "Large")

print(pet1.speak())
print(pet2.speak())

# 特殊方法（双下划线方法）
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

# 封装示例（使用属性）
class Circle:
    def __init__(self, radius):
        self._radius = radius  # 保护约定

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value >= 0:
            self._radius = value
        else:
            raise ValueError("半径不能为负数")

    @property
    def area(self):
        return 3.14159 * self._radius ** 2

c = Circle(5)
print("Radius:", c.radius)
print("Area:", c.area)
c.radius = 10
print("New radius:", c.radius)
print("New area:", c.area)
# c.radius = -5  # 这会引发ValueError