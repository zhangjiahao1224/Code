# 07-Exception-Handling demo: try-except, custom exceptions, assertions, logging

# Basic exception handling
try:
    x = int(input("Enter a number: "))  # This will fail in non-interactive environment, so we'll simulate
    result = 10 / x
except ValueError:
    print("Error: Please enter a valid integer")
except ZeroDivisionError:
    print("Error: Cannot divide by zero")
else:
    print("Result:", result)
finally:
    print("This block always executes")

# Simulating input for demonstration
print("\n--- Simulating exceptions ---")
try:
    # Simulate ValueError
    x = int("not_a_number")
except ValueError as e:
    print(f"Caught ValueError: {e}")

try:
    # Simulate ZeroDivisionError
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Caught ZeroDivisionError: {e}")

# Raising exceptions
def check_positive(number):
    if number < 0:
        raise ValueError("Number must be positive")
    return number

try:
    check_positive(-5)
except ValueError as e:
    print(f"Caught ValueError: {e}")

# Custom exceptions
class CustomError(Exception):
    """A custom exception type"""
    pass

def do_something_risky():
    raise CustomError("Something went wrong!")

try:
    do_something_risky()
except CustomError as e:
    print(f"Caught CustomError: {e}")

# Assertions
def divide(a, b):
    assert b != 0, "Divisor cannot be zero"
    return a / b

try:
    print(divide(10, 2))
    print(divide(10, 0))  # This will raise AssertionError
except AssertionError as e:
    print(f"AssertionError: {e}")

# Logging example
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")

try:
    result = 10 / 0
except Exception as e:
    logger.exception("An error occurred: %s", e)