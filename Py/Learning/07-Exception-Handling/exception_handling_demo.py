# 07-异常处理演示：try-except、自定义异常、断言、日志记录

# 基本异常处理
try:
    x = int(input("Enter a number: "))  # 这将在非交互环境中失败，所以我们模拟
    result = 10 / x
except ValueError:
    print("错误：请输入有效的整数")
except ZeroDivisionError:
    print("错误：不能除以零")
else:
    print("结果:", result)
finally:
    print("此块总是执行")

# 模拟输入以进行演示
print("\n--- 模拟异常 ---")
try:
    # 模拟ValueError
    x = int("not_a_number")
except ValueError as e:
    print(f"Caught ValueError: {e}")

try:
    # 模拟ZeroDivisionError
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Caught ZeroDivisionError: {e}")

# 引发异常
def check_positive(number):
    if number < 0:
        raise ValueError("数字必须为正数")
    return number

try:
    check_positive(-5)
except ValueError as e:
    print(f"Caught ValueError: {e}")

# 自定义异常
class CustomError(Exception):
    """自定义异常类型"""
    pass

def do_something_risky():
    raise CustomError("出错了！")

try:
    do_something_risky()
except CustomError as e:
    print(f"Caught CustomError: {e}")

# 断言
def divide(a, b):
    assert b != 0, "除数不能为零"
    return a / b

try:
    print(divide(10, 2))
    print(divide(10, 0))  # 这将引发AssertionError
except AssertionError as e:
    print(f"AssertionError: {e}")

# 日志记录示例
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("这是一个信息消息")
logger.warning("这是一个警告消息")
logger.error("这是一个错误消息")

try:
    result = 10 / 0
except Exception as e:
    logger.exception("发生错误: %s", e)