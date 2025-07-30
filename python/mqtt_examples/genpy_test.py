"""
@ Author: yizuotian
@ Date: 2025-03-15 15:13:18
@ LastEditors: yizuotian
@ LastEditTime: 2025-03-15 15:13:19
@ FilePath: /mqtt_examples/genpy_test.py
@ Description: Do edit!
"""
from loguru import logger

def test():
    """
    @ description: 
    @ param {type} 
    @ return: 
    """
    import genpy

    # 假设我们有一个genpy.Time对象
    # 通常，你会从ROS的消息或其他地方得到这个对象
    # 这里我们只是为了演示而手动创建一个
    time_obj = genpy.rostime.Time(secs=1633046400, nsecs=0)  # 示例时间，2021年10月1日00:00:00 UTC

    # 转换为时间戳
    ts = time_obj.to_nsec()
    logger.debug(f"{ts=},{type(ts)=}")


if __name__ == "__main__":
	test()
