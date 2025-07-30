#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2025-02-19 11:36:51
@ LastEditors: yizuotian
@ LastEditTime: 2025-02-19 11:36:52
@ FilePath: /mqtt_examples/dataclass_test.py
@ Description: Do edit!
"""

from dataclasses import dataclass

# 定义一个 dataclass
@dataclass
class Person:
    name: str
    age: int
    is_active: bool
def test():
    # 假设有一个包含多余字段的字典
    config_dict = {
        "name": "Alice",
        "age": 30,
        "is_active": True,
        "extra_field": "This is not needed"
    }

    # 提取与 dataclass 字段匹配的子集
    fields = {field for field in Person.__dataclass_fields__}
    filtered_dict = {key: value for key, value in config_dict.items() if key in fields}

    # 使用子集构造 dataclass
    person = Person(**filtered_dict)
    person = Person(**config_dict)
    print(person)

if __name__ == "__main__":
	test()
