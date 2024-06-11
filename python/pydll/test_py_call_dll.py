# -*- coding: utf-8 -*-
"""
 @File    : test_py_call_dll.py
 @Time    : 2021/6/1 上午11:22
 @Author  : yizuotian
 @Description    :
"""
import ctypes
from ctypes import *


# 创建结构体对象
class StuStruct(Structure):
    # _fields_是容纳每个结构体成员类型和值的列表，可以配合自动生成fields list和value list的函数使用
    _fields_ = [("name_1", c_char),
                ("class", c_short),
                ("num", c_double),
                ("age", c_int)]
    """
    # 也可以直接初始化，适用于结构体数量不多的情况
    _fields_ = [("name", c_char， b'\x05),
                ("class", c_short, 1),
                ("num", c_double, 10244096),
                ("age", c_int, 2)]
    """


def test_struct():
    stu_value_list = [c_char(b'\x05'), c_short(1), c_double(10244096), c_int(2)]
    stu_obj = StuStruct(*stu_value_list)

    # print("stu name: %s" % stu_obj.name_1)
    # # 这里使用的时候需要注意，结构体成员的名称不能和python内置关键字重复，如果真出现了这种情况。。。
    # # print(stu_obj.`class`)
    # print("stu num: %s" % stu_obj.num)
    # print("stu age: %s" % stu_obj.age)

    struct_so = cdll.LoadLibrary("./libstruct.so.0")
    # 调用函数，根据入参类型需要把结构体转换成对应的指针
    stu_p = pointer(stu_obj)

    struct_so.stu_test(stu_p)


def on_callback(i):
    print('callback {}'.format(i))


def test_callback():
    CMPFUNC = ctypes.CFUNCTYPE(None, ctypes.c_int)
    struct_so = cdll.LoadLibrary("./libstruct.so.0")
    # CMPFUNC(OnPyTestCallback)
    struct_so.register_callback(CMPFUNC(on_callback))
    struct_so.cpp_space_init()


# 创建嵌套结构体对象
class NestStu(Structure):
    _fields_ = [("stu_array1", StuStruct * 2)]


if __name__ == '__main__':
    # test_struct()
    test_callback()

# # 创建StuStruct的数组
# stu_array = StuStruct * 2
# stu_obj_list = [stu_obj, stu_obj]
# # 实例化stu_array
# stu_array_obj = stu_array(*stu_obj_list)
# # 实例化NestStu，因为stu_array1成员是结构体数组类型，只能以同类型的实例进行初始化
# nest_obj = NestStu(stu_array_obj)
# # 打印信息
# print("name: %s" % nest_obj.stu_array1[0].name_1)
# print("num: %s" % nest_obj.stu_array1[0].num)
# print("age: %s" % nest_obj.stu_array1[0].age)
#
# # 载入动态链接库
# struct_so = cdll.LoadLibrary("./libstruct.so.0")
# # 调用函数，根据入参类型需要把结构体转换成对应的指针
# stu_p = pointer(stu_obj)
# nest_stu_p = pointer(nest_obj)
# # ctypes模块提供了快速创建char类型数组的方法
# in_buff = create_string_buffer(b"", size=100)
# rest = struct_so.struct_test(stu_p, nest_stu_p, in_buff)
# # 一般情况下若被调用的函数没有返回值，成功执行后则会返回0，若有其他返回值则返回对应的值
# print("rest: %s" % rest)
