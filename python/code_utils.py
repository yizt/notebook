# -*- coding: utf-8 -*-
"""
 @File    : code_utils.py
 @Time    : 2022/2/9 上午9:50
 @Author  : yizuotian
 @Description    : 代码统计，行数统计
"""
import codecs

from file_utils import get_sub_files


def number_of_lines(file_path):
    """

    :param file_path:
    :return:
    """
    with codecs.open(file_path, encoding='utf-8') as f:
        lines = f.readlines()

    return len(lines)


def is_suffix_match(file_path: str, suffix_list):
    """

    :param file_path:
    :param suffix_list:
    :return:
    """
    for suffix in suffix_list:
        if file_path.endswith(suffix):
            return True
    return False


def stat_code_lines(dir_path, suffix_list):
    """

    :param dir_path:
    :param suffix_list:
    :return:
    """
    files = get_sub_files(dir_path, recursive=True)
    code_lines = [number_of_lines(file) for file in files
                  if is_suffix_match(file, suffix_list)]
    return sum(code_lines)


def mistake_proof_project():
    """
    工步防错项目代码统计
    :return:
    """
    py_dir = "/Users/yizuotian/pyspace/control_flow"
    c_dir = "/Users/yizuotian/cspace/sophon-hdmi"
    c_reuse_dir = "/Users/yizuotian/cspace/sophon-hdmi/include"
    result = "python 代码行数:{}\n" \
             "c++ 代码行数:{}, 复用行数:{} ".format(stat_code_lines(py_dir, [".py"]),
                                            stat_code_lines(c_dir, [".h", ".c", ".cpp"]),
                                            stat_code_lines(c_reuse_dir, [".h", ".c", ".cpp"]))
    print(result)


if __name__ == "__main__":
    print(stat_code_lines("./", [".py"]))
    mistake_proof_project()
