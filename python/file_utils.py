# -*- coding: utf-8 -*-
"""
 @File    : file_utils.py
 @Time    : 2022/2/9 上午10:06
 @Author  : yizuotian
 @Description    :
"""
import os


def get_sub_files(dir_path, recursive=True):
    """

    :param dir_path:
    :param recursive:
    :return:
    """
    file_path_list = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if file_name.startswith("."):
            continue
        if os.path.isdir(file_path):
            # print(file_path)
            if recursive:
                file_path_list.extend(get_sub_files(file_path, True))
        else:
            file_path_list.append(file_path)
    file_path_list.sort()
    return file_path_list


if __name__ == '__main__':
    print(len(get_sub_files("./..")))
    print(len(get_sub_files("./..", False)))
    print(get_sub_files("./../big_data", True))
    print(get_sub_files("../", True))
