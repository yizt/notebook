#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2024-06-15 16:46:03
@ LastEditors: yizuotian
@ LastEditTime: 2024-06-15 16:46:04
@ FilePath: /gradio/df_plot.py
@ Description: Do edit!
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def fig2im(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im

def fig2im2(fig):
    with BytesIO() as buff:
        fig.savefig(buff, format='png')
        buff.seek(0)
        im = plt.imread(buff)
    return im

def fig2im3(fig):
    with BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im



# 定义一个处理函数，该函数接受 DataFrame 并返回图表
def plot_data(num):
    # 将输入的 DataFrame 转换为 pandas DataFrame
    df = np.arange(num[0])
    pd_df = pd.DataFrame(df)
    
    # 使用 Matplotlib 创建图表
    fig=plt.figure(figsize=(10, 6))
    for column in pd_df.columns:
        plt.plot(pd_df.index, pd_df[column], label=column)
    
    plt.title('Data Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    
    # # 将图表保存到内存中的 BytesIO 对象
    # buf = BytesIO()
    # plt.savefig(buf, format='png')
    # plt.close()
    # buf.seek(0)
    # img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    # # 将 BytesIO 对象转换为 base64 编码的字符串，以便在 Gradio 界面中显示
    # # chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    # print(img_arr.shape)
    # # 返回图像数据
    # return img_arr
    return fig2im(fig)



def test():
    """
    @ description: 
    @ param {type} 
    @ return: 
    """
    # 创建 Gradio 界面
    # iface = gr.Interface(
    #     fn=plot_data,  # 指定处理函数
    #     inputs=gr.Dataframe(label="Enter your data"),  # 使用gr.Dataframe创建表格输入
    #     outputs=gr.Image(label="Plot")  # 指定输出类型为图像
    # )
    
    # iface = gr.Interface(
    #     fn=plot_data,  # 指定处理函数
    #     inputs=gr.Dropdown([10, 20, 30]),  # 使用gr.Dataframe创建表格输入
    #     outputs=gr.Image(label="Plot")  # 指定输出类型为图像
    # )

    iface = gr.Interface(
        fn=plot_data,  # 指定处理函数
        inputs=gr.CheckboxGroup([10, 20, 30,"数据科学"]),  # 使用gr.Dataframe创建表格输入
        outputs=gr.Image(label="Plot")  # 指定输出类型为图像
    )

    

    # 启动界面
    iface.launch()

if __name__ == "__main__":
	test()
