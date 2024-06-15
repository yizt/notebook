#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2024-06-15 17:06:33
@ LastEditors: yizuotian
@ LastEditTime: 2024-06-15 17:06:33
@ FilePath: /gradio/dynamic_interface.py
@ Description: Do edit!
"""
import gradio as gr

def dynamic_interface(num_items):
    # 创建一个 Blocks 对象
    with gr.Blocks() as demo:
        # 动态生成文本框列表
        items = []
        for i in range(num_items):
            item = gr.Textbox(f"Item {i + 1}")
            items.append(item)
        
        # 创建一个按钮，点击时将输出所有文本框的内容
        with gr.Column():
            submit_button = gr.Button("Submit")
            submit_button.click(lambda x:str(x), inputs=items, outputs="text")

        # 设置输出组件
        output = gr.Textbox()

    # 定义当按钮点击时的回调函数
    def on_submit(*args):
        # 将所有文本框的内容以列表形式返回
        return ", ".join(args)

    # 将回调函数与按钮的点击事件关联
    submit_button.click(on_submit, inputs=items, outputs=output)

    return demo



def test():
    """
    @ description: 
    @ param {type} 
    @ return: 
    """
    # 创建一个下拉菜单，用于选择要生成的文本框数量
    num_items_selector = gr.Dropdown(
    choices=[1, 2, 3, 4, 5],
    value=1,
    label="Number of items",
    elem_id="num_items"
    )

    # 使用gr.Interface创建界面，并传入动态生成组件的函数
    iface = gr.Interface(
    fn=dynamic_interface,
    inputs=num_items_selector,
    outputs="text",
    title="Dynamic Interface Example",
    description="Select the number of items to generate text boxes for."
    )

    # 启动界面
    iface.launch()

if __name__ == "__main__":
	test()
