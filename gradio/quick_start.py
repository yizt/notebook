#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2024-06-15 16:17:27
@ LastEditors: yizuotian
@ LastEditTime: 2024-06-15 16:17:28
@ FilePath: /gradio/quick_start.py
@ Description: Do edit!
"""
import gradio as gr

import time

def trim_words(words, lens):
    trimmed_words = []
    time.sleep(5)
    for w, l in zip(words, lens):
        trimmed_words.append(w[:int(l)])
    return [trimmed_words]

with gr.Blocks() as demo:
    with gr.Row():
        word = gr.Textbox(label="word")
        leng = gr.Number(label="leng")
        output = gr.Textbox(label="Output")
    with gr.Row():
        run = gr.Button()

    event = run.click(trim_words, [word, leng], output, batch=True, max_batch_size=16)

def process_selection(choice):
    # 根据用户的选择返回相应的信息
    if choice == "Option 1":
        return "You selected Option 1."
    elif choice == "Option 2":
        return "You selected Option 2."
    else:
        return "You selected Option 3."




def test():
    """
    @ description: 
    @ param {type} 
    @ return: 
    """
    # demo = gr.Interface(
    # fn=greet,
    # inputs=["text", "slider"],
    # outputs=["text"],
    # )
    # demo = gr.Interface(
    # fn=trim_words, 
    # inputs=["textbox", "number"], 
    # outputs=["output"],
    # batch=True, 
    # max_batch_size=16
    # )

    # 创建 Gradio 界面
    demo = gr.Interface(
        fn=process_selection,  # 指定处理函数
        inputs=gr.Dropdown(['Option 1', 'Option 2', 'Option 3']),  # 使用gr.Dropdown创建下拉菜单，选项为字符串列表
        outputs="text"  # 指定输出类型为文本
    )

    demo.launch(share=False)


def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)




if __name__ == "__main__":
	test()
