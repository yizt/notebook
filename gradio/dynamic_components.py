#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2024-06-15 17:44:23
@ LastEditors: yizuotian
@ LastEditTime: 2024-06-15 17:44:24
@ FilePath: /gradio/dynamic_components.py
@ Description: Do edit!
"""

import gradio as gr

with gr.Blocks() as demo:
    input_text = gr.Textbox(label="input")

    @gr.render(inputs=input_text)
    def show_split(text):
        if len(text) == 0:
            gr.Markdown("## No Input Provided")
        else:
            for letter in text:
                gr.Textbox(letter,label=letter)



def test():
	"""
	@ description: 
	@ param {type} 
	@ return: 
	"""
	demo.launch()

if __name__ == "__main__":
	test()
