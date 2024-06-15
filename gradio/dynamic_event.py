# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2024-06-15 17:32:45
@ LastEditors: yizuotian
@ LastEditTime: 2024-06-15 17:32:45
@ FilePath: /gradio/dynamic_event.py
@ Description: Do edit!
"""

import gradio as gr

with gr.Blocks() as demo:
    text_count = gr.State(1)
    add_btn = gr.Button("Add Box")
    add_btn.click(lambda x: x + 1, text_count, text_count)

    

    @gr.render(inputs=text_count)
    def render_count(count):
        boxes = []
        for i in range(count):
            box = gr.Textbox(key=i, label=f"Box {i}")
            boxes.append(box)

        def merge(*args):
            return " ".join(args)
        
        merge_btn.click(merge, boxes, output)

    merge_btn = gr.Button("Merge")
    output = gr.Textbox(label="Merged Output")


    



def test():
	"""
	@ description: 
	@ param {type} 
	@ return: 
	"""
	demo.launch()

if __name__ == "__main__":
	test()
