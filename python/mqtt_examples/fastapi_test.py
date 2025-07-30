#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2025-02-27 08:48:40
@ LastEditors: yizuotian
@ LastEditTime: 2025-02-27 08:48:40
@ FilePath: /mqtt_examples/fastapi_test.py
@ Description: Do edit!
"""

from fastapi import FastAPI
from dataclasses import dataclass
import draccus
import uvicorn
from loguru import logger


@dataclass
class GPTConfig():
    """GPT Model Config"""
    num_layers: int = 12
    num_heads: int = 12
    hidden_size: int = 768

@draccus.wrap()
def create_app(cfg: GPTConfig) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def read_root():
        return {"message": "Hello, FastAPI with Hydra!"}

    # 你可以在这里使用cfg中的配置来配置你的应用
    # 例如，设置应用的host和port（虽然这些通常由uvicorn设置）
    # 但你可以使用这些配置来设置其他参数或初始化其他组件

    return app,cfg

@draccus.wrap()
def create_server(cfg: GPTConfig):
    return BaseServer(cfg)
    

class BaseServer(object):
    def __init__(self,cfg: GPTConfig):
        self.cfg = cfg
        self.app = FastAPI()
    
    def read_root(self):
        return self.cfg.__dict__
    
    def run(self):
        self.app.get("/")(self.read_root)
        self.app.get("/")(self.read_root)
        uvicorn.run(self.app, host='0.0.0.0', port=8000)
        
    

if __name__ == "__main__":
    # 由于FastAPI通常使用uvicorn来运行，我们不能直接在这里调用create_app()
    # 因为@hydra.main装饰器会处理命令行参数并调用create_app()
    # 但为了演示，我们可以手动加载配置并创建应用（不推荐在生产环境中这样做）
    # 正确的做法是使用uvicorn和Hydra的命令行集成（见下一步）
    # 这里只是展示如何手动加载配置

    server = create_server()
    server.run()

    # app,cfg = create_app()
    # logger.debug(cfg)
    # # 注意：下面的代码仅用于演示，不要在生产环境中使用
    # # 你应该使用uvicorn来运行你的应用，如：uvicorn my_module:create_app --reload
    # # 下面的代码只是为了展示如何访问应用实例（但不运行它）
    # import uvicorn
    # # 注意：下面的代码行被注释掉了，因为它会尝试直接运行应用，这不是我们想要的
    # uvicorn.run(app, host='0.0.0.0', port=8000)
    # logger.debug(cfg)
    # 相反，我们应该告诉用户如何正确地使用uvicorn和Hydra来运行应用

    # 正确的运行方式是通过命令行，如下：
    # uvicorn my_module:create_app --reload --config-dir . --config-name config
    # 注意：上面的命令行可能需要根据你的实际文件结构和配置进行调整
    # 特别是`my_module`应该替换为你的Python文件名（不包含.py扩展名）
    # 并且确保该文件包含`create_app`函数

