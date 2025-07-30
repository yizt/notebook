
from contextlib import contextmanager
from loguru import logger
import traceback

@contextmanager
def temporary_logging():
    handler_id = logger.add("temp.log",level=W)
    try:
        yield
    except Exception as e:
        traceback.print_exc()
    finally:
        logger.remove(handler_id)

def main():
    logger.add("app.log")  # 全局配置
    with temporary_logging():
        logger.info("临时日志写入 temp.log 和 app.log")
    logger.info("这条日志仅写入 app.log")

if __name__=='__main__':
    main()