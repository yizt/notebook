import numpy as np
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory

def worker(shm_name, shape, dtype):
    # 连接到已存在的共享内存
    shm = SharedMemory(name=shm_name)
    # 将共享内存包装为 NumPy 数组
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    # 修改共享数据
    arr[0] = 100
    arr[-1] = 999
    print(f"[子进程] 修改后的数组: {arr}")
    # 显式关闭（不销毁共享内存）
    shm.close()

if __name__ == '__main__':
    # 创建初始数组
    arr = np.zeros(10, dtype=np.int64)
    print("[主进程] 初始数组:", arr)

    # 创建共享内存并复制数据
    shm = SharedMemory(name='SharedMemory001',create=True, size=arr.nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr[:]

    # 启动子进程
    p = Process(target=worker, args=(shm.name, arr.shape, arr.dtype))
    p.start()
    p.join()

    # 读取修改后的数据
    print("[主进程] 最终数组:", shm_arr)
    import time
    time.sleep(30)
    # 清理资源
    shm.close()
    shm.unlink()  # 标记为销毁